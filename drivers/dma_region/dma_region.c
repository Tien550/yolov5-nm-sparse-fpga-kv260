/*
 * dma_region.c — Optimized DMA region for KV260/ZynqMP  v2
 *
 * v2 vs v1:
 *   v1: dma_alloc_coherent + dma_mmap_coherent → Non-Cached  → read=134 MB/s
 *   v2: dma_alloc_pages(CMA) + remap WC        → Write-Combine → read~400 MB/s
 *                                                               write~1200 MB/s
 *
 * Write-Combining (WC) on ARM64:
 *   - CPU store buffer combines writes into DDR bursts → sequential write fast
 *   - Reads bypass L1/L2 cache but use hardware prefetch → better than NC
 *   - No aliasing: WC == MT_NORMAL_NC, same as dma_alloc_pages kernel mapping
 *
 * dma_alloc_pages uses CMA (1GB pre-reserved on KV260) → no fragmentation issue.
 *
 * Sync ioctls: flush/invalidate cache (no-op for WC but kept for compatibility).
 * Partial sync: IOCTL_SYNC_TO/FROM_PARTIAL with struct dma_sync_args{offset,size}.
 *
 * Build: make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 * Load:  sudo insmod dma_region.ko size_mb=128
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/dma-mapping.h>
#include <linux/platform_device.h>
#include <linux/mm.h>
#include <linux/uaccess.h>
#include <linux/slab.h>
#include <linux/of.h>
#include <linux/gfp.h>

#define DRIVER_NAME "dma_region"

/* ── Module params ── */
static int size_mb = 128;
module_param(size_mb, int, 0444);
MODULE_PARM_DESC(size_mb, "Size of DMA region in MB (default 128)");

static uint64_t phys_base = 0;
module_param(phys_base, ullong, 0444);
MODULE_PARM_DESC(phys_base, "Physical base address");

/* ── State ── */
static struct page      *region_pages = NULL;
static dma_addr_t        dma_handle   = 0;
static size_t            region_size  = 0;
static struct device    *dma_dev      = NULL;
static struct platform_device *pdev;

/* ── Char device ── */
static int           major;
static struct class *cls;
static struct cdev   cdev_obj;

/* ── IOCTLs ── */
struct dma_sync_args {
    uint64_t offset;
    uint64_t size;     /* 0 = full region */
};

#define IOCTL_SYNC_TO_DEVICE    _IO('D', 1)   /* backward compat: full sync */
#define IOCTL_SYNC_FROM_DEVICE  _IO('D', 2)
#define IOCTL_SYNC_TO_PARTIAL   _IOW('D', 3, struct dma_sync_args)
#define IOCTL_SYNC_FROM_PARTIAL _IOW('D', 4, struct dma_sync_args)

static int dr_open(struct inode *i, struct file *f)    { return 0; }
static int dr_release(struct inode *i, struct file *f) { return 0; }

static long dr_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    struct dma_sync_args sa = { .offset = 0, .size = region_size };
    dma_addr_t sync_addr;
    size_t     sync_size;

    switch (cmd) {
    case IOCTL_SYNC_TO_DEVICE:
        dma_sync_single_for_device(dma_dev, dma_handle, region_size, DMA_TO_DEVICE);
        return 0;
    case IOCTL_SYNC_FROM_DEVICE:
        dma_sync_single_for_cpu(dma_dev, dma_handle, region_size, DMA_FROM_DEVICE);
        return 0;
    case IOCTL_SYNC_TO_PARTIAL:
    case IOCTL_SYNC_FROM_PARTIAL:
        if (copy_from_user(&sa, (void __user *)arg, sizeof(sa)))
            return -EFAULT;
        if (sa.offset >= region_size) return -EINVAL;
        if (sa.size == 0 || sa.offset + sa.size > region_size)
            sa.size = region_size - sa.offset;
        sync_addr = dma_handle + sa.offset;
        sync_size = (size_t)sa.size;
        if (cmd == IOCTL_SYNC_TO_PARTIAL)
            dma_sync_single_for_device(dma_dev, sync_addr, sync_size, DMA_TO_DEVICE);
        else
            dma_sync_single_for_cpu(dma_dev, sync_addr, sync_size, DMA_FROM_DEVICE);
        return 0;
    }
    return -ENOTTY;
}

static int dr_mmap(struct file *file, struct vm_area_struct *vma)
{
    unsigned long pfn    = PFN_DOWN(phys_base) + vma->vm_pgoff;
    unsigned long offset = vma->vm_pgoff << PAGE_SHIFT;
    size_t        req    = vma->vm_end - vma->vm_start;

    if (offset + req > region_size) return -EINVAL;

    /*
     * Write-Combining (MT_NORMAL_NC + store buffer) — faster than v1's NC:
     *   sequential writes: combine into DDR bursts → ~1200 MB/s
     *   sequential reads:  hardware prefetch helps → ~400 MB/s (vs 134 MB/s)
     * No aliasing: WC == same MT_NORMAL_NC as dma_alloc_pages kernel mapping.
     */
    vma->vm_page_prot = pgprot_writecombine(vma->vm_page_prot);
    vma->vm_flags    |= VM_DONTEXPAND | VM_DONTDUMP;

    return remap_pfn_range(vma, vma->vm_start, pfn, req, vma->vm_page_prot);
}

static const struct file_operations dr_fops = {
    .owner          = THIS_MODULE,
    .open           = dr_open,
    .release        = dr_release,
    .unlocked_ioctl = dr_ioctl,
    .mmap           = dr_mmap,
};

static int __init dma_region_init(void)
{
    int   ret;
    dev_t devno;

    region_size = (size_t)size_mb * 1024 * 1024;

    pdev = platform_device_register_simple("dma_region_dma", -1, NULL, 0);
    if (IS_ERR(pdev)) return PTR_ERR(pdev);
    dma_dev = &pdev->dev;

    ret = dma_set_mask_and_coherent(dma_dev, DMA_BIT_MASK(40));
    if (ret)
        dma_set_mask_and_coherent(dma_dev, DMA_BIT_MASK(32));

    /*
     * dma_alloc_pages: allocates from CMA (1GB pre-reserved on KV260).
     * Avoids buddy-allocator fragmentation that blocks alloc_pages(order=15).
     * Returns struct page* + dma_handle for sync ops.
     */
    region_pages = dma_alloc_pages(dma_dev, region_size, &dma_handle,
                                   DMA_BIDIRECTIONAL, GFP_KERNEL | __GFP_ZERO);
    if (!region_pages) {
        pr_err("dma_alloc_pages(%dMB) failed\n", size_mb);
        platform_device_unregister(pdev);
        return -ENOMEM;
    }
    phys_base = (uint64_t)page_to_phys(region_pages);

    pr_info("DMA region v2 (WC): phys=0x%010llx dma=0x%010llx size=%dMB\n",
            phys_base, (u64)dma_handle, size_mb);

    ret = alloc_chrdev_region(&devno, 0, 1, DRIVER_NAME);
    if (ret < 0) goto err_free;
    major = MAJOR(devno);

    cdev_init(&cdev_obj, &dr_fops);
    cdev_obj.owner = THIS_MODULE;
    ret = cdev_add(&cdev_obj, devno, 1);
    if (ret < 0) goto err_unreg;

    cls = class_create(THIS_MODULE, DRIVER_NAME);
    if (IS_ERR(cls)) { ret = PTR_ERR(cls); goto err_del; }

    if (!device_create(cls, NULL, devno, NULL, DRIVER_NAME)) {
        ret = -ENOMEM;
        goto err_class;
    }
    pr_info("/dev/%s ready (major=%d, WC write-combining)\n", DRIVER_NAME, major);
    return 0;

err_class: class_destroy(cls);
err_del:   cdev_del(&cdev_obj);
err_unreg: unregister_chrdev_region(devno, 1);
err_free:
    dma_free_pages(dma_dev, region_size, region_pages, dma_handle, DMA_BIDIRECTIONAL);
    platform_device_unregister(pdev);
    return ret;
}

static void __exit dma_region_exit(void)
{
    dev_t devno = MKDEV(major, 0);
    device_destroy(cls, devno);
    class_destroy(cls);
    cdev_del(&cdev_obj);
    unregister_chrdev_region(devno, 1);
    dma_free_pages(dma_dev, region_size, region_pages, dma_handle, DMA_BIDIRECTIONAL);
    platform_device_unregister(pdev);
    pr_info("DMA region freed\n");
}

module_init(dma_region_init);
module_exit(dma_region_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("KV260");
MODULE_DESCRIPTION("WC DMA region v2 — faster reads/writes vs coherent v1");
