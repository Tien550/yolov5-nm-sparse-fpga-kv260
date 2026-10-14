#!/usr/bin/env python3
"""YOLOv5s-Lite inference: conv1x1 via unified_acc_top (I-SPEC + layer_seq), rest CPU.

Routes 41 conv1x1 layers (Cin%16==0, Cout%4==0, INT8) to unified_acc_top LAYER_MODE=1.
Conv3x3, DW, FP32 layers (42/49/56/62) and detect-head tail (60/61/62) stay on CPU.

Weights preloaded to DMA at startup → warm FPS measures pure HW+numpy overhead.

Usage (on KV260, in /home/ubuntu/rtl_bd/):
  sudo python3 inference_unified_v4.py --image car.jpg --frames 3
  sudo python3 inference_unified_v4.py --verify          # compare each HW layer to CPU
"""
import argparse, ctypes, glob, json, mmap, os, select, sys, time
if os.geteuid() == 0:
    _us = f"/home/ubuntu/.local/lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
    if os.path.isdir(_us) and _us not in sys.path:
        sys.path.insert(0, _us)
import numpy as np

# ── Locate repo root ──────────────────────────────────────────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(_this_dir)
# Add script's own dir first (board: all files co-located in rtl_bd/),
# then tools/ subdir (PC dev mode).
sys.path.insert(0, _this_dir)
if os.path.isdir(os.path.join(ROOT, 'tools')):
    sys.path.insert(0, os.path.join(ROOT, 'tools'))
import inference_cpu as C
from cpu_conv import quant_conv, float_conv, quantize_to_int8

SPARSE_DIR = os.environ.get(
    'SPARSE_EXTRACTED',
    os.path.join(ROOT, 'yolov5/runs/quant/extracted_sparse'))
# UNI_NAME env: weights dir. Default 'weights_unified_v4' (legacy SPEC0-only). For B-w all-SPEC use
# 'weights_unified_v4_ns4all' (NSPEC=4) etc — OC_PER_KICK auto-read from that dir's layer_params.json.
UNI_DIR = os.path.join(SPARSE_DIR, os.environ.get('UNI_NAME', 'weights_unified_v4'))

# ── Register map (unified_acc_top, AXI base 0xA0000000, 16-byte stride) ──────
UNIFIED_BASE   = 0xA0000000
REG_CTRL       = 0x000
REG_AP_GIE     = 0x010   # global interrupt enable
REG_AP_IER     = 0x020   # ap_done interrupt enable
REG_AP_ISR     = 0x030   # interrupt status (W1C) — set khi ap_done, clear = ghi bit0=1
REG_CIN        = 0x040
REG_COUT       = 0x050
REG_HIN        = 0x060
REG_WIN        = 0x070
REG_N_CONFIG   = 0x080
REG_POC        = 0x090   # 0→poc=1 (4 oc/kick via ACCU_MODE=2)
REG_ACCU_MODE  = 0x0A0   # 2→4-in (sum SPEC 0..3 → 4 output channels)
REG_APPLY_RELU = 0x0B0
REG_WT_ADDR_LO = 0x0C0
REG_IN_ADDR_LO = 0x0E0
REG_OUT_ADDR_LO= 0x100
REG_STRIDE     = 0x120
REG_PAD        = 0x130
REG_LAYER_TYPE = 0x140   # 0=conv1x1, 1=conv3x3, 2=DW
REG_OUT_BYTES  = 0x150
REG_WT_BYTES   = 0x160
REG_C3_CFG     = 0x170   # Rev139: c3 9-pass cfg (write-only; read = debug seq state)
REG_ACT_TOTAL  = 0x180   # Rev139: LS_ACTL chunk count (write-only; read = debug kernel state)
REG_LAYER_MODE = 0x190   # bit0=layer_sequencer; bits[4:1]=REG_FEAT_EN
# REG_FEAT_EN bits (data bits): bit1=B-i ACT-stationary, bit2=B-c3 9-pass, bit3=fastkick
# (skip RQ reload khi keep_psum + skip writeback khi no_flush — Rev139).
# env: BI=1 bật B-i; FK=1 bật fastkick; C3=1 bật conv3x3 HW (cần weights ns12 + c3_ blobs);
#      W11=1 wide-feed 1x1 (bit5, cần BI); RR4=1 quad-writeback (bit6); RQD=1 RQ dirty-skip
#      (bit7); OW=1 out-DMA 128b/beat (bit8) — Đòn 4a' Rev141, mỗi cờ A/B độc lập.
_LM_4A = ((0x40 if os.environ.get('RR4') else 0x0) | (0x80 if os.environ.get('RQD') else 0x0)
          | (0x100 if os.environ.get('OW') else 0x0))
_LM_1X1 = (0x1 | (0x2 if os.environ.get('BI') else 0x0) | (0x8 if os.environ.get('FK') else 0x0)
           | (0x20 if (os.environ.get('W11') and os.environ.get('BI')) else 0x0) | _LM_4A)
_LM_C3  = (0x1 | 0x4 | (0x8 if os.environ.get('FK') else 0x0)
           | (0x10 if os.environ.get('C3W') else 0x0) | _LM_4A)   # bit4 = wide-ACT c3 (A/B)
_LAYER_MODE_VAL = _LM_1X1   # giữ tên cũ cho đường 1x1/DW
REG_N_OCB      = 0x1A0
REG_N_POS      = 0x1B0
REG_N_ICB      = 0x1C0
REG_WT_BLK_BYTES= 0x1D0
REG_ACT_CHUNK  = 0x1E0
AP_START = 0x01
AP_DONE  = 0x02
DMA_TOTAL= 64 * 1024 * 1024
# DMA layout (fixed base offsets)
# PIPE (pipeline 2 ảnh): mỗi process 1 vùng ACT/OUT riêng qua env; weights share (ghi trùng nội dung).
ACT_OFF  = int(os.environ.get('PIPE_ACT_OFF', '0x0000000'), 0)   # 0MB (proc2: 0x2000000=32MB)
WT_BASE  = 0x0200000   # 2MB   – conv1x1 weight blobs (preloaded)
DW_WT_BASE = 0x0800000 # 8MB   – DW weight blobs (preloaded)
OUT_OFF  = int(os.environ.get('PIPE_OUT_OFF', '0x0C00000'), 0)  # 12MB (proc2: 0x2400000=36MB)
C3_WT_BASE = 0x1000000 # 16MB  – c3 9-pass weight blobs (~4.5MB, preloaded)
# ── Đòn 4b Phase 1 (Rev144): netseq_v2 segment scheduler ──────────────────────
DESC_OFF   = 0x2000000 # 32MB  – descriptor tables (128B/layer, 2 bank ping-pong)
CH_A_OFF   = 0x2100000 # 33MB  – chain arena A (bump-alloc OUT trong segment chẵn)
CH_B_OFF   = 0x3100000 # 49MB  – chain arena B (segment lẻ — LazyOut sống thêm 1 segment)
_NS_EN = bool(os.environ.get('NS'))   # env NS=1: bật segment scheduler (cần bit Rev144+)
_LM_NS = 0x800                        # LAYER_MODE data bit11 = NS mode (feat_en[10])
_LM_OT = 0x1000                       # bit12 = OT out-transpose (feat_en[11])
DW_ACT_CHUNK = 256     # ACT tile stride for DW (4ch×54B data padded to 256B)
C3_ACT_CHUNK = 1024    # LS_ACTL chunk size for c3 plane load (dma_in_w128_addr 6-bit = 64 words = 1024B MAX!)


class HWAccel:
    """Manages unified_acc_top register map and shared DMA buffer."""
    def __init__(self):
        devmem = os.open('/dev/mem', os.O_RDWR | os.O_SYNC)
        reg_map = mmap.mmap(devmem, 0x10000, mmap.MAP_SHARED,
                            mmap.PROT_READ | mmap.PROT_WRITE, offset=UNIFIED_BASE)
        self.regs = (ctypes.c_uint32 * (0x10000 // 4)).from_buffer(reg_map)
        os.close(devmem)
        with open('/sys/module/dma_region/parameters/phys_base') as f:
            self.phys_base = int(f.read().strip())
        dma_fd = os.open('/dev/dma_region', os.O_RDWR | os.O_SYNC)
        self.dma = mmap.mmap(dma_fd, DMA_TOTAL, mmap.MAP_SHARED,
                             mmap.PROT_READ | mmap.PROT_WRITE)
        os.close(dma_fd)
        self._warmed = False
        print(f'[HW] IP @ 0x{UNIFIED_BASE:08X}  DMA phys=0x{self.phys_base:08x}')
        # PIPE=1: khoá liên-process quanh [reg-writes → kick → wait_done] mỗi layer
        self._lockf = open('/tmp/hw_unified.lock', 'w') if os.environ.get('PIPE') else None
        # INTWAIT=1: chờ AP_DONE bằng ngắt UIO (os.read nhả GIL) thay vì busy-poll (giữ GIL).
        # → thread CPU (DWASYNC…) chạy song song trong lúc HW kick. Chuỗi HLS: irq=gie&ier&isr,
        #   isr set khi ap_done, clear W1C @0x030. UIO "fabric" = irq_out→pl_ps_irq0→GIC61.
        self.uio_fd = None
        if os.environ.get('INTWAIT') and not os.environ.get('STPROF'):
            uio = None
            for d in glob.glob('/sys/class/uio/uio*'):
                try:
                    if open(d + '/name').read().strip() == 'fabric':
                        uio = '/dev/' + os.path.basename(d); break
                except OSError:
                    pass
            if uio:
                self.uio_fd = os.open(uio, os.O_RDWR)
                self.regs[REG_AP_GIE >> 2] = 1          # bật ngắt toàn cục + ap_done
                self.regs[REG_AP_IER >> 2] = 1
                self.regs[REG_AP_ISR >> 2] = 1          # clear stale
                print(f'[HW] INTWAIT qua {uio} (ngắt, nhả GIL trong wait)')
            else:
                print('[HW] INTWAIT: không thấy UIO "fabric" → giữ busy-poll')

    def wr(self, off, val):
        self.regs[off >> 2] = int(val) & 0xFFFFFFFF

    def hwlock(self):
        if self._lockf is not None:
            import fcntl; fcntl.flock(self._lockf, fcntl.LOCK_EX)
    def hwunlock(self):
        if self._lockf is not None:
            import fcntl; fcntl.flock(self._lockf, fcntl.LOCK_UN)

    def wr64(self, off_lo, addr):
        addr = int(addr)
        self.regs[off_lo >> 2]          = addr & 0xFFFFFFFF
        self.regs[(off_lo + 0x10) >> 2] = (addr >> 32) & 0xFFFFFFFF

    def dma_write(self, offset, data: bytes):
        n = len(data)
        ctypes.memmove(
            ctypes.c_char_p(ctypes.addressof(
                (ctypes.c_char * n).from_buffer(self.dma, offset))),
            data, n)

    def dma_read(self, offset, n):
        return bytes(self.dma[offset:offset+n])

    ST_NAMES = ['IDLE','WAIT_DMA','PIX_CLR','ACT_LOAD','WT_LOAD','KW_SHUF','COMPUTE',
                'FLUSH_MAC','FLUSH_RR','DONE','LD_ACT_1X1','SHUF_WAIT','WRITEBACK',
                'LOAD_RQ','WT_LOAD_W','LD_ACT_W']

    def wait_done_prof(self, timeout=30.0):
        """STPROF: poll done + sample dbg_kernel_state (0x180) → histogram cycle-attribution."""
        import collections
        h = collections.Counter()
        t0 = time.time()
        while not (self.regs[REG_CTRL >> 2] & AP_DONE):
            h[self.regs[0x180 >> 2] & 0xF] += 1
            if time.time() - t0 > timeout:
                raise RuntimeError('wait_done timeout')
        return h

    def wait_done(self, timeout=30.0):
        if self.uio_fd is not None:
            # ngắt: select+read nhả GIL → thread CPU chạy song song. select có timeout an toàn.
            r, _, _ = select.select([self.uio_fd], [], [], timeout)
            if not r:
                raise RuntimeError('TIMEOUT waiting IRQ (AP_DONE)')
            os.read(self.uio_fd, 4)                 # tiêu thụ event count
            self.regs[REG_AP_ISR >> 2] = 1          # W1C clear → deassert irq (level)
            return
        t0 = time.time()
        while not (self.regs[REG_CTRL >> 2] & AP_DONE):
            if time.time() - t0 > timeout:
                raise RuntimeError('TIMEOUT waiting AP_DONE')

    def kick(self):
        if self.uio_fd is not None:
            self.regs[REG_AP_ISR >> 2] = 1          # clear isr (line low trước khi arm)
            os.write(self.uio_fd, b'\x01\x00\x00\x00')  # arm: bật lại ngắt UIO (genirq re-enable)
        self.wr(REG_CTRL, AP_START)

    def warmup(self):
        if self._warmed:
            return
        self.kick(); self.wait_done()
        self._warmed = True


def preload_weights(hw, layer_params):
    """Load all layer weight blobs into DMA at fixed offsets. Returns wt_offsets dict."""
    wt_offsets = {}
    cur = WT_BASE
    for k, p in layer_params.items():
        blob = open(os.path.join(UNI_DIR, f'{k}.bin'), 'rb').read()
        hw.dma_write(cur, blob)
        wt_offsets[k] = cur
        cur += len(blob)
        cur = (cur + 63) & ~63   # 64B align
    total_kb = (cur - WT_BASE) // 1024
    print(f'[HW] preloaded {len(wt_offsets)} layer weight blobs ({total_kb} KB)')
    return wt_offsets


def conv1x1_hw(hw, x, k, layer_params, wt_offsets, relu, PROF):
    """Route one conv1x1 layer through unified_acc_top."""
    if _NS_EN:   # Đòn 4b: segment scheduler — build descriptor, không kick per-layer
        return _conv1x1_ns(hw, x, k, layer_params, wt_offsets, relu, PROF)
    p = layer_params[k]
    Cin  = p['Cin'];  Cout = p['Cout']
    N_IC = p['N_IC_BLKS']; N_OCB = p['N_OCB']
    WT_BLK = p['WT_BLK']; N_CFG = p['N_CFG']
    OCK  = p.get('OC_PER_KICK', 4)   # B-w: OC per kick (4 legacy SPEC0-only / 16 NS4-all / 48 NS12)

    C_dim, H, W = x.shape
    assert C_dim == Cin, f'{k}: x.shape[0]={C_dim} != Cin={Cin}'
    HW_   = H * W
    N_POS = (HW_ + 15) // 16     # ceil(H*W/16) spatial tiles
    P_pad = N_POS * 16            # padded spatial count

    # Pack activation: tile-ic order (N_POS, N_IC, 16ch, 16sp) = layer_seq DDR layout.
    # Each 256B chunk = one M-block (16 channels) x one pos-tile (16 positions).
    # act_ddr[(pos*N_IC + ic)*256 + ch*16 + sp] = x[ic*16+ch, pos*16+sp]
    t0 = time.time()
    act_buf = np.zeros((N_IC * 16, P_pad), dtype=np.int8)
    act_buf[:Cin, :HW_] = x.reshape(Cin, HW_)
    # (N_IC,16,N_POS,16) -> transpose -> (N_POS,N_IC,16,16) -> flatten
    act_bytes = act_buf.reshape(N_IC, 16, N_POS, 16).transpose(2, 0, 1, 3).reshape(-1).tobytes()
    PROF['pack'] = PROF.get('pack', 0) + time.time() - t0

    t0 = time.time()
    hw.dma_write(ACT_OFF, act_bytes)
    PROF['act_dma'] = PROF.get('act_dma', 0) + time.time() - t0

    OUT_SZ = N_OCB * N_POS * OCK * 16   # per-kick OUT tile = OCK ch × 16 pos
    hw.dma_write(OUT_OFF, bytes(OUT_SZ))   # clear output

    # Registers (LAYER_MODE written last to arm sequencer)
    hw.hwlock()   # PIPE: giữ HW từ reg-writes tới done (process kia chờ ở đây)
    hw.wr(REG_CIN,         16)      # one M-block per kick; kernel n_ic_blks=ceil(16/16)=1
    # B-w: OC per kick = NSPEC*4 (all-SPEC) or 4 (legacy SPEC0-only). From packer layer_params.
    hw.wr(REG_COUT,        p.get('OC_PER_KICK', 4))   # = requant count loaded (ST_LOAD_RQ uses reg_cout)
    hw.wr(REG_HIN,         H)
    hw.wr(REG_WIN,         W)
    hw.wr(REG_N_CONFIG,    N_CFG)
    hw.wr(REG_POC,         0)       # poc=0 (4 OC/kick via ACCU_MODE=2)
    hw.wr(REG_ACCU_MODE,   2)       # 4-in: sum SPEC 0..3
    hw.wr(REG_APPLY_RELU,  1 if relu else 0)
    hw.wr(REG_LAYER_TYPE,  0)       # conv1x1
    hw.wr64(REG_IN_ADDR_LO,  hw.phys_base + ACT_OFF)
    hw.wr64(REG_WT_ADDR_LO,  hw.phys_base + wt_offsets[k])
    hw.wr64(REG_OUT_ADDR_LO, hw.phys_base + OUT_OFF)
    hw.wr(REG_N_OCB,        N_OCB)
    hw.wr(REG_N_POS,        N_POS)
    hw.wr(REG_N_ICB,        N_IC)   # layer_seq ic-loop count = Cin//16 M-blocks per pos
    hw.wr(REG_WT_BLK_BYTES, WT_BLK)
    hw.wr(REG_ACT_CHUNK,    256)    # one M-block per ACT DMA = 16ch x 16pos = 256B
    hw.wr(REG_OUT_BYTES,    OCK * 16)   # B-w: per-kick OUT tile = OCK ch × 16 pos (→ layer_seq out_tile_bytes)
    hw.wr(REG_ACT_TOTAL,    N_POS * N_IC)   # Rev139: LS_ACTL chunk count (dùng khi BI=1)
    # DEINT-ON-HW (env DEINT + bit12=0x2000): chỉ N_POS≤169 (deint_bram 128KB: OCK*N_POS≤8192)
    _deint = bool(os.environ.get('DEINT')) and (N_POS <= 682)
    # OTV=1: bật OT (bit12=0x1000, feat_en[11]) — OUT pos-major [pos][ob][OCK][16] (nấc 1 đòn A)
    _otv = bool(os.environ.get('OTV'))
    _c3f = 0x4000 if os.environ.get('C3FAST') else 0   # Rev167 seqfast (bit14 = feat_en[13])
    _lm_c1 = _LAYER_MODE_VAL | (0x2000 if _deint else 0) | (_LM_OT if _otv else 0) | _c3f
    hw.wr(REG_LAYER_MODE,   _lm_c1)   # MUST be written last (bit0=lseq, bit1=B-i)

    # Warm-up on first ever call (hardware state unknown)
    hw.warmup()

    t0 = time.time()
    hw.kick(); hw.wait_done()
    hw.hwunlock()
    PROF['kick'] = PROF.get('kick', 0) + time.time() - t0

    # Deinterleave output: [N_OCB][N_POS][OCK][16] → (Cout, H, W)  (vectorized)
    # OCK = OC per kick (4 legacy / 16 NS4-all / 48 NS12). DDR order: ob,pos,oc_local,pos16.
    # dma_read = bytes-copy (an toàn khỏi kick kế); transpose (ascontiguousarray nhả GIL) → thread.
    t0 = time.time()
    out_raw = np.frombuffer(hw.dma_read(OUT_OFF, OUT_SZ), dtype=np.int8)
    PROF['deint'] = PROF.get('deint', 0) + time.time() - t0
    def _detile(raw=out_raw):
        if _otv:
            r = np.ascontiguousarray(
                raw.reshape(N_POS, N_OCB, OCK, 16).transpose(1, 2, 0, 3)
            ).reshape(N_OCB * OCK, P_pad)[:Cout]
        elif _deint:
            r = raw.reshape(N_OCB * OCK, P_pad)[:Cout]
        else:
            r = np.ascontiguousarray(
                raw.reshape(N_OCB, N_POS, OCK, 16).transpose(0, 2, 1, 3)
            ).reshape(N_OCB * OCK, P_pad)[:Cout]
        return r[:, :HW_].reshape(Cout, H, W)
    if os.environ.get('DTASYNC'):
        return _DetileFuture((Cout, H, W), _detile)
    t0 = time.time()
    out = _detile()
    PROF['deint'] = PROF.get('deint', 0) + time.time() - t0
    return out


# ═══════════ Đòn 4b Phase 1: segment scheduler (env NS=1, bit Rev144+) ═══════════
# conv1x1 liên tiếp = 1 SEGMENT: mỗi layer 1 descriptor 128B (ảnh reg-file 0x40-0x1E0),
# netseq_v2 trên PL tự fetch/ghi-reg/kick/chờ từng layer — host kick 1 lần/segment.
# Chuỗi nội bộ dùng OT (OUT pos-major = byte-for-byte ACT chunks layer kế, pad-OC = 0
# đã verify packer) → skip pack/deint giữa 2 conv1x1. LazyOut = kết quả còn nằm DDR;
# np.asarray(LazyOut) (mọi graph op) tự flush segment + deint.
_SEG = {'descs': [], 'flip': 0, 'arena': 0, 'last_lazy': None}

def _seg_flush(hw, PROF):
    if not _SEG['descs']:
        return
    descs = _SEG['descs']; _SEG['descs'] = []
    _SEG['last_lazy'] = None
    descs[-1][30] |= 1                      # w30 bit0 = LAST
    import struct as _st
    table = b''.join(_st.pack('<32I', *[w & 0xFFFFFFFF for w in d]) for d in descs)
    dbase = DESC_OFF + (_SEG['flip'] & 1) * 0x10000
    hw.dma_write(dbase, table)
    hw.wr64(REG_IN_ADDR_LO, hw.phys_base + dbase)      # desc_base (netseq latch tại kick)
    hw.wr(REG_LAYER_MODE, _LAYER_MODE_VAL | _LM_NS)    # NS mode — reroute ap_start → netseq
    hw.warmup()
    t0 = time.time()
    hw.kick(); hw.wait_done()
    hw.hwunlock()
    PROF['kick'] = PROF.get('kick', 0) + time.time() - t0
    PROF['ns_seg'] = PROF.get('ns_seg', 0) + 1
    PROF['ns_lay'] = PROF.get('ns_lay', 0) + len(descs)
    _SEG['flip'] += 1
    _SEG['arena'] = 0


class LazyOut:
    """Kết quả conv1x1 (layout OT pos-major) còn trong DDR — materialize khi cần numpy."""
    def __init__(self, hw, off, N_OCB, N_POS, OCK, Cout, H, W, PROF, ot=True):
        self.hw = hw; self.off = off; self.N_OCB = N_OCB; self.N_POS = N_POS
        self.OCK = OCK; self.Cout = Cout; self.H = H; self.W = W; self.PROF = PROF
        self.ot = ot; self._np = None
    @property
    def shape(self):
        return (self.Cout, self.H, self.W)
    def numpy(self):
        if self._np is None:
            _seg_flush(self.hw, self.PROF)
            t0 = time.time()
            sz = self.N_POS * self.N_OCB * self.OCK * 16
            raw = np.frombuffer(self.hw.dma_read(self.off, sz), dtype=np.int8)
            if self.ot:
                # OT: [N_POS][N_OCB*OCK][16] → (Cout_pad, P_pad) → crop
                r = np.ascontiguousarray(
                    raw.reshape(self.N_POS, self.N_OCB * self.OCK, 16).transpose(1, 0, 2)
                ).reshape(self.N_OCB * self.OCK, self.N_POS * 16)
            else:
                # legacy oc-major: [N_OCB][N_POS][OCK][16] (debug NSOT=0)
                r = np.ascontiguousarray(
                    raw.reshape(self.N_OCB, self.N_POS, self.OCK, 16).transpose(0, 2, 1, 3)
                ).reshape(self.N_OCB * self.OCK, self.N_POS * 16)
            self._np = r[:self.Cout, :self.H * self.W].reshape(self.Cout, self.H, self.W)
            self.PROF['deint'] = self.PROF.get('deint', 0) + time.time() - t0
        return self._np
    def __array__(self, dtype=None):
        a = self.numpy()
        return a.astype(dtype) if dtype is not None else a
    def __getattr__(self, name):
        # attribute lạ (astype/reshape/sum/...) → materialize + delegate sang ndarray
        # (chỉ gọi khi attribute KHÔNG có trên LazyOut — shape/numpy/... vẫn đi đường nhanh)
        return getattr(self.numpy(), name)
    def __getitem__(self, idx):
        return self.numpy()[idx]


def _conv1x1_ns(hw, x, k, layer_params, wt_offsets, relu, PROF):
    """NS mode: không kick — build descriptor, nối chuỗi, trả LazyOut."""
    p = layer_params[k]
    Cin = p['Cin']; Cout = p['Cout']
    N_IC = p['N_IC_BLKS']; N_OCB = p['N_OCB']
    WT_BLK = p['WT_BLK']; N_CFG = p['N_CFG']; OCK = p.get('OC_PER_KICK', 4)

    # ── input: chain (LazyOut CHƯA flush, cùng segment) hay pack mới ──
    _ot = os.environ.get('NSOT') != '0'
    chained = (isinstance(x, LazyOut) and _SEG['last_lazy'] is x and x._np is None and x.ot)
    if chained:
        H, W = x.H, x.W
        HW_ = H * W; N_POS = (HW_ + 15) // 16
        assert N_POS == x.N_POS, f'{k}: chain N_POS mismatch'
        cin_pad = x.N_OCB * x.OCK              # pad-OC = 0 (packer verify) → vô hại
        assert cin_pad >= Cin and cin_pad % 16 == 0
        n_ic_eff = cin_pad // 16               # đọc cả chunk pad (zeros)
        in_off = x.off
    else:
        x = np.asarray(x)                       # materialize nếu là LazyOut segment cũ
        C_dim, H, W = x.shape
        assert C_dim == Cin, f'{k}: x.shape[0]={C_dim} != Cin={Cin}'
        HW_ = H * W; N_POS = (HW_ + 15) // 16; P_pad = N_POS * 16
        t0 = time.time()
        act_buf = np.zeros((N_IC * 16, P_pad), dtype=np.int8)
        act_buf[:Cin, :HW_] = x.reshape(Cin, HW_)
        act_bytes = act_buf.reshape(N_IC, 16, N_POS, 16).transpose(2, 0, 1, 3).reshape(-1).tobytes()
        PROF['pack'] = PROF.get('pack', 0) + time.time() - t0
        t0 = time.time()
        # bump-alloc trong arena (KHÔNG dùng ACT_OFF cố định — 2 input không-chain trong
        # cùng segment sẽ đè nhau trước khi netseq chạy vì kick bị hoãn tới flush)
        arena_base0 = CH_A_OFF if (_SEG['flip'] & 1) == 0 else CH_B_OFF
        in_off = arena_base0 + _SEG['arena']
        _SEG['arena'] = (_SEG['arena'] + len(act_bytes) + 63) & ~63
        hw.dma_write(in_off, act_bytes)
        PROF['act_dma'] = PROF.get('act_dma', 0) + time.time() - t0
        n_ic_eff = N_IC

    # ── output: bump-alloc trong arena của segment (không reuse — resid an toàn) ──
    OUT_SZ = N_OCB * N_POS * OCK * 16
    arena_base = CH_A_OFF if (_SEG['flip'] & 1) == 0 else CH_B_OFF
    out_off = arena_base + _SEG['arena']
    _SEG['arena'] = (_SEG['arena'] + OUT_SZ + 63) & ~63

    # ── descriptor 32 word: word k ↔ reg 0x40+k*16 ──
    pb = hw.phys_base
    lm = _LAYER_MODE_VAL | _LM_NS | (_LM_OT if _ot else 0)   # NS giữ mọi desc (mux!)
    d = [0] * 32
    d[0] = 16;            d[1] = OCK;    d[2] = H;      d[3] = W
    d[4] = N_CFG;         d[5] = 0;      d[6] = 2;      d[7] = 1 if relu else 0
    wt = pb + wt_offsets[k];  d[8] = wt & 0xFFFFFFFF;  d[9] = wt >> 32
    ia = pb + in_off;         d[10] = ia & 0xFFFFFFFF; d[11] = ia >> 32
    oa = pb + out_off;        d[12] = oa & 0xFFFFFFFF; d[13] = oa >> 32
    d[14] = 1;            d[15] = 0;     d[16] = 0      # STRIDE/PAD/LTYPE(conv1x1)
    d[17] = OCK * 16;     d[18] = 0;     d[19] = 0      # OUT_BYTES/WT_BYTES/C3_CFG
    d[20] = N_POS * n_ic_eff                            # ACT_TOTAL (gồm chunk pad)
    d[21] = lm
    d[22] = N_OCB;        d[23] = N_POS; d[24] = n_ic_eff
    d[25] = WT_BLK;       d[26] = 256                   # ACT_CHUNK
    d[30] = 0;            d[31] = 0x4B44                # flags / magic
    _SEG['descs'].append(d)

    lz = LazyOut(hw, out_off, N_OCB, N_POS, OCK, Cout, H, W, PROF, ot=_ot)
    _SEG['last_lazy'] = lz
    if os.environ.get('NSLEN') == '1':
        lz.numpy()          # debug: flush từng lớp (segment 1 desc, không chain)
    return lz


def preload_c3_weights(hw, c3_layer_params):
    """Nạp c3 9-pass weight blobs vào DMA tại C3_WT_BASE. Trả về c3_wt_offsets."""
    c3_wt_offsets = {}
    cur = C3_WT_BASE
    for k, p in c3_layer_params.items():
        blob_path = os.path.join(UNI_DIR, f'c3_{k}.bin')
        if not os.path.exists(blob_path):
            print(f'[C3] CẢNH BÁO: {blob_path} không tồn tại, bỏ qua {k}')
            continue
        blob = open(blob_path, 'rb').read()
        hw.dma_write(cur, blob)
        c3_wt_offsets[k] = cur
        cur += len(blob)
        cur = (cur + 63) & ~63
    total_mb = (cur - C3_WT_BASE) / 1024 / 1024
    print(f'[C3] Đã nạp {len(c3_wt_offsets)} c3 layer weight blobs ({total_mb:.1f} MB)')
    return c3_wt_offsets


def conv3x3_hw(hw, x, k, c3_layer_params, c3_wt_offsets, relu, PROF):
    """B-c3 9-pass (Rev139): conv3x3 = 9 conv1x1 tap-shifted passes trên engine 12-SPEC.
    ACT = raw CHW plane trong act_uram; sequencer cấp pixel-base + pad mask per kick."""
    p = c3_layer_params[k]
    Cin = p['Cin']; Cout = p['Cout']; s = p['stride']
    NICC = p['NICC']; N_ICB = p['N_ICB']; N_OCB = p['N_OCB']
    OCK = p['OC_PER_KICK']; WT_BLK = p['WT_BLK']

    C_dim, H, W = x.shape
    assert C_dim == Cin, f'{k}: x.shape[0]={C_dim} != Cin={Cin}'
    H_out = (H - 1) // s + 1
    W_out = (W - 1) // s + 1
    ncolg = (W_out + 15) // 16
    N_POS = H_out * ncolg

    # RESACT (Rev168): chuỗi c3→c3 resident. resident_in = plane đã ở act_uram @0 (scatter từ
    # layer trước) → ACT_TOTAL=0 (LS skip ACTL). resident_out = scatter output vào act_uram @0
    # (bit12 RESOUT) + input plane ở act_bram_in (bit14 INBANK) → không DMA-out/deint/pack.
    # RESACT=1: L00→L01 (Rev168). RESACT=2: + L01 in-place lag (Rev169) → chuỗi L00→L01→L02.
    _reslvl = int(os.environ.get('RESACT') or 0)
    resident_in  = isinstance(x, _ResidentPlane)
    resident_out = (_reslvl >= 1 and k == 'layer_00') or (_reslvl >= 2 and k == 'layer_01')
    _lag = resident_out and k == 'layer_01'                # in-place: out đè plane input (lag 2 hàng)
    if resident_out:
        assert N_OCB == 1, f'{k}: RESOUT cần N_OCB=1 (scatter addr block-relative)'
        # scatter CHỈ Cout kênh thật (REG_OUT_BYTES=Cout*16 → deint_ock=Cout), plane ≤ act_uram
        assert Cout * N_POS * 16 <= 1536 * 1024, f'{k}: out plane {Cout*N_POS*16}B > act_uram 1.5MB'
        if _lag:
            assert resident_in and s == 1 and Cin == Cout and H_out == H and W_out == W, \
                f'{k}: in-place lag cần same-shape s1 + input resident'
            assert 2 * ncolg <= 31, f'{k}: lag D {2*ncolg} > 31 (slot buffer 32)'
        else:
            assert Cin * H * W <= 512 * 1024, f'{k}: input plane {Cin*H*W}B > bank 512KB'

    # ACT = raw CHW plane (không pack) → LS_ACTL copy tuyến tính vào act_uram
    t0 = time.time()
    if resident_in:
        nact = 0                                # LS skip ACTL — plane resident @0
    else:
        plane = np.ascontiguousarray(x, dtype=np.int8).tobytes()
        nact = (len(plane) + C3_ACT_CHUNK - 1) // C3_ACT_CHUNK
        hw.dma_write(ACT_OFF, plane)
    PROF['c3_pack'] = PROF.get('c3_pack', 0) + time.time() - t0

    OUT_SZ = N_OCB * N_POS * OCK * 16
    if not resident_out:
        hw.dma_write(OUT_OFF, bytes(OUT_SZ))

    # c3_cfg: [7:0]=W_in [11:8]=ncolg [16:12]=NICC [24:17]=H_out [25]=s2 [26]=mask_bot
    #         [27]=mask_right [31:28]=last_pos
    mask_bot   = 1 if ((H_out - 1) * s + 1) >= H else 0
    mask_right = 1 if ((W_out - 1) * s + 1) >= W else 0
    last_pos   = (W_out - 1) & 15
    # Rev147 C3_CFG (env C3V2=1): W_in [8:0] 9b (L00=416), ncolg [12:9], (NICC-1) [16:13]
    # Format CŨ (bit ≤146): W_in [7:0], ncolg [11:8], NICC [16:12] — mặc định.
    if os.environ.get('C3V2'):
        cfg = (W | (ncolg << 9) | ((NICC - 1) << 13) | (H_out << 17) |
               ((1 if s == 2 else 0) << 25) | (mask_bot << 26) | (mask_right << 27) |
               (last_pos << 28))
    else:
        cfg = (W | (ncolg << 8) | (NICC << 12) | (H_out << 17) |
               ((1 if s == 2 else 0) << 25) | (mask_bot << 26) | (mask_right << 27) |
               (last_pos << 28))

    hw.hwlock()   # PIPE
    hw.wr(REG_CIN,         16)      # 1 M-block/kick (n_ic_blks=1)
    hw.wr(REG_COUT,        OCK)     # requant count = 48
    hw.wr(REG_HIN,         H)       # engine c3_hw = HIN*WIN (input plane)
    hw.wr(REG_WIN,         W)
    hw.wr(REG_N_CONFIG,    p['N_CFG'])  # sparse-c3: N kept/M-block (dense=16). engine: entry
                                    # count + requant offset (N<<6) + flush loop đều theo reg_n_config
    hw.wr(REG_POC,         0)
    hw.wr(REG_ACCU_MODE,   2)
    hw.wr(REG_APPLY_RELU,  1 if relu else 0)
    hw.wr(REG_STRIDE,      16 * H * W)  # c3: REG_STRIDE tái dụng = icc-slice bytes (16 kênh × H×W)
    hw.wr(REG_LAYER_TYPE,  0)       # conv1x1 datapath
    hw.wr64(REG_IN_ADDR_LO,  hw.phys_base + ACT_OFF)
    hw.wr64(REG_WT_ADDR_LO,  hw.phys_base + c3_wt_offsets[k])
    hw.wr64(REG_OUT_ADDR_LO, hw.phys_base + OUT_OFF)
    hw.wr(REG_N_OCB,        N_OCB)
    hw.wr(REG_N_POS,        N_POS)
    hw.wr(REG_N_ICB,        N_ICB)  # 9*NICC kicks per (oc,pos)
    hw.wr(REG_WT_BLK_BYTES, WT_BLK)
    hw.wr(REG_ACT_CHUNK,    C3_ACT_CHUNK)
    # RESOUT: deint_ock = REG_OUT_BYTES/16 → chỉ scatter Cout kênh THẬT (48 kick-oc có 16 junk
    # → 48*2704 word = 2.1MB TRÀN act_uram; Cout=32 → 86528 word = 1.38MB vừa khít plane).
    hw.wr(REG_OUT_BYTES,    (Cout if resident_out else OCK) * 16)
    hw.wr(REG_C3_CFG,       cfg)
    hw.wr(REG_ACT_TOTAL,    nact)
    # Rev169: lag SUY TRONG HW (RESOUT & ~INBANK) — host chỉ set bit LAYER_MODE, KHÔNG reg riêng.
    # DEINT-ON-HW cho c3 (env DEINT + bit12): CHẾT từ Rev165 (deint_bram xóa) — giữ tương thích,
    # loại trừ với RESOUT (bit12 nghĩa mới Rev168).
    _deint = bool(os.environ.get('DEINT')) and (N_POS <= 682) and not resident_out
    # OTV3=1: OT cho c3 — OUT pos-major [H_out][ncolg][N_OCB][OCK][16] (thử de-tile nhanh hơn)
    _otv3 = bool(os.environ.get('OTV3'))
    _c3f3 = 0x4000 if os.environ.get('C3FAST') else 0   # Rev167 seqfast
    # Rev168: RESOUT (0x2000) + INBANK (0x8000, chỉ L00 — L01 lag đọc input từ act_uram)
    _resb = (0x2000 | (0 if _lag else 0x8000)) if resident_out else 0
    hw.wr(REG_LAYER_MODE,   _LM_C3 | (0x2000 if _deint else 0) | (_LM_OT if _otv3 else 0) | _c3f3 | _resb)   # viết cuối

    hw.warmup()

    t0 = time.time()
    if os.environ.get('STPROF'):
        hw.kick(); _h = hw.wait_done_prof(timeout=60.0)
        _tot = sum(_h.values()) or 1
        _top = '  '.join(f'{HWAccel.ST_NAMES[st]}={100*c/_tot:.1f}%'
                         for st, c in _h.most_common(6))
        print(f'  [STPROF {k}] samples={_tot}  {_top}')
    else:
        hw.kick(); hw.wait_done(timeout=60.0)
    hw.hwunlock()
    PROF['c3_kick'] = PROF.get('c3_kick', 0) + time.time() - t0

    if resident_out:
        # Rev168: out plane CHW [Cout][H_out][W_out] ĐÃ nằm act_uram @0 (HW scatter trong kick)
        # → không DMA-out, không de-tile; layer kế nhận sentinel + ACT_TOTAL=0.
        return _ResidentPlane((Cout, H_out, W_out))

    # De-tile: (N_OCB, H_out, ncolg, OCK, 16) → (Cout, H_out, W_out) + crop cột thừa
    t0 = time.time()
    out_raw = np.frombuffer(hw.dma_read(OUT_OFF, OUT_SZ), dtype=np.int8)
    if _otv3:
        # OT: [H_out][ncolg][N_OCB][OCK][16] → (C_pad, H_out, ncolg*16)
        result = np.ascontiguousarray(
            out_raw.reshape(H_out, ncolg, N_OCB, OCK, 16).transpose(2, 3, 0, 1, 4)
        ).reshape(N_OCB * OCK, H_out, ncolg * 16)[:Cout, :, :W_out]
    elif _deint:
        # HW đã ghi CHW [N_OCB][OCK][H_out][ncolg][16] → reshape+crop (KHÔNG transpose)
        result = out_raw.reshape(N_OCB * OCK, H_out, ncolg * 16)[:Cout, :, :W_out]
    else:
        result = np.ascontiguousarray(
            out_raw.reshape(N_OCB, H_out, ncolg, OCK, 16).transpose(0, 3, 1, 2, 4)
        ).reshape(N_OCB * OCK, H_out, ncolg * 16)[:Cout, :, :W_out]
    PROF['c3_deint'] = PROF.get('c3_deint', 0) + time.time() - t0

    return np.ascontiguousarray(result)


def preload_dw_weights(hw, dw_layer_params):
    """Nạp tất cả DW weight blobs vào DMA region tại DW_WT_BASE. Trả về dw_wt_offsets."""
    dw_wt_offsets = {}
    cur = DW_WT_BASE
    for k, p in dw_layer_params.items():
        blob_path = os.path.join(UNI_DIR, f'dw_{k}.bin')
        if not os.path.exists(blob_path):
            print(f'[DW] CẢNH BÁO: {blob_path} không tồn tại, bỏ qua {k}')
            continue
        blob = open(blob_path, 'rb').read()
        hw.dma_write(cur, blob)
        dw_wt_offsets[k] = cur
        cur += len(blob)
        cur = (cur + 63) & ~63   # align 64B
    total_kb = (cur - DW_WT_BASE) // 1024
    print(f'[DW] Đã nạp {len(dw_wt_offsets)} DW layer weight blobs ({total_kb} KB)')
    return dw_wt_offsets


class _QInt8(np.ndarray):
    """HEADQ: int8 activation mang theo qscale (output_scale của head). run_graph gọi
    quantize_to_int8(x, isc) sẽ rescale int→int (bit-exact, đọc int8) thay vì dequant→float→quantize.
    Chỉ 3 head L42/49/56 dùng; consumer duy nhất là quantize_to_int8. CPU-path (float) không ảnh hưởng."""
    def __new__(cls, arr, qscale):
        obj = np.asarray(arr, np.int8).view(cls); obj.qscale = qscale; return obj
    def __array_finalize__(self, obj):
        if obj is not None:
            self.qscale = getattr(obj, 'qscale', None)


class _ResidentPlane:
    """RESACT (Rev168): output plane ĐANG nằm trong act_uram @0 (HW scatter) — host không có
    dữ liệu, chỉ mang shape cho layer kế lập reg (HIN/WIN/Cin). Chỉ được dùng .shape."""
    def __init__(self, shape):
        self.shape = shape


class _DetileFuture:
    """DTASYNC: de-tile (transpose+crop, nhả GIL) chạy thread nền — dma_read đã copy out_raw
    nên kick layer kế ghi đè OUT_OFF vô hại. shape biết trước → .shape KHÔNG join (giữ overlap).
    Consumer numpy (cat/quantize/pad) join qua __array__. Chồng kick nhánh độc lập nhờ INTWAIT."""
    def __init__(self, shape, fn):
        import threading
        self.shape = shape                       # biết trước → .shape KHÔNG join (giữ overlap)
        self._r = [None]
        def run(): self._r[0] = fn()
        self._t = threading.Thread(target=run, daemon=True); self._t.start()
    def get(self):
        self._t.join(); return self._r[0]
    def __array__(self, dtype=None):
        r = self.get(); return r if dtype is None else r.astype(dtype)
    # Bất kỳ method/thuộc tính ndarray khác (.astype/.repeat/.reshape/.dtype…) → join rồi ủy quyền.
    # (chỉ gọi khi attribute chưa tồn tại: shape/_r/_t/get/__array__ đã có → không đệ quy.)
    def __getattr__(self, name):
        return getattr(self.get(), name)
    def __getitem__(self, idx):
        return self.get()[idx]


class _DwFuture:
    """DWASYNC: DW chạy thread nền (numpy nhả GIL) — consumer conv() sẽ .get() (join)."""
    def __init__(self, fn, *args):
        import threading
        self._r = [None]
        def run(): self._r[0] = fn(*args)
        self._t = threading.Thread(target=run, daemon=True); self._t.start()
    def get(self):
        self._t.join(); return self._r[0]
    def __array__(self, dtype=None):
        r = self.get(); return r if dtype is None else r.astype(dtype)


def dw_cpu_fast(x, w, bias, M0, sh, relu):
    """DW 3x3 s1 CPU vectorized — 9 shift-add + requant BIT-EXACT quant_conv (golden loop 55s -> ~ms)."""
    C, H, W = x.shape
    # v1 (bench board 229->173ms): pad GIỮ int8 (1/4 traffic đọc), product int16, acc int32
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    acc = np.zeros((C, H, W), np.int32)
    w16 = w[:, 0].astype(np.int16)
    for r in range(3):
        for c in range(3):
            np.add(acc, xp[:, r:r+H, c:c+W] * w16[:, r, c][:, None, None],
                   out=acc, casting='unsafe')
    acc = acc.astype(np.int64) + bias.astype(np.int64)[:, None, None]
    prod = acc * M0.astype(np.int64)[:, None, None]
    s64 = sh.astype(np.int64)[:, None, None]
    scaled = np.where(s64 > 0, (prod + (np.int64(1) << np.maximum(s64 - 1, 0))) >> s64, prod)
    if relu:
        scaled = np.where(scaled >= 0, scaled, (scaled * np.int64(205) + np.int64(1024)) >> 11)
    return np.clip(scaled, -128, 127).astype(np.int8)


def dw_hw(hw, x, k, dw_layer_params, dw_wt_offsets, relu, PROF):
    """Chạy một DW layer qua unified_acc_top swc FSM (LAYER_TYPE=2).

    ACT layout (OCB-major, layer_seq DW mode):
      act_buf[(ob*N_POS + pos)*DW_ACT_CHUNK + ch_local*54 + kh*18 + col]
    WT layout: dw_wt_offsets[k] + ob * WT_BLK  (1 blob / OCB)
    OUT layout: out_buf[(ob*N_POS + pos)*64]  → (N_OCB, N_POS, 4, 16)
    """
    p = dw_layer_params[k]
    N_CH  = p['N_CH']
    N_OCB = p['N_OCB']
    WT_BLK = p['WT_BLK']
    stride  = p.get('stride', 1)
    assert stride == 1, f'{k}: DW stride={stride} chưa hỗ trợ (cần stride=1)'

    _, H, W = x.shape
    assert N_CH == x.shape[0], f'{k}: N_CH={N_CH} != x.shape[0]={x.shape[0]}'
    NCOL_TILES = (W + 15) // 16
    N_POS = H * NCOL_TILES

    # ── Đóng gói ACT: vectorized sliding_window (thay Python loop) ──────────
    t0 = time.time()
    from numpy.lib.stride_tricks import sliding_window_view
    x_pad = np.pad(x, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)
    # Thêm padding phải để tile cuối luôn đủ 18 cột (tránh edge clip)
    W_full = NCOL_TILES * 16 + 2
    if W + 2 < W_full:
        x_pad = np.pad(x_pad, ((0, 0), (0, 0), (0, W_full - W - 2)), mode='constant')
    act_3d = np.zeros((N_OCB * N_POS, DW_ACT_CHUNK), dtype=np.int8)
    # VECTORIZED (Rev-hostopt): bỏ Python for-ob loop → 1 sliding_window_view toàn bộ kênh.
    # N_CH = N_OCB*4 (DW luôn bội 4). Bit-exact với loop cũ (giữ nguyên thứ tự trục).
    win = sliding_window_view(x_pad, (3, 18), axis=(1, 2))   # (N_CH, H, W_full-17, 3, 18)
    tiles = win[:, :, ::16].reshape(N_OCB, 4, H, NCOL_TILES, 3, 18)  # (ob, ch_local, H, NCOL, 3, 18)
    # → (ob, H, NCOL, ch_local, 3, 18) → (N_OCB*N_POS, 216); idx = ch_local*54 + kh*18 + col
    act_3d[:, :216] = np.ascontiguousarray(
        tiles.transpose(0, 2, 3, 1, 4, 5)).reshape(N_OCB * N_POS, 216)
    act_bytes = act_3d.tobytes()
    PROF['dw_pack'] = PROF.get('dw_pack', 0) + time.time() - t0

    t0 = time.time()
    hw.dma_write(ACT_OFF, act_bytes)
    PROF['dw_act_dma'] = PROF.get('dw_act_dma', 0) + time.time() - t0

    OUT_SZ = N_OCB * N_POS * 64
    hw.dma_write(OUT_OFF, bytes(OUT_SZ))

    # ── Cài registers (LAYER_MODE viết CUỐI) ────────────────────────────────
    hw.hwlock()   # PIPE
    hw.wr(REG_CIN,          N_CH)
    hw.wr(REG_COUT,         N_CH)
    hw.wr(REG_HIN,          3)
    hw.wr(REG_WIN,          18)
    hw.wr(REG_N_CONFIG,     1)           # n_cfg=1: dense (1 NZ per tap)
    hw.wr(REG_POC,          0)
    hw.wr(REG_ACCU_MODE,    0)           # 1-in mode (accu_top pass-through SPEC 0)
    hw.wr(REG_APPLY_RELU,   1 if relu else 0)
    hw.wr(REG_STRIDE,       1)
    hw.wr(REG_PAD,          0)
    hw.wr(REG_LAYER_TYPE,   2)           # DW
    hw.wr(REG_ACT_CHUNK,    DW_ACT_CHUNK)
    hw.wr(REG_WT_BLK_BYTES, WT_BLK)
    hw.wr(REG_WT_BYTES,     WT_BLK)
    hw.wr(REG_OUT_BYTES,    64)
    hw.wr64(REG_IN_ADDR_LO,  hw.phys_base + ACT_OFF)
    hw.wr64(REG_WT_ADDR_LO,  hw.phys_base + dw_wt_offsets[k])
    hw.wr64(REG_OUT_ADDR_LO, hw.phys_base + OUT_OFF)
    hw.wr(REG_N_OCB,         N_OCB)
    hw.wr(REG_N_POS,         N_POS)
    hw.wr(REG_N_ICB,         1)          # DW: không có IC loop
    hw.wr(REG_ACT_TOTAL,     N_OCB * N_POS)  # Rev139d DW B-i: LS_ACTL chunk count (khi DWBI=1)
    # DWBI gate (2026-07-03: "105 det" hóa ra do stale host file — DWBI đã minh oan, bật được).
    # Rev142: WDW (bit9, wide-feed DW — cần DWBI) + FR4 (bit10, quad-flush swc/DW).
    _lm_dw = _LAYER_MODE_VAL if os.environ.get('DWBI') else (_LAYER_MODE_VAL & ~0x2)
    if os.environ.get('WDW') and os.environ.get('DWBI'):
        _lm_dw |= 0x200
    if os.environ.get('FR4'):
        _lm_dw |= 0x400
    hw.wr(REG_LAYER_MODE,    _lm_dw)   # PHẢI viết cuối

    hw.warmup()

    t0 = time.time()
    hw.kick(); hw.wait_done()
    hw.hwunlock()
    PROF['dw_kick'] = PROF.get('dw_kick', 0) + time.time() - t0

    # ── Giải mã output: vectorized → (N_CH, H, W) ──────────────────────────
    t0 = time.time()
    out_raw = np.frombuffer(hw.dma_read(OUT_OFF, OUT_SZ), dtype=np.int8)
    # Reshape (N_OCB, H, NCOL_TILES, 4, 16) rồi chuyển vị thành (N_CH, H, NCOL_TILES*16)
    out_t = np.ascontiguousarray(
        out_raw.reshape(N_OCB, H, NCOL_TILES, 4, 16).transpose(0, 3, 1, 2, 4))
    result = np.ascontiguousarray(out_t.reshape(N_CH, H, NCOL_TILES * 16)[:, :, :W])
    PROF['dw_deint'] = PROF.get('dw_deint', 0) + time.time() - t0

    return result


def make_runner(meta, req, weights, bias, hw, layer_params, wt_offsets,
                dw_layer_params=None, dw_wt_offsets=None,
                c3_layer_params=None, c3_wt_offsets=None,
                maxlayers=999, verify=False):
    # HEADQ=1: PTQ head L42/49/56 chạy INT8 trên HW (req gộp từ requant_params_headq.json).
    if os.environ.get('HEADQ'):
        req = json.load(open(os.path.join(SPARSE_DIR, 'requant_params_headq.json')))
        print('[HEADQ] head L42/49/56 -> INT8 HW (PTQ)', flush=True)
    # Đòn 4b: --verify cần materialize từng lớp để so CPU → tắt NS trong phiên verify
    global _NS_EN
    if verify and _NS_EN and not os.environ.get('NSVERIFY'):
        print('[NS] verify mode → tắt segment scheduler (per-layer kick để so CPU)')
        _NS_EN = False
    osc = lambda lid: meta[f"layer_{lid:02d}"]["output_scale"]
    isc = lambda lid: meta[f"layer_{lid:02d}"]["input_scale"]
    PROF = {}; n_hw = [0]; n_dw_hw = [0]
    _dw_params  = dw_layer_params  or {}
    _dw_offsets = dw_wt_offsets    or {}
    _c3_params  = c3_layer_params  or {}
    _c3_offsets = c3_wt_offsets    or {}
    _prev_exit  = [None]   # timestamp khi conv() vừa kết thúc (để đo khoảng giữa)

    def conv(lid, x, relu=True):
        # Đo thời gian "giữa" hai conv() calls liên tiếp (graph ops, Python overhead)
        t_enter = time.time()
        if _prev_exit[0] is not None:
            PROF['between'] = PROF.get('between', 0) + (t_enter - _prev_exit[0])

        k = f"layer_{lid:02d}"; m = meta[k]
        ws = m["weight_shape"]; stride = m["stride"][0]; pad = m["padding"][0]
        isdw = bool(m["is_depthwise"])

        # Đòn 4b: LazyOut chỉ được giữ khi layer này đi đường conv1x1-HW (NS chain);
        # mọi đường khác (DW/c3/CPU/FP32) cần numpy → materialize (tự flush segment).
        if isinstance(x, LazyOut):
            _c1 = (k in req) and (not isdw) and ws[2] == 1 and (k in layer_params)
            if not (_NS_EN and _c1):
                x = x.numpy()

        # FP32 layers (42/49/56/62): must stay float
        if k not in req:
            t0 = time.time()
            out = float_conv(x, weights[k], bias[k],
                             m["input_scale"], m["weight_scale_per_channel"],
                             stride=stride, pad=pad,
                             is_depthwise=isdw, apply_relu=(lid != 62))
            dt = time.time() - t0
            PROF['fp32_cpu'] = PROF.get('fp32_cpu', 0) + dt
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} FP32 {tuple(out.shape)} {dt*1e3:6.1f}ms")
            _prev_exit[0] = time.time()
            return out

        r = relu and (lid not in C.FP32_NO_LEAKY)
        _headq_deq = bool(os.environ.get('HEADQ')) and lid in (42, 49, 56)
        if _headq_deq:
            r = relu   # head neck conv CÓ LeakyReLU trong float path — giữ đúng khi INT8
        M0 = np.array(req[k]["M0"], np.uint16)
        sh = np.array(req[k]["shift"], np.uint8)
        b  = bias[k]; w = weights[k]

        # DWCPU=1: DW chạy CPU (quant_conv bit-exact golden) — HW DW engine kick 17.5ms/layer
        # quá đắt cho 15 MMAC; CPU vectorized rẻ hơn + nhả khoá HW cho pipeline.
        if isinstance(x, _DwFuture):
            x = x.get()                      # consumer của DW-async: join thread

        use_dw_hw  = isdw and (k in _dw_params) and (k in _dw_offsets)                      and not os.environ.get('DWCPU') and not os.environ.get('DWASYNC')
        if isdw and os.environ.get('DWASYNC'):
            return _DwFuture(dw_cpu_fast, np.asarray(x), w, b, M0, sh, r)
        if isdw and not use_dw_hw and os.environ.get('DWCPU'):
            t0 = time.time()
            out = dw_cpu_fast(np.asarray(x), w, b, M0, sh, r)
            PROF['dw_cpu'] = PROF.get('dw_cpu', 0) + time.time() - t0
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} DW-CPU {tuple(out.shape)} {(time.time()-t0)*1e3:6.1f}ms")
            return out
        use_c1x1_hw = (not isdw) and (n_hw[0] < maxlayers) and (k in layer_params)
        use_c3_hw  = (not isdw) and ws[2] == 3 and (k in _c3_params) and (k in _c3_offsets)

        if use_c3_hw:
            t0 = time.time()
            out = conv3x3_hw(hw, x, k, _c3_params, _c3_offsets, r, PROF)
            dt = time.time() - t0
            PROF['c3_hw']   = PROF.get('c3_hw',   0) + dt
            PROF['c3_hw_n'] = PROF.get('c3_hw_n', 0) + 1
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} C3-HW {tuple(out.shape)} Cin={ws[1]} s={stride} {dt*1e3:6.1f}ms")
            if verify:
                ref = quant_conv(x, w, b, M0, sh, stride=stride, pad=pad,
                                 is_depthwise=False, apply_relu=r)
                d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                print(f"  L{lid:02d} C3 {str(tuple(out.shape)):16s} HW vs CPU "
                      f"within1={100*np.mean(d<=1):.1f}% max={d.max()} ({dt*1e3:.1f}ms)")
        elif use_dw_hw:
            n_dw_hw[0] += 1
            t0 = time.time()
            out = dw_hw(hw, x, k, _dw_params, _dw_offsets, r, PROF)
            dt = time.time() - t0
            PROF['dw_hw']   = PROF.get('dw_hw',   0) + dt
            PROF['dw_hw_n'] = PROF.get('dw_hw_n', 0) + 1
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} DW-HW {tuple(out.shape)} {dt*1e3:6.1f}ms")
            if verify:
                ref = quant_conv(x, w, b, M0, sh, stride=stride, pad=pad,
                                 is_depthwise=True, apply_relu=r)
                d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                print(f"  L{lid:02d} DW {str(tuple(out.shape)):16s} HW vs CPU "
                      f"within1={100*np.mean(d<=1):.1f}% max={d.max()} ({dt*1e3:.1f}ms)")
        elif use_c1x1_hw:
            n_hw[0] += 1
            t0 = time.time()
            out = conv1x1_hw(hw, x, k, layer_params, wt_offsets, r, PROF)
            dt = time.time() - t0
            PROF['c1x1_hw']   = PROF.get('c1x1_hw',   0) + dt
            PROF['c1x1_hw_n'] = PROF.get('c1x1_hw_n', 0) + 1
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} HW  {tuple(out.shape)} Cin={ws[1]} {dt*1e3:6.1f}ms")
            if verify:
                ref = quant_conv(x, w, b, M0, sh, stride=stride, pad=pad,
                                 is_depthwise=False, apply_relu=r)
                d = np.abs(out.astype(np.int16) - ref.astype(np.int16))
                print(f"  L{lid:02d} {str(tuple(out.shape)):16s} HW vs CPU "
                      f"within1={100*np.mean(d<=1):.1f}% max={d.max()} ({dt*1e3:.1f}ms)")
            if _headq_deq:
                # HEADQ opt: KHÔNG dequant về float. Giữ int8 + qscale=output_scale (np.float32
                # để khớp bit-exact công thức cũ). Consumer quantize_to_int8 rescale int→int.
                out = _QInt8(out, np.float32(m['output_scale']))
        else:
            t0 = time.time()
            out = quant_conv(x, w, b, M0, sh, stride=stride, pad=pad,
                             is_depthwise=isdw, apply_relu=r)
            dt = time.time() - t0
            cat = 'dw' if isdw else ('c1x1' if ws[2] == 1 else 'c3x3')
            PROF[cat + '_cpu'] = PROF.get(cat + '_cpu', 0) + dt
            if os.environ.get('PERLAYER'):
                print(f"  L{lid:02d} CPU {tuple(out.shape)} Cin={ws[1]} {dt*1e3:6.1f}ms")
            if _headq_deq:
                # BUG FIX (2026-07-09): nhánh CPU fallback (maxlayers < tổng lớp, hoặc
                # --maxlayers giới hạn) THIẾU bọc _QInt8 → quantize_to_int8 downstream đọc
                # nhầm int8 thô làm float, chia cho scale → giá trị rác → sai lệch det
                # (106→129 khi maxlayers<46 + HEADQ=1). Bọc giống hệt nhánh HW phía trên.
                out = _QInt8(out, np.float32(m['output_scale']))

        # BUG FIX (Rev150): KHÔNG dequant ở đây. FP32_NO_LEAKY = {42,49,56,60,61,62};
        # 42/49/56/62 đã return sớm ở nhánh float_conv (k not in req) nên chỉ 60/61 (detect
        # head có requant) tới đây. run_graph LUÔN gọi dequant(out_small_i/out_medium_i, osc)
        # cho 60/61 → nếu ×osc ở đây nữa = DOUBLE-DEQUANT (×osc²) → HW nhỏ 1/osc lần
        # (5.7×@L60, 7.7×@L61 = 1/osc chính xác) → detect head sai → over-detect. inference_cpu
        # trả int8 thẳng (không ×osc) nên đúng. Ở đây cũng trả int8 (conv1x1_hw) → run_graph dequant 1 lần.
        _prev_exit[0] = time.time()
        return out

    return conv, osc, isc, PROF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image',  default=None)
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--maxlayers', type=int, default=999)
    ap.add_argument('--save',   default='')
    args = ap.parse_args()

    # RESACT (Rev168) loại trừ: --verify (cần out/x numpy để so CPU) + PIPE (act_uram là state
    # chia sẻ trên HW — process khác chen giữa L00→L01 sẽ đè plane resident).
    if os.environ.get('RESACT') and (args.verify or os.environ.get('PIPE')):
        print('[!] RESACT tắt (không tương thích --verify/PIPE)')
        del os.environ['RESACT']

    print('[*] Đang nạp model...')
    C.E = SPARSE_DIR
    meta, req, weights, bias = C.load_model()

    print('[*] Đang nạp layer params...')
    layer_params = json.load(open(os.path.join(UNI_DIR, 'layer_params.json')))
    print(f'    {len(layer_params)} layer conv1x1 chạy trên HW')

    dw_params_path = os.path.join(UNI_DIR, 'dw_layer_params.json')
    if os.path.exists(dw_params_path):
        dw_layer_params = json.load(open(dw_params_path))
        print(f'    {len(dw_layer_params)} layer DW chạy trên HW')
    else:
        dw_layer_params = {}
        print('    [CẢNH BÁO] dw_layer_params.json không có → DW chạy CPU')

    # B-c3 9-pass (Rev139): chỉ bật khi env C3=1 VÀ có c3_layer_params.json
    c3_params_path = os.path.join(UNI_DIR, 'c3_layer_params.json')
    if os.environ.get('C3') and os.path.exists(c3_params_path):
        c3_layer_params = json.load(open(c3_params_path))
        print(f'    {len(c3_layer_params)} layer conv3x3 chạy trên HW (9-pass)')
    else:
        c3_layer_params = {}
        if os.environ.get('C3'):
            print('    [CẢNH BÁO] C3=1 nhưng c3_layer_params.json không có → c3x3 chạy CPU')

    print('[*] Đang mở HW...')
    hw = HWAccel()
    wt_offsets = preload_weights(hw, layer_params)
    dw_wt_offsets = preload_dw_weights(hw, dw_layer_params) if dw_layer_params else {}
    c3_wt_offsets = preload_c3_weights(hw, c3_layer_params) if c3_layer_params else {}

    if args.image:
        from PIL import Image
        im = Image.open(args.image).convert('RGB').resize((C.IMG_SIZE, C.IMG_SIZE))
        arr = np.asarray(im, np.float32).transpose(2, 0, 1) / 255.0
        img = np.round(arr / C.INPUT_SCALE).clip(-128, 127).astype(np.int8)
    else:
        shp = json.load(open(os.path.join(ROOT, 'golden/layer_shapes.json')))['layer_00']['in']
        img = np.fromfile(os.path.join(ROOT, 'golden/L00_in_CHW.bin'), np.int8).reshape(shp)
    print(f'[*] Input {img.shape}  maxlayers={args.maxlayers}')

    outs = None
    for fr in range(args.frames):
        conv, osc, isc, PROF = make_runner(meta, req, weights, bias,
                                           hw, layer_params, wt_offsets,
                                           dw_layer_params, dw_wt_offsets,
                                           c3_layer_params, c3_wt_offsets,
                                           args.maxlayers, args.verify)
        t0 = time.time()
        outs = C.run_graph(img, conv, osc, isc)
        fwd = time.time() - t0
        tag = 'lạnh' if fr == 0 else 'nóng'
        print(f'[*] frame {fr} ({tag}) forward {fwd:.2f}s  '
              f'scales={[o.shape for o in outs]}')
        c1_t  = PROF.get('c1x1_hw', 0);  c1_n  = PROF.get('c1x1_hw_n', 0)
        dw_t  = PROF.get('dw_hw',   0);  dw_n  = PROF.get('dw_hw_n',   0)
        kick1 = PROF.get('kick', 0);     dkick = PROF.get('dw_kick', 0)
        print(f'    HW conv1x1: {c1_t:.2f}s ({c1_n} layer)  kick={kick1:.2f}s'
              f'  pack+actdma={PROF.get("pack",0)+PROF.get("act_dma",0):.2f}s'
              f'  deint={PROF.get("deint",0):.2f}s')
        print(f'    HW DW:      {dw_t:.2f}s ({dw_n} layer)  kick={dkick:.2f}s'
              f'  pack={PROF.get("dw_pack",0):.2f}s'
              f'  actdma={PROF.get("dw_act_dma",0):.2f}s'
              f'  deint={PROF.get("dw_deint",0):.2f}s')
        c3h_t = PROF.get('c3_hw', 0); c3h_n = PROF.get('c3_hw_n', 0)
        if c3h_n:
            print(f'    HW c3x3:    {c3h_t:.2f}s ({c3h_n} layer)  kick={PROF.get("c3_kick",0):.2f}s'
                  f'  plane-dma={PROF.get("c3_pack",0):.2f}s'
                  f'  deint={PROF.get("c3_deint",0):.2f}s')
        cpu_c3 = PROF.get('c3x3_cpu', 0); cpu_dw = PROF.get('dw_cpu', 0)
        cpu_c1 = PROF.get('c1x1_cpu', 0); fp32 = PROF.get('fp32_cpu', 0)
        between = PROF.get('between', 0)
        total_prof = (PROF.get('c1x1_hw',0) + PROF.get('dw_hw',0) + PROF.get('c3_hw',0) +
                      cpu_c3 + cpu_dw + cpu_c1 + fp32)
        print(f'    CPU: c3x3={cpu_c3:.2f}s  dw-cpu={cpu_dw:.2f}s  '
              f'c1x1(đuôi)={cpu_c1:.2f}s  FP32={fp32:.2f}s')
        print(f'    Giữa các conv()={between:.2f}s  tổng-prof={total_prof:.2f}s  '
              f'không-giải-thích={fwd-total_prof-between:.2f}s')
        fps = 1.0 / fwd if fwd > 0 else 0
        print(f'    FPS={fps:.2f}')
        if os.environ.get('PROFDUMP'):
            print('    PROF: ' + '  '.join(f'{k}={v:.3f}' if isinstance(v,float) else f'{k}={v}'
                                           for k,v in sorted(PROF.items())))

    td = time.time(); dets = C.decode(outs); print(f'[*] decode/NMS {time.time()-td:.2f}s')
    print(f'[*] {len(dets)} phát hiện')
    for bx in dets[:15]:
        print(f'    cls={C.CLASS_NAMES[int(bx[5])]:14s} conf={bx[4]:.3f} '
              f'box=[{bx[0]:.0f},{bx[1]:.0f},{bx[2]:.0f},{bx[3]:.0f}]')

    if args.save and args.image:
        from PIL import Image, ImageDraw
        vis = Image.open(args.image).convert('RGB').resize((C.IMG_SIZE, C.IMG_SIZE))
        dr = ImageDraw.Draw(vis); col = {0: (0, 200, 0), 1: (220, 0, 0)}
        for bx in dets:
            x1,y1,x2,y2,cf,cl = bx[0],bx[1],bx[2],bx[3],bx[4],int(bx[5])
            dr.rectangle([x1,y1,x2,y2], outline=col.get(cl,(128,128,128)), width=2)
        vis.save(args.save); print(f'[*] saved {args.save}')


if __name__ == '__main__':
    main()
