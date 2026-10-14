"""
densify_train.py -- Tái tạo DENSE-PARENT cho DominoSearch (paper NeurIPS'21).
Bối cảnh: pruned_fused.pt (dense parent của sparse_best) đã mất; kiến trúc deployed chỉ còn
trong sparse_best.pt (81% conv1x1 = 0). Finetune KHÔNG MASK → zeros mọc lại → dense model
đúng kiến trúc + accuracy ≥ sparse (khởi điểm 0.7976). Đây là "pre-trained dense weights"
mà DominoSearch Alg 3.1 yêu cầu làm input.
Hyps y hệt gpu_sparse_train.py (SGD 0.9/5e-4, cosine 0.01→1e-4, hyp.scratch-low, bs16, 416).
Usage: venv python densify_train.py [--epochs 15] [--bs 16]
"""
import torch, sys, warnings, argparse, math, yaml
torch.backends.cudnn.benchmark = True   # kernel autotune (input size cố định)
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr, init_seeds
from utils.loss import ComputeLoss

DATA = "data/pklot.yaml"; IMGSZ = 416
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))


def cosine_lr(optimizer, ep, total_ep, lr_max=0.01, lr_min=1e-4):
    lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * ep / total_ep))
    for g in optimizer.param_groups: g['lr'] = lr
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.003)   # 0.01 → NaN (diverge lần 1)
    ap.add_argument("--save", type=str, default="runs/sparse/dense_parent.pt")
    ap.add_argument("--init", type=str, default="runs/sparse/sparse_best.pt")
    args = ap.parse_args(); init_seeds(0)

    print(f"=== DENSIFY: sparse_best (0.7976, 81%-zero) → dense parent, {args.epochs} epochs ===")
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    print(f"init từ {args.init} (mAP50 lưu kèm: {ck.get('mAP50')})")
    model = ck["model"].float().to(DEV)
    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]
    gs = max(int(model.stride.max()), 32)

    vl, _ = create_dataloader(dd["val"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=None, augment=False, cache="disk", pad=0.5, rect=True,
                              workers=args.workers, prefix=colorstr("val: "), shuffle=False)

    def evalmap(tag):
        model.eval()
        sd_ = Path("runs/val/densify"); sd_.mkdir(parents=True, exist_ok=True)
        res, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, save_dir=sd_,
                                 imgsz=IMGSZ, batch_size=args.bs, device=DEV, workers=0,
                                 half=False, plots=False, save_json=False, verbose=False)
        print(f"  [{tag}] mAP50={res[2]:.4f} mAP50-95={res[3]:.4f}", flush=True)
        return float(res[2])

    m0 = evalmap("sparse_best baseline (phải ~0.7976 — SANITY GATE)")
    if m0 < 0.70:
        print("!!! SANITY FAIL: baseline < 0.70 — harness/model sai, DỪNG."); sys.exit(1)

    hyp_full = yaml.safe_load(open("data/hyps/hyp.scratch-low.yaml"))
    for k, v in dict(box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, fl_gamma=0.0,
                     anchor_t=4.0, label_smoothing=0.0).items():
        hyp_full.setdefault(k, v)
    model.hyp = hyp_full; model.nc = dd["nc"]; model.gr = 1.0

    tl, _ = create_dataloader(dd["train"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=hyp_full, augment=True, cache="disk", pad=0.0, rect=False,
                              workers=args.workers, prefix=colorstr("train: "), shuffle=True)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    compute_loss = ComputeLoss(model)

    best = float(ck.get('mAP50') or 0.0); gstep = 0   # resume: đừng để ep1 tệ hơn đè best cũ
    for ep in range(args.epochs):
        lr = cosine_lr(opt, ep, args.epochs, args.lr)
        model.train()
        tot = 0.0; nb = 0
        for i, (imgs, targets, _, _) in enumerate(tl):
            gstep += 1
            if gstep <= 200:   # warmup tuyến tính 200 iter (chống spike đầu — lần 1 NaN)
                for g in opt.param_groups: g['lr'] = lr * gstep / 200
            elif gstep == 201:
                for g in opt.param_groups: g['lr'] = lr
            imgs = imgs.to(DEV).float() / 255; targets = targets.to(DEV)
            loss, _ = compute_loss(model(imgs), targets)
            if not torch.isfinite(loss):
                print(f"!!! NaN/Inf loss tại ep{ep} it{i} — DỪNG (đừng đốt GPU vô ích)", flush=True)
                import sys as _s; _s.exit(2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            # KHÔNG apply_masks — zeros tự mọc lại (densify)
            tot += loss.item(); nb += 1
            if i % 200 == 0:
                print(f"    ep{ep} it{i}/{len(tl)} loss={loss.item():.3f} lr={lr:.5f}", flush=True)
        print(f"  ep{ep} avg_loss={tot/nb:.3f}")
        mE = evalmap(f"densify ep{ep+1}")
        # đo lại độ dense của conv1x1
        import torch.nn as nn
        totw = nzw = 0
        for mod in model.modules():
            if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1):
                totw += mod.weight.numel(); nzw += (mod.weight != 0).sum().item()
        print(f"  conv1x1 nonzero: {nzw/totw:.1%}")
        torch.save({"model": model, "epoch": ep, "mAP50": mE}, args.save + ".last")
        if mE > best:
            best = mE
            torch.save({"model": model, "epoch": ep, "mAP50": mE}, args.save)
            print(f"  ★ best mAP50={best:.4f} → {args.save}", flush=True)

    print(f"\n=== DENSIFY DONE: sparse 0.7976 → dense_parent {best:.4f} ===")


if __name__ == "__main__":
    main()
