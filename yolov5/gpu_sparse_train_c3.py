"""
gpu_sparse_train_c3.py -- Finetune sparse MỞ RỘNG sang conv3x3 (DominoSearch bước 2, đợt 2).
Init = sparse_lw_best.pt (0.8383, c1x1 đã N per-layer). Thêm mask c3 theo runs/sens/c3_scheme.json
(model.6/8/19/22 N=4; model.1/2/4 giữ dense N=16 — bimodal từ c3_sens_scan).
Mask c3 khớp HW drain-N: per M-block Cin, top-N theo Σ|w| qua TOÀN BỘ oc VÀ 9 tap
(idx CHIA SẺ qua tap — HW chỉ cần 1 idx list/M-block như c1x1).
c1x1 mask XÂY LẠI TỪ idx_map ĐÃ LƯU trong checkpoint (không re-derive — giữ đúng pattern deploy).
Output: runs/sparse_c3/sparse_c3_best.pt (+ c3_idx_map cho packer).
Usage: venv python gpu_sparse_train_c3.py [--epochs 25] [--bs 32]
"""
import torch, sys, warnings, argparse, math, json, yaml
torch.backends.cudnn.benchmark = True
import numpy as np
import torch.nn as nn
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr, init_seeds
from utils.loss import ComputeLoss

DATA = "data/pklot.yaml"; M = 16
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEV}" + (f" ({torch.cuda.get_device_name(0)})" if DEV == "cuda" else ""))


def masks_c1_from_idx(model, idx_map):
    """Mask c1x1 dựng lại từ idx_map checkpoint (giữ ĐÚNG pattern đã train/deploy)."""
    masks = {}
    for name, mod in model.named_modules():
        if name not in idx_map: continue
        w = mod.weight.data
        mask = torch.zeros_like(w.reshape(w.shape[0], w.shape[1]))
        for mblk, keep in enumerate(idx_map[name]):
            for i in keep: mask[:, mblk*M + i] = 1.0
        masks[name] = mask.reshape(w.shape).to(w.device)
    return masks


def masks_c3(model, scheme):
    """Mask c3: per M-block Cin, top-N theo Σ|w| (oc × 9 tap), idx chia sẻ tap."""
    masks = {}; idx_map = {}
    for name, mod in model.named_modules():
        if name not in scheme: continue
        N = int(scheme[name])
        w = mod.weight.data                       # [Cout,Cin,3,3]
        if N >= M: continue                        # dense — bỏ qua
        assert isinstance(mod, nn.Conv2d) and mod.kernel_size == (3, 3) and mod.groups == 1
        Cin = w.shape[1]; assert Cin % M == 0
        score = w.abs().sum(dim=(0, 2, 3))         # [Cin]
        mask = torch.zeros_like(w); layer_idx = []
        for mblk in range(Cin // M):
            top = score[mblk*M:(mblk+1)*M].topk(N).indices.sort().values
            for i in top: mask[:, mblk*M + i] = 1.0
            layer_idx.append(top.tolist())
        masks[name] = mask.to(w.device)
        idx_map[name] = layer_idx
    return masks, idx_map


def apply_masks(model, masks):
    for name, mod in model.named_modules():
        if name in masks: mod.weight.data.mul_(masks[name])


def cosine_lr(opt, ep, total, lr_max, lr_min=1e-4):
    lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * ep / total))
    for g in opt.param_groups: g['lr'] = lr
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.005)   # thấp hơn 0.01: init đã hội tụ, prune nhỏ
    ap.add_argument("--c3scheme", type=str, default="runs/sens/c3_scheme.json")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--init", type=str, default="runs/sparse_lw/sparse_lw_best.pt")
    ap.add_argument("--save", type=str, default="runs/sparse_c3/sparse_c3_best.pt")
    ap.add_argument("--best_init", type=float, default=0.0,
                    help="sàn best khi tiếp tục từ checkpoint đã tốt (không ghi đè bằng epoch tệ)")
    args = ap.parse_args(); init_seeds(0)

    c3sch = json.loads(Path(args.c3scheme).read_text())
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = ck["model"].float().to(DEV)
    c1sch, c1idx = ck["scheme"], ck["idx_map"]
    print(f"init: {args.init} mAP50={ck.get('mAP50'):.4f}")
    print(f"c1x1 scheme: {len(c1sch)} layers | c3 scheme: "
          f"{ {k: v for k, v in sorted(c3sch.items())} }")

    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]
    gs = max(int(model.stride.max()), 32)
    IMGSZ = args.imgsz

    vl, _ = create_dataloader(dd["val"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=None, augment=False, cache="disk", pad=0.5, rect=True,
                              workers=args.workers, prefix=colorstr("val: "), shuffle=False)

    def evalmap(tag):
        model.eval()
        sd_ = Path("runs/val/sparse_c3"); sd_.mkdir(parents=True, exist_ok=True)
        res, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, save_dir=sd_,
                                 imgsz=IMGSZ, batch_size=args.bs, device=DEV, workers=0,
                                 half=False, plots=False, save_json=False, verbose=False)
        print(f"  [{tag}] mAP50={res[2]:.4f} mAP50-95={res[3]:.4f}", flush=True)
        return float(res[2])

    masks = masks_c1_from_idx(model, c1idx)
    mc3, c3idx = masks_c3(model, c3sch)
    masks.update(mc3)
    apply_masks(model, masks)
    m0 = evalmap("one-shot +c3-pruned")

    hyp = yaml.safe_load(open("data/hyps/hyp.scratch-low.yaml"))
    for k, v in dict(box=0.05, cls=0.5, cls_pw=1.0, obj=1.0, obj_pw=1.0, fl_gamma=0.0,
                     anchor_t=4.0, label_smoothing=0.0).items():
        hyp.setdefault(k, v)
    model.hyp = hyp; model.nc = dd["nc"]; model.gr = 1.0
    tl, _ = create_dataloader(dd["train"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=hyp, augment=True, cache="disk", pad=0.0, rect=False,
                              workers=args.workers, prefix=colorstr("train: "), shuffle=True)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    compute_loss = ComputeLoss(model)

    save_dir = Path("runs/sparse_c3"); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "c3_idx_map.json").write_text(json.dumps(c3idx))
    best = args.best_init
    if best > 0: print(f"  best floor = {best:.4f} (chỉ lưu khi vượt)")
    for ep in range(args.epochs):
        lr = cosine_lr(opt, ep, args.epochs, args.lr)
        model.train()
        tot = 0.0; nb = 0
        for i, (imgs, targets, _, _) in enumerate(tl):
            imgs = imgs.to(DEV).float() / 255; targets = targets.to(DEV)
            loss, _ = compute_loss(model(imgs), targets)
            if not torch.isfinite(loss):
                print(f"!!! NaN/Inf tại ep{ep} it{i} — DỪNG", flush=True); sys.exit(2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            apply_masks(model, masks)          # giữ pattern c1x1 CŨ + c3 MỚI (static)
            tot += loss.item(); nb += 1
            if i % 200 == 0:
                print(f"    ep{ep} it{i}/{len(tl)} loss={loss.item():.3f} lr={lr:.5f}", flush=True)
        print(f"  ep{ep} avg_loss={tot/nb:.3f}")
        mE = evalmap(f"c3 ep{ep+1}")
        if mE > best:
            best = mE
            torch.save({"model": model, "epoch": ep, "mAP50": mE,
                        "idx_map": c1idx, "scheme": c1sch,
                        "c3_idx_map": c3idx, "c3_scheme": c3sch}, args.save)
            print(f"  ★ best mAP50={best:.4f} → {args.save}")
        if (ep + 1) % 10 == 0:
            torch.save({"model": model, "epoch": ep, "mAP50": mE,
                        "idx_map": c1idx, "scheme": c1sch,
                        "c3_idx_map": c3idx, "c3_scheme": c3sch},
                       save_dir / f"sparse_c3_ep{ep+1}.pt")

    print(f"\n=== FINAL: one-shot={m0:.4f} → best={best:.4f} "
          f"(target ≈0.8383 = sparse_lw_best; c3 sens dự báo -0..-1.4pp one-shot) ===")


if __name__ == "__main__":
    main()
