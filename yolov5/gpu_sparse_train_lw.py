"""
gpu_sparse_train_lw.py -- Finetune structured sparse với N PER-LAYER (DominoSearch bước 2).
Y hệt gpu_sparse_train.py (mask cột-chia-sẻ, apply sau mỗi step, cosine LR, 50ep) nhưng:
  - N đọc từ runs/sens/layerwise_nm.json (output domino_search.py) thay vì uniform.
  - Khởi tạo từ dense_parent.pt (paper Table 7 E5: pre-trained dense init tốt nhất).
Output: runs/sparse_lw/sparse_lw_best.pt + idx_per_mblk.json (cho packer FPGA).
Usage: venv python gpu_sparse_train_lw.py [--epochs 50] [--bs 16]
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


def make_masks_lw(model, scheme):
    """Mask cột-chia-sẻ theo N per-layer. Trả (masks, idx_map) như bản gốc."""
    masks = {}; idx_map = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1) and mod.weight.shape[1] % M == 0:
            if name not in scheme: continue
            N = int(scheme[name])
            w = mod.weight.data
            Cout, Cin = w.shape[0], w.shape[1]
            a = w.reshape(Cout, Cin).abs()
            mask = torch.zeros_like(a); layer_idx = []
            for mblk in range(Cin // M):
                blk = a[:, mblk*M:(mblk+1)*M]
                top = blk.mean(dim=0).topk(N).indices.sort().values
                for i in top: mask[:, mblk*M + i] = 1.0
                layer_idx.append(top.tolist())
            masks[name] = mask.reshape(w.shape).to(w.device)
            idx_map[name] = layer_idx
    return masks, idx_map


def apply_masks(model, masks):
    for name, mod in model.named_modules():
        if name in masks: mod.weight.data.mul_(masks[name])


def cosine_lr(opt, ep, total, lr_max=0.01, lr_min=1e-4):
    lr = lr_min + (lr_max - lr_min) * 0.5 * (1 + math.cos(math.pi * ep / total))
    for g in opt.param_groups: g['lr'] = lr
    return lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--scheme", type=str, default="runs/sens/layerwise_nm.json")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--init", type=str, default="runs/sparse/dense_parent.pt")
    ap.add_argument("--save", type=str, default="runs/sparse_lw/sparse_lw_best.pt")
    args = ap.parse_args(); init_seeds(0)

    sch = json.loads(Path(args.scheme).read_text())["scheme"]
    hist = {}
    for v in sch.values(): hist[v] = hist.get(v, 0) + 1
    print(f"=== LW-SPARSE FINETUNE: {len(sch)} layers, N-hist={dict(sorted(hist.items()))}, "
          f"{args.epochs} epochs ===")

    IMGSZ = args.imgsz
    ck = torch.load(args.init, map_location="cpu", weights_only=False)
    model = ck["model"].float().to(DEV)
    print(f"init: dense_parent mAP50={ck.get('mAP50'):.4f}")
    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]
    gs = max(int(model.stride.max()), 32)

    vl, _ = create_dataloader(dd["val"], imgsz=IMGSZ, batch_size=args.bs, stride=gs,
                              hyp=None, augment=False, cache="disk", pad=0.5, rect=True,
                              workers=args.workers, prefix=colorstr("val: "), shuffle=False)

    def evalmap(tag):
        model.eval()
        sd_ = Path("runs/val/sparse_lw"); sd_.mkdir(parents=True, exist_ok=True)
        res, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, save_dir=sd_,
                                 imgsz=IMGSZ, batch_size=args.bs, device=DEV, workers=0,
                                 half=False, plots=False, save_json=False, verbose=False)
        print(f"  [{tag}] mAP50={res[2]:.4f} mAP50-95={res[3]:.4f}", flush=True)
        return float(res[2])

    masks, idx_map = make_masks_lw(model, sch)
    apply_masks(model, masks)
    m0 = evalmap("one-shot lw-pruned (chưa finetune)")

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

    save_dir = Path("runs/sparse_lw"); save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "idx_per_mblk.json").write_text(json.dumps(idx_map))
    best = 0.0
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
            apply_masks(model, masks)          # giữ đúng sparse pattern (static mask)
            tot += loss.item(); nb += 1
            if i % 200 == 0:
                print(f"    ep{ep} it{i}/{len(tl)} loss={loss.item():.3f} lr={lr:.5f}", flush=True)
        print(f"  ep{ep} avg_loss={tot/nb:.3f}")
        mE = evalmap(f"lw ep{ep+1}")
        if mE > best:
            best = mE
            torch.save({"model": model, "epoch": ep, "mAP50": mE, "idx_map": idx_map,
                        "scheme": sch}, args.save)
            print(f"  ★ best mAP50={best:.4f} → sparse_lw_best.pt")
        if (ep + 1) % 10 == 0:
            torch.save({"model": model, "epoch": ep, "mAP50": mE, "idx_map": idx_map,
                        "scheme": sch}, save_dir / f"sparse_lw_ep{ep+1}.pt")

    print(f"\n=== FINAL: one-shot={m0:.4f} → best={best:.4f} "
          f"(uniform-N3 deployed = 0.7976 để so) ===")


if __name__ == "__main__":
    main()
