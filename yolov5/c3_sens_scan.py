"""
c3_sens_scan.py -- Sensitivity scan cho CONV3x3 (mở rộng DominoSearch sang c3).
Câu hỏi: mỗi lớp conv3x3 chịu được N tối thiểu bao nhiêu (N:16 trên Cin) trước khi mAP tụt?
conv3x3 = 45% MAC, hiện DENSE 9-pass (kick 0.41s lớn nhất) — chưa hưởng sparsity.
Prune c3 UNIFORM aggressive từng NO-GO (-39pp); scan này đo per-layer headroom để biết
DominoSearch-per-layer có cứu được không TRƯỚC khi đầu tư engine sparse-c3.

Cấu trúc mask (khớp HW i_spec drain-N + 9-pass): mỗi M-block của Cin, chọn top-N channel
theo Σ|w| trên TOÀN BỘ oc VÀ 9 tap (idx CHIA SẺ qua tap — HW-simple). Giữ N channel đó ở
cả 9 tap. Bit-width KHÔNG liên quan (INT8, đo trên model FP32 dense_parent).

Usage: SPARSE... venv python c3_sens_scan.py [--subset 250] [--ns 4,6,8,10,12]
"""
import os, sys, warnings, argparse, json, random
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr

PARENT = "runs/sparse/dense_parent.pt"   # FP32 dense sạch (0.8167) — tránh wrong-domain
DATA = "data/pklot.yaml"; IMGSZ = 416; BATCH = 16; M = 16
DEV  = "cuda" if torch.cuda.is_available() else "cpu"
OUTD = Path("runs/sens"); OUTD.mkdir(parents=True, exist_ok=True)
OUTJ = OUTD / "c3_sens.json"


def c3_mask(w, N):
    """w [Cout,Cin,3,3] -> pruned: per M-block Cin, top-N theo Σ|w| (oc × 9 tap), chia sẻ tap."""
    Cout, Cin, kh, kw = w.shape
    a = w.detach().cpu().numpy()
    out = np.zeros_like(a)
    score = np.abs(a).sum(axis=(0, 2, 3))          # [Cin] — tổng qua oc + 9 tap
    for mb in range(Cin // M):
        seg = score[mb*M:(mb+1)*M]
        k = np.argsort(seg)[-N:]
        for i in k:
            out[:, mb*M + i] = a[:, mb*M + i]      # giữ channel i ở CẢ 9 tap
    return torch.from_numpy(out).to(w.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=250)
    ap.add_argument("--ns", type=str, default="4,6,8,10,12")
    args = ap.parse_args()
    NS = [int(x) for x in args.ns.split(",")]

    ck = torch.load(PARENT, map_location="cpu", weights_only=False)
    model = ck["model"].float().to(DEV).eval()
    print(f"parent {PARENT}: FP32 mAP50={ck.get('mAP50')}")
    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]

    vdir = Path(dd["val"])
    imgs = sorted([str(p) for p in vdir.glob("*.jpg")] + [str(p) for p in vdir.glob("*.png")])
    random.Random(0).shuffle(imgs); sub = imgs[:args.subset]
    subtxt = OUTD / "c3_val_subset.txt"; subtxt.write_text("\n".join(sub))
    gs = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)
    vl, _ = create_dataloader(str(subtxt), imgsz=IMGSZ, batch_size=BATCH, stride=gs, hyp=None,
                              augment=False, cache=False, pad=0.5, rect=True, workers=2,
                              prefix=colorstr("val: "), shuffle=False)
    svdir = OUTD / "c3_val_tmp"; svdir.mkdir(exist_ok=True)

    def ev():
        res, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, save_dir=svdir,
                                 imgsz=IMGSZ, batch_size=BATCH, device=DEV, workers=0,
                                 half=False, plots=False, save_json=False, verbose=False)
        return float(res[2])

    # conv3x3 prunable (kernel 3x3, không depthwise, Cin%16==0)
    convs = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d) and mod.kernel_size == (3, 3) \
           and mod.groups == 1 and mod.weight.shape[1] % M == 0:
            convs.append((name, mod, mod.weight.data.clone()))
    print(f"conv3x3 prunable: {len(convs)}")

    results = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}
    def save(): OUTJ.write_text(json.dumps(results, indent=1))
    if "dense" not in results:
        results["dense"] = ev(); save()
        print(f"BASE dense mAP50={results['dense']:.4f}")
    d0 = results["dense"]

    for li, (name, mod, w0) in enumerate(convs):
        for N in NS:
            key = f"{name}|N{N}"
            if key in results: continue
            mod.weight.data = c3_mask(w0, N).to(DEV)
            m = ev()
            mod.weight.data = w0.clone().to(DEV)
            results[key] = {"mAP50": m, "shape": list(w0.shape)}
            save()
            print(f"[{li+1}/{len(convs)}] {name} Cin={w0.shape[1]} N={N}: "
                  f"mAP50={m:.4f} (Δ={m-d0:+.4f})", flush=True)

    print(f"\n===== C3 SENSITIVITY (ΔmAP50 vs dense={d0:.4f}) =====")
    print(f"{'layer':30s} {'Cin':>5s} " + " ".join(f"N={n:<3d}" for n in NS))
    for name, mod, w0 in convs:
        row = [f"{(results[f'{name}|N{n}']['mAP50']-d0)*100:+5.1f}" if f'{name}|N{n}' in results
               else "  -  " for n in NS]
        print(f"{name:30s} {w0.shape[1]:5d} " + " ".join(row))


if __name__ == "__main__":
    main()
