"""
sens_scan.py -- SENSITIVITY SCAN cho structured N:16 sparsity KHÔNG ĐỒNG ĐỀU (per-layer N).
Đo trên model QAT dense (qat_3fixes13, KHÔNG train lại): với TỪNG conv1x1, prune riêng lớp đó
ở N ∈ {1,2,3,4,6} (structured: idx top-N per M-block CHUNG cho MỌI oc — y hệt
gpu_sparse_train.make_nm_masks_structured / FPGA wt_rd_idx), đo mAP50 trên SUBSET val cố định.
Kết quả → bảng sensitivity per-layer → chọn N per-layer cho lần retrain non-uniform.

Baselines: dense (subset) + uniform-N3-all (one-shot, pre-finetune — số này SẼ thấp, retrain kéo lại).
Ghi incremental vào runs/sens/sens_scan.json (resume được).
Usage: venv python sens_scan.py [--subset 250] [--ns 1,2,3,4,6]
"""
import torch, sys, warnings, argparse, json, random
import numpy as np
import torch.nn as nn
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
from convert_to_quant import convert_model
import val as yolo_val
from utils.dataloaders import create_dataloader
from utils.general import check_dataset, colorstr
import brevitas.nn as qnn

# pruned_fused.pt đã mất (dọn ổ 2026-07-03) — RECOVERY: skeleton = sparse_best.pt (đúng kiến
# trúc Lite deployed) + convert_model (brevitas wrap, đúng recipe eval_structured_prune) + nạp
# ĐẦY ĐỦ 190 tensor qat_3fixes13 (kể cả scale brevitas). LƯU Ý: nạp weights FP32 KHÔNG kèm
# scale brevitas cho dense=0.19 (weights đồng-thích-nghi fake-quant) — bắt buộc QAT domain.
SKEL  = "runs/sparse/sparse_best.pt"
QAT   = "runs/train/qat_3fixes13/weights/best.pt"
DATA  = "data/pklot.yaml"; IMGSZ = 416; BATCH = 16; M = 16
DEV   = "cuda" if torch.cuda.is_available() else "cpu"
OUTD  = Path("runs/sens"); OUTD.mkdir(parents=True, exist_ok=True)
OUTJ  = OUTD / "sens_scan.json"


def structured_mask(w, N):
    """w [Cout,Cin,1,1] -> pruned copy: per M-block, idx top-N theo Σ|w| trên TOÀN BỘ oc."""
    Cout, Cin = w.shape[0], w.shape[1]
    a = w.detach().cpu().numpy().reshape(Cout, Cin).astype(np.float32)
    out = np.zeros_like(a)
    for mb in range(Cin // M):
        b = a[:, mb*M:(mb+1)*M]
        k = np.argsort(np.abs(b).sum(0))[-N:]          # idx chung mọi oc (FPGA idxmem)
        out[:, mb*M + k] = b[:, k]
    return torch.from_numpy(out.reshape(w.shape)).to(w.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", type=int, default=250)
    ap.add_argument("--ns", type=str, default="1,2,3,4,6")
    args = ap.parse_args()
    NS = [int(x) for x in args.ns.split(",")]

    # ── model: skeleton sparse_best → convert_model (brevitas) → FULL qat_3fixes13 sd ──
    base = torch.load(SKEL, map_location="cpu", weights_only=False)
    model = base["model"].float(); convert_model(model)
    ck = torch.load(QAT, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"load: miss={len(miss)} unexp={len(unexp)} (kỳ vọng ~0/~0 sau convert_model)")
    model = model.to(DEV).eval()
    dd = check_dataset(DATA)
    if not hasattr(model, "names"): model.names = dd["names"]

    # ── subset val cố định (seed 0) ──
    vdir = Path(dd["val"]);  # .../valid/images
    imgs = sorted([str(p) for p in vdir.glob("*.jpg")] + [str(p) for p in vdir.glob("*.png")])
    random.Random(0).shuffle(imgs)
    sub = imgs[:args.subset]
    subtxt = OUTD / "val_subset.txt"
    subtxt.write_text("\n".join(sub))
    print(f"subset: {len(sub)}/{len(imgs)} ảnh")
    gs = max(int(model.stride.max()) if hasattr(model, "stride") else 32, 32)
    vl, _ = create_dataloader(str(subtxt), imgsz=IMGSZ, batch_size=BATCH, stride=gs, hyp=None,
                              augment=False, cache=False, pad=0.5, rect=True, workers=2,
                              prefix=colorstr("val: "), shuffle=False)
    svdir = OUTD / "val_tmp"; svdir.mkdir(exist_ok=True)

    def ev():
        res, _, _ = yolo_val.run(data=dd, model=model, dataloader=vl, save_dir=svdir,
                                 imgsz=IMGSZ, batch_size=BATCH, device=DEV, workers=0,
                                 half=False, plots=False, save_json=False, verbose=False)
        return float(res[2]), float(res[3])   # mAP50, mAP50-95

    # ── danh sách conv1x1 prunable + weight gốc (brevitas QuantConv2d) ──
    convs = []
    for name, mod in model.named_modules():
        if isinstance(mod, qnn.QuantConv2d) and mod.kernel_size == (1, 1) \
           and mod.weight.shape[1] % M == 0:
            convs.append((name, mod, mod.weight.data.clone()))
    print(f"conv1x1 prunable: {len(convs)}")

    results = json.loads(OUTJ.read_text()) if OUTJ.exists() else {}

    def save():
        OUTJ.write_text(json.dumps(results, indent=1))

    # ── baselines ──
    if "dense" not in results:
        m50, m95 = ev(); results["dense"] = {"mAP50": m50, "mAP5095": m95}; save()
        print(f"BASE dense: mAP50={m50:.4f}")
    if "uniformN3" not in results:
        for name, mod, w0 in convs:
            mod.weight.data = structured_mask(w0, 3).to(DEV)
        m50, m95 = ev(); results["uniformN3"] = {"mAP50": m50, "mAP5095": m95}; save()
        for name, mod, w0 in convs: mod.weight.data = w0.clone().to(DEV)
        print(f"BASE uniformN3 (one-shot, chưa finetune): mAP50={m50:.4f}")

    # ── per-layer scan ──
    d50 = results["dense"]["mAP50"]
    for li, (name, mod, w0) in enumerate(convs):
        shape = list(w0.shape)
        for N in NS:
            key = f"{name}|N{N}"
            if key in results: continue
            mod.weight.data = structured_mask(w0, N).to(DEV)
            m50, m95 = ev()
            mod.weight.data = w0.clone().to(DEV)
            results[key] = {"mAP50": m50, "mAP5095": m95, "shape": shape}
            save()
            print(f"[{li+1}/{len(convs)}] {name} {shape[1]}->{shape[0]} N={N}: "
                  f"mAP50={m50:.4f} (Δ={m50-d50:+.4f})", flush=True)

    # ── bảng tóm tắt ──
    print("\n===== SENSITIVITY (ΔmAP50 vs dense, one-shot) =====")
    print(f"{'layer':38s} {'Cin':>5s} " + " ".join(f"N={n:<2d}" for n in NS))
    for name, mod, w0 in convs:
        row = []
        for N in NS:
            r = results.get(f"{name}|N{N}")
            row.append(f"{(r['mAP50']-d50)*100:+5.1f}" if r else "  -  ")
        print(f"{name:38s} {w0.shape[1]:5d} " + " ".join(row))
    print(f"\ndense={d50:.4f}  uniformN3(one-shot)={results['uniformN3']['mAP50']:.4f}")


if __name__ == "__main__":
    main()
