"""
create_sparse_extracted.py -- Tạo extracted/ directory cho sparse model.

Giữ nguyên: metadata.json, requant_params.json, bias_bin (từ QAT dense model)
Thay: weights_bin/layer_XX.bin → sparse INT8 (quantize FP32 sparse bằng same per-channel scale)

Usage:
  python create_sparse_extracted.py
  python create_sparse_extracted.py --sparse runs/sparse/sparse_best.pt --out runs/quant/extracted_sparse_ep41
"""
import torch, sys, json, numpy as np, shutil, argparse, warnings
from pathlib import Path
warnings.filterwarnings("ignore"); sys.path.insert(0, str(Path(__file__).parent))
import torch.nn as nn

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sparse", default="runs/sparse/sparse_best.pt")
    ap.add_argument("--src",    default="runs/quant/extracted")
    ap.add_argument("--out",    default="runs/quant/extracted_sparse")
    args = ap.parse_args()

    print(f"=== Create sparse extracted ===")
    print(f"  Sparse model : {args.sparse}")
    print(f"  Dense src    : {args.src}")
    print(f"  Output       : {args.out}")

    # Load sparse model
    ckpt = torch.load(args.sparse, map_location="cpu", weights_only=False)
    sp_model = ckpt["model"].float().eval()
    ep = ckpt.get("epoch","?"); mAP = ckpt.get("mAP50", 0)
    print(f"  Sparse ep={ep} mAP50={mAP:.4f}")

    # Build name→module map for sparse model
    sp_modules = {name: mod for name, mod in sp_model.named_modules()
                  if isinstance(mod, nn.Conv2d)}

    # Load metadata
    src = Path(args.src)
    meta = json.load(open(src / "metadata.json"))

    # Setup output dir: copy everything from src first
    out = Path(args.out)
    if out.exists(): shutil.rmtree(out)
    shutil.copytree(src, out)
    print(f"  Copied base extracted → {out}/")

    # Replace weights_bin for conv1x1 layers with sparse weights
    replaced = 0; kept = 0
    for lid in range(63):
        k = f"layer_{lid:02d}"
        if k not in meta: continue
        m = meta[k]
        ws = m["weight_shape"]
        layer_name = m.get("name", "")

        # Only replace conv1x1 with Cin%16==0
        if (m.get("kernel_size") != [1,1] or
            len(ws) < 2 or ws[1] % 16 != 0 or
            m.get("is_depthwise", False)):
            kept += 1
            continue

        # Find matching sparse module
        sp_mod = sp_modules.get(layer_name)
        if sp_mod is None:
            print(f"  WARNING: {k} name={layer_name} not in sparse model, keeping dense")
            kept += 1
            continue

        # Quantize sparse FP32 weights using same per-channel scale
        wsc = np.array(m["weight_scale_per_channel"], dtype=np.float32)  # [Cout]
        w_fp32 = sp_mod.weight.data.numpy().reshape(ws[0], -1)            # [Cout, Cin*kH*kW]

        # Per output channel: w_int8[oc,:] = clip(round(w_fp32[oc,:] / wsc[oc]), -128, 127)
        w_int8 = np.zeros_like(w_fp32, dtype=np.float32)
        for oc in range(ws[0]):
            if wsc[oc] > 0:
                w_int8[oc] = np.round(w_fp32[oc] / wsc[oc])
        w_int8 = np.clip(w_int8, -128, 127).astype(np.int8).reshape(ws)

        # Write
        bin_path = out / "weights_bin" / f"{k}.bin"
        w_int8.tofile(bin_path)

        nz = int((w_int8 != 0).sum()); total = w_int8.size
        print(f"  {k} [{ws[0]},{ws[1]}] {layer_name}: NZ={nz}/{total}={100*nz/total:.1f}%")
        replaced += 1

    print(f"\nDone: replaced={replaced} conv1x1 sparse, kept={kept} other layers")
    print(f"Output → {out}/")
    print(f"  Use: set E='{out}' in inference_cpu.py, or --extracted {out}")

if __name__ == "__main__":
    main()
