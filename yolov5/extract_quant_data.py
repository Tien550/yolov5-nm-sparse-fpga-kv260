"""
extract_quant_data.py - Sub-step 2.1 + 2.2: Extract INT8 weights + requant params
==================================================================================
Output structure:
  runs/quant/extracted/
    weights_bin/
      layer_00.bin              # INT8 [Cout, Cin, Kh, Kw] raw bytes
      layer_01.bin
      ...
    bias_bin/
      layer_00.bin              # INT32 little-endian, len = Cout * 4 bytes
      ...
    metadata.json               # Per-layer config + scales
    requant_params.json         # M0 (UINT16) + shift (UINT5) per-channel
    extraction_log.txt          # Human-readable summary

Cach chay:
    cd D:\\yolov5-fpga\\yolov5
    python extract_quant_data.py
"""
import torch
import numpy as np
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from convert_to_quant import convert_model
import brevitas.nn as qnn

# ========================== CONFIG ==========================
FUSED_BASE = "runs/export/pruned_fused.pt"
QAT_CKPT   = "runs/train/qat_3fixes13/weights/best.pt"
OUT_DIR    = Path("runs/quant/extracted")
IMGSZ      = 416
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ============================================================


def find_best_shift(M_float):
    """
    Find shift such that M_float * 2^shift in [2^15, 2^16).
    Returns (M0_uint16, shift_uint5).
    Edge case: M_float == 0 -> return (0, 0).
    """
    if M_float <= 0:
        return 0, 0
    shift = 0
    val = M_float
    # Scale up if too small
    while val < (1 << 15) and shift < 31:
        val *= 2.0
        shift += 1
    # Scale down if too large (rare)
    while val >= (1 << 16) and shift > 0:
        val /= 2.0
        shift -= 1
    M0 = int(round(val))
    if M0 >= (1 << 16):
        M0 = (1 << 16) - 1
    return M0, shift


def main():
    print("=" * 60)
    print("  EXTRACT QUANT DATA (Sub-step 2.1 + 2.2)")
    print("=" * 60)

    device = torch.device(DEVICE)

    # ----- 1. Load model -----
    print(f"\n[1] Load FP32 base + rebuild quant + load QAT weights")
    base = torch.load(FUSED_BASE, map_location="cpu", weights_only=False)
    model = base["model"].float()
    convert_model(model)
    qat = torch.load(QAT_CKPT, map_location="cpu", weights_only=False)
    sd = qat["state_dict"] if "state_dict" in qat else qat
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"  load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model = model.to(device).eval()

    # ----- 2. Collect QuantConv2d list (in declaration order) -----
    quant_layers = []  # list of (name, module)
    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantConv2d):
            quant_layers.append((name, m))
    print(f"  Found {len(quant_layers)} QuantConv2d layers")

    # ----- 3. Hook input_quant of QuantConv2d + Detect.m to build unified timeline -----
    print(f"\n[2] Forward hooks: capture S_in + unified execution timeline")
    captured = {}      # quant_layer_name -> S_in
    timeline = []      # list of ('quant', name) or ('detect', i) in execution order

    def make_quant_hook(layer_name):
        def hook(module, inp, out):
            try:
                s = float(out.scale.detach().cpu().flatten()[0].item())
            except Exception:
                s = None
            if s is not None and layer_name not in captured:
                captured[layer_name] = s
                timeline.append(("quant", layer_name))
        return hook

    detect_hits = []  # list of (timeline_index, detect_idx)
    def make_detect_hook(detect_idx):
        def hook(module, inp, out):
            timeline.append(("detect", detect_idx))
        return hook

    handles = []
    for name, m in quant_layers:
        handles.append(m.input_quant.register_forward_hook(make_quant_hook(name)))

    # FIX 2: Detect head is now QuantConv2d, so hooks are already registered above
    # (detect layers are included in quant_layers)
    # Keep this code for timeline tracking if needed
    from models.yolo import Detect
    for mod in model.modules():
        if isinstance(mod, Detect):
            for i, conv in enumerate(mod.m):
                # Check if detect layers are QuantConv2d (FIX 2)
                if not isinstance(conv, qnn.QuantConv2d):
                    handles.append(conv.register_forward_hook(make_detect_hook(i)))
            break

    print(f"  Forward 1 dummy image to trigger hooks")
    with torch.no_grad():
        dummy = torch.rand(1, 3, IMGSZ, IMGSZ, device=device)
        _ = model(dummy)

    for h in handles:
        h.remove()

    print(f"  Captured S_in for {len(captured)}/{len(quant_layers)} layers")
    print(f"  Timeline length: {len(timeline)} events")

    # Build exec_order (only quant layers) for backwards compat
    exec_order = [name for kind, name in timeline if kind == "quant"]

    # Identify quant layers feeding Detect via model topology (Detect.f indices)
    from models.yolo import Detect
    fp32_output_set = set()
    id_to_name = {id(m): name for name, m in quant_layers}

    detect_mod = None
    for mod in model.modules():
        if isinstance(mod, Detect):
            detect_mod = mod
            break

    if detect_mod is not None and hasattr(detect_mod, "f"):
        feeding_indices = detect_mod.f  # e.g. [17, 20, 23]
        print(f"  Detect.f (feeding layer indices): {feeding_indices}")
        for idx in feeding_indices:
            feeder_module = model.model[idx]
            # YOLOv5 convention: C3 module's output is cv3.conv,
            # plain Conv module's output is .conv
            output_conv = None
            if hasattr(feeder_module, "cv3") and hasattr(feeder_module.cv3, "conv"):
                output_conv = feeder_module.cv3.conv          # C3, C3_DWS
            elif hasattr(feeder_module, "conv"):
                output_conv = feeder_module.conv               # plain Conv
            elif hasattr(feeder_module, "cv2") and hasattr(feeder_module.cv2, "conv"):
                output_conv = feeder_module.cv2.conv           # SPPF fallback
            
            if output_conv is not None and isinstance(output_conv, qnn.QuantConv2d):
                qname = id_to_name.get(id(output_conv))
                if qname:
                    fp32_output_set.add(qname)
                    print(f"    model.model[{idx}] ({type(feeder_module).__name__}) -> {qname}")
            else:
                print(f"    model.model[{idx}] ({type(feeder_module).__name__}) -> NOT FOUND, type={type(output_conv).__name__}")

    print(f"  Layers feeding Detect head: {len(fp32_output_set)} (expect 3)")

    print(f"  Captured S_in for {len(captured)}/{len(quant_layers)} layers")
    if len(captured) != len(quant_layers):
        missing = [n for n, _ in quant_layers if n not in captured]
        print(f"  WARN: missing scales for: {missing[:5]}")

    # ----- 5. Build S_out map: S_out[i] = S_in[i+1] in exec order, EXCEPT fp32_output ones -----
    print(f"\n[3] Compute S_out per layer (skip fp32_output layers)")
    s_out_map = {}
    for i, name in enumerate(exec_order):
        if name in fp32_output_set:
            s_out_map[name] = None
        elif i + 1 < len(exec_order):
            s_out_map[name] = captured[exec_order[i + 1]]
        else:
            s_out_map[name] = None

    # ----- 6. Output dirs -----
    weights_dir = OUT_DIR / "weights_bin"
    bias_dir    = OUT_DIR / "bias_bin"
    weights_dir.mkdir(parents=True, exist_ok=True)
    bias_dir.mkdir(parents=True, exist_ok=True)

    metadata = {}
    requant = {}
    log_lines = []

    print(f"\n[4] Extract weights, biases, requant params")
    for idx, (name, m) in enumerate(quant_layers):
        layer_id = f"layer_{idx:02d}"

        # ---- INT8 weights ----
        qw = m.quant_weight()
        w_int = qw.int(float_datatype=False).detach().cpu().numpy().astype(np.int8)
        s_w = qw.scale.detach().cpu().numpy().astype(np.float32).flatten()  # per-channel
        # Validate range
        if w_int.min() < -128 or w_int.max() > 127:
            print(f"  WARN {name}: INT8 out of range [{w_int.min()},{w_int.max()}]")

        # Save raw bytes [Cout, Cin, Kh, Kw]
        (weights_dir / f"{layer_id}.bin").write_bytes(w_int.tobytes())

        # ---- Scales ----
        s_in = captured.get(name, None)
        s_out = s_out_map.get(name, None)
        is_fp32_output = s_out is None

        # ---- Bias folding (FP32 -> INT32) ----
        bias_int32 = None
        if m.bias is not None and s_in is not None:
            bias_fp32 = m.bias.detach().cpu().numpy().astype(np.float32)
            # bias_int32[c] = round(bias_fp32[c] / (S_in * S_w[c]))
            denom = s_in * s_w  # per-channel
            denom = np.where(denom == 0, 1e-12, denom)
            bias_int32 = np.round(bias_fp32 / denom).astype(np.int32)
            (bias_dir / f"{layer_id}.bin").write_bytes(bias_int32.tobytes())

        # ---- M0 + shift (per-channel) ----
        M0_list = []
        shift_list = []
        if not is_fp32_output and s_in is not None and s_out is not None:
            for c in range(len(s_w)):
                M = (s_in * float(s_w[c])) / s_out
                M0_c, shift_c = find_best_shift(M)
                M0_list.append(M0_c)
                shift_list.append(shift_c)
        # else: leave empty -> HLS will dequantize to FP32 for Detect head

        # ---- Metadata ----
        is_dw = (m.groups > 1 and m.groups == m.in_channels)
        metadata[layer_id] = {
            "name": name,
            "in_channels": m.in_channels,
            "out_channels": m.out_channels,
            "kernel_size": list(m.kernel_size),
            "stride": list(m.stride),
            "padding": list(m.padding),
            "groups": m.groups,
            "is_depthwise": is_dw,
            "has_bias": m.bias is not None,
            "weight_shape": list(w_int.shape),
            "weight_scale_per_channel": s_w.tolist(),
            "input_scale": s_in,
            "output_scale": s_out,
            "fp32_output": is_fp32_output,
        }

        if not is_fp32_output:
            requant[layer_id] = {
                "M0": M0_list,
                "shift": shift_list,
            }

        s_in_str = f"{s_in:.6e}" if s_in is not None else "None"
        s_out_str = f"{s_out:.6e}" if s_out is not None else "None"
        log_lines.append(
            f"{layer_id} {name:50s} shape={list(w_int.shape)} "
            f"dw={is_dw} S_in={s_in_str} S_out={s_out_str}"
        )

    # ----- 7. Save JSON files -----
    print(f"\n[5] Save metadata + requant params")
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    with open(OUT_DIR / "requant_params.json", "w") as f:
        json.dump(requant, f, indent=2)

    # ----- 8. Summary log -----
    n_w_files = len(list(weights_dir.glob("*.bin")))
    n_b_files = len(list(bias_dir.glob("*.bin")))
    n_fp32_out = sum(1 for k in metadata if metadata[k]["fp32_output"])

    summary = [
        "=" * 60,
        "EXTRACTION SUMMARY (FIX 2: Detect head now quantized)",
        "=" * 60,
        f"Total QuantConv2d layers : {len(quant_layers)}  (includes detect head 59-61)",
        f"Weight .bin files written: {n_w_files}",
        f"Bias   .bin files written: {n_b_files}",
        f"Layers with fp32 output  : {n_fp32_out}  (backbone layers feeding FPN)",
        f"Layers with M0+shift     : {len(requant)}  (all quantized layers)",
        "",
        "First 5 layers:",
    ]
    for k in list(metadata.keys())[:5]:
        d = metadata[k]
        s_in_v = d['input_scale']
        s_out_v = d['output_scale']
        s_in_p = f"{s_in_v:.4e}" if s_in_v is not None else "None"
        s_out_p = f"{s_out_v:.4e}" if s_out_v is not None else "None"
        summary.append(
            f"  {k}: {d['name']} shape={d['weight_shape']} "
            f"S_in={s_in_p} S_out={s_out_p}"
        )

    summary_text = "\n".join(summary)
    print("\n" + summary_text)

    with open(OUT_DIR / "extraction_log.txt", "w") as f:
        f.write(summary_text + "\n\n")
        f.write("=" * 60 + "\n")
        f.write("PER-LAYER LOG\n")
        f.write("=" * 60 + "\n")
        for line in log_lines:
            f.write(line + "\n")

    print(f"\nAll outputs in: {OUT_DIR}")
    print("\nNext: Sub-step 2.3 - Bit-accuracy verification (verify_extracted.py)")


if __name__ == "__main__":
    main()