"""
gen_golden.py  --  Export INT8 per-layer golden activation tensors.

Phương pháp đúng:
  - INPUT  của layer i: hook input_quant → S_in từ metadata
  - OUTPUT của layer i: hook Conv WRAPPER (conv+BN+act) output → S_out từ metadata

Vấn đề trước: hook QuantConv2d.output là raw conv (trước BN+act),
còn S_out được tính từ post-BN+act scale → mismatch.
Giải pháp: find Conv wrapper cho mỗi QuantConv2d, hook output của nó.
"""
import argparse, json, os, sys, warnings
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn

warnings.filterwarnings("ignore")
sys.path.insert(0, "yolov5")
from convert_to_quant import convert_model
import brevitas.nn as qnn

FP32_OUTPUT_LAYERS = {42, 49, 56, 62}
FUSED_BASE = "yolov5/runs/export/pruned_fused.pt"


def letterbox(img, sz=416):
    h, w = img.shape[:2]
    r = sz / max(h, w)
    img = cv2.resize(img, (int(round(w*r)), int(round(h*r))))
    dh, dw = sz - img.shape[0], sz - img.shape[1]
    img = cv2.copyMakeBorder(img, dh//2, dh-dh//2, dw//2, dw-dw//2,
                              cv2.BORDER_CONSTANT, value=(114,114,114))
    return img


def find_conv_wrappers(model, quant_layers):
    """
    Với mỗi QuantConv2d, tìm module cha trực tiếp (Conv wrapper) trong model.
    Conv wrapper = module có .conv = QuantConv2d và .bn, .act.
    Nếu không có wrapper (Detect head), dùng chính QuantConv2d.
    """
    # Build parent map
    child_to_parent = {}
    for parent_name, parent_mod in model.named_modules():
        for child_name, child_mod in parent_mod.named_children():
            child_to_parent[id(child_mod)] = (parent_name, parent_mod)

    wrappers = {}   # layer_idx -> (wrapper_module, has_act)
    for idx, (name, m) in enumerate(quant_layers):
        parent_info = child_to_parent.get(id(m))
        if parent_info is not None:
            pname, pmod = parent_info
            # Conv wrapper có bn attribute
            if hasattr(pmod, 'bn') and hasattr(pmod, 'act'):
                wrappers[idx] = (pmod, True, pname)
            elif hasattr(pmod, 'bn'):
                wrappers[idx] = (pmod, False, pname)
            else:
                wrappers[idx] = (m, False, name)   # fallback: dùng QuantConv2d
        else:
            wrappers[idx] = (m, False, name)
    return wrappers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fused_base',  default='yolov5/runs/export/pruned_fused.pt')
    parser.add_argument('--qat_ckpt',    default='yolov5/runs/train/qat_3fixes13/weights/best.pt')
    parser.add_argument('--image',       default='test_images/pklot_sample.jpg')
    parser.add_argument('--output',      default='golden')
    parser.add_argument('--extracted',   default='yolov5/runs/quant/extracted')
    parser.add_argument('--imgsz',       type=int, default=416)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    with open(os.path.join(args.extracted, 'metadata.json')) as f:
        metadata = json.load(f)

    # Load model
    base = torch.load(args.fused_base, map_location='cpu', weights_only=False)
    model = base['model'].float()
    convert_model(model)
    qat = torch.load(args.qat_ckpt, map_location='cpu', weights_only=False)
    sd  = qat['state_dict'] if 'state_dict' in qat else qat
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[INFO] load_state_dict: missing={len(miss)} unexpected={len(unexp)}")
    model = model.to(device).eval()

    quant_layers = [(n, m) for n, m in model.named_modules() if isinstance(m, qnn.QuantConv2d)]
    total = len(quant_layers)
    print(f"[INFO] Found {total} QuantConv2d layers")

    # Find Conv wrappers
    wrappers = find_conv_wrappers(model, quant_layers)

    # ── Hook 1: input_quant để lấy S_in (execution order) ────────────────────
    s_in_map   = {}   # layer_name -> float S_in
    exec_order = []

    def make_sin_hook(name):
        def hook(m, inp, out):
            if name not in s_in_map:
                try:
                    s_in_map[name] = float(out.scale.detach().cpu().flatten()[0])
                    exec_order.append(name)
                except Exception: pass
        return hook

    # ── Hook 2: QuantConv2d để capture float input ────────────────────────────
    float_in_data = {}   # idx -> float numpy [C,H,W]

    def make_in_hook(idx):
        def hook(m, inp, out):
            if idx not in float_in_data:
                arr = inp[0].detach().cpu().numpy()
                float_in_data[idx] = arr[0] if arr.ndim == 4 else arr
        return hook

    # ── Hook 3: Conv wrapper để capture float output (post-BN+act) ───────────
    float_out_data = {}  # idx -> float numpy [C,H,W]
    wrapper_out_seen = set()

    def make_out_hook(idx):
        def hook(m, inp, out):
            if idx not in wrapper_out_seen:
                wrapper_out_seen.add(idx)
                arr = out.detach().cpu().numpy() if not hasattr(out, 'value') \
                      else out.value.detach().cpu().numpy()
                float_out_data[idx] = arr[0] if arr.ndim == 4 else arr
        return hook

    handles = []
    for i, (name, m) in enumerate(quant_layers):
        handles.append(m.input_quant.register_forward_hook(make_sin_hook(name)))
        handles.append(m.register_forward_hook(make_in_hook(i)))

    # Hook wrapper output
    hooked_wrapper_ids = set()
    for idx, (wrap_mod, has_act, wrap_name) in wrappers.items():
        if id(wrap_mod) not in hooked_wrapper_ids:
            hooked_wrapper_ids.add(id(wrap_mod))
            handles.append(wrap_mod.register_forward_hook(make_out_hook(idx)))
        else:
            # Wrapper shared (không xảy ra thường), hook trực tiếp vào QuantConv2d output
            _, m = quant_layers[idx]
            handles.append(m.register_forward_hook(make_out_hook(idx)))

    # Forward pass
    img = cv2.imread(args.image)
    orig_h, orig_w = img.shape[:2]
    img = letterbox(img, args.imgsz)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_t = torch.from_numpy(np.ascontiguousarray(img.transpose(2,0,1))).unsqueeze(0).to(device)
    print(f"[INFO] Image: {args.image} ({orig_w}x{orig_h} -> {args.imgsz}x{args.imgsz})")

    with torch.no_grad():
        _ = model(img_t)

    for h in handles: h.remove()

    print(f"[INFO] Captured: S_in={len(s_in_map)}, float_in={len(float_in_data)}, float_out={len(float_out_data)}")

    # Build S_out map (S_out[i] = S_in[i+1] in exec_order)
    name_to_idx = {n: i for i, (n, _) in enumerate(quant_layers)}
    s_out_map   = {}
    for pos, name in enumerate(exec_order):
        info = metadata.get(f"layer_{name_to_idx[name]:02d}", {})
        if info.get('fp32_output', False):
            s_out_map[name] = None
        elif pos + 1 < len(exec_order):
            s_out_map[name] = s_in_map.get(exec_order[pos + 1])
        else:
            s_out_map[name] = None

    def to_int8(arr, scale):
        if scale is None or scale == 0: return arr.astype(np.int8)
        return np.round(arr.astype(np.float64) / scale).clip(-128, 127).astype(np.int8)

    print("[INFO] Saving golden tensors:")
    layer_shapes = {}
    in_saved = out_saved = skipped = 0

    for idx in range(total):
        name = quant_layers[idx][0]
        key  = f"layer_{idx:02d}"
        info = metadata.get(key, {})
        s_in    = s_in_map.get(name)
        s_out   = s_out_map.get(name)
        is_fp32 = info.get('fp32_output', False)

        fin = float_in_data.get(idx)
        if fin is None:
            print(f"  Layer {idx:02d}: no input captured"); continue

        in_int8  = to_int8(fin, s_in)
        in_shape = list(in_int8.shape)
        in_path  = os.path.join(args.output, f"L{idx:02d}_in_CHW.bin")
        in_int8.astype(np.int8).tofile(in_path)
        in_saved += 1

        out_int8 = None; out_shape = None
        if not is_fp32 and s_out is not None:
            fout = float_out_data.get(idx)
            if fout is not None:
                out_int8  = to_int8(fout, s_out)
                out_shape = list(out_int8.shape)

        if out_int8 is not None:
            out_path = os.path.join(args.output, f"L{idx:02d}_out_CHW.bin")
            out_int8.astype(np.int8).tofile(out_path)
            out_saved += 1
            print(f"  Layer {idx:02d} ({info.get('name','?'):32s}) in={in_shape} out={out_shape}")
        else:
            skipped += 1
            tag = '[fp32-out]' if is_fp32 else '[no-s_out]'
            print(f"  Layer {idx:02d} ({info.get('name','?'):32s}) in={in_shape} out=SKIP {tag}")

        layer_shapes[key] = {'in': in_shape, 'out': out_shape}

    with open(os.path.join(args.output, 'layer_shapes.json'), 'w') as f:
        json.dump(layer_shapes, f, indent=2)

    print(f"\n[SUMMARY] in={in_saved}  out={out_saved}  skipped={skipped}")
    print(f"[DONE] {args.output}/ has {in_saved+out_saved} .bin + layer_shapes.json")


if __name__ == '__main__':
    main()
