"""
fuse_and_export.py - PHASE 1 BUOC 5: BN Fusion + Export

Lam 4 viec:
1. BN Fusion: merge BatchNorm vao Conv weights/bias
2. Verify bit-accuracy: output truoc/sau fuse phai giong nhau
3. Export channel config JSON cho Phase 2 PE design
4. Export ONNX de inspect bang Netron

Cach chay:
    cd D:\yolov5-fpga\yolov5
    python fuse_and_export.py
"""
import torch
import torch.nn as nn
import json
import warnings
import contextlib
import io
from pathlib import Path
import sys

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from models.yolo import Detect

# ========================== CONFIG ==========================
WEIGHTS = "../checkpoints/yolov5s_pruned_newstem.pt"
OUTPUT_DIR = Path("runs/export")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FUSED_PT = OUTPUT_DIR / "pruned_fused.pt"
CONFIG_JSON = OUTPUT_DIR / "model_config.json"
ONNX_PATH = OUTPUT_DIR / "pruned_fused.onnx"
INPUT_SIZE = 416
# ============================================================


def fuse_conv_bn(conv, bn):
    """Fuse Conv2d + BatchNorm2d -> single Conv2d.

    Cong thuc:
        W_fused = W * gamma / sqrt(var + eps)
        b_fused = (b - mean) * gamma / sqrt(var + eps) + beta

    Where:
        W, b   = original conv weight, bias
        gamma  = bn.weight (scale)
        beta   = bn.bias (shift)
        mean   = bn.running_mean
        var    = bn.running_var
        eps    = bn.eps
    """
    fused = nn.Conv2d(
        in_channels=conv.in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=True,
    )

    # Fuse weights
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    
    # [ĐÃ SỬA]: Thêm .data để tránh lỗi in-place operation
    fused.weight.data.copy_(torch.mm(w_bn, w_conv).view(fused.weight.size()))

    # Fuse bias
    if conv.bias is not None:
        b_conv = conv.bias
    else:
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device)
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    
    # [ĐÃ SỬA]: Thêm .data để tránh lỗi in-place operation
    fused.bias.data.copy_(torch.matmul(w_bn, b_conv) + b_bn)

    return fused


def fuse_model(model):
    """Walk through model va fuse all Conv+BN pairs.

    YOLOv5 'Conv' module structure: self.conv (Conv2d), self.bn (BatchNorm2d), self.act
    Sau khi fuse: self.conv = fused, self.bn = Identity
    """
    fused_count = 0
    for module in model.modules():
        # Tim cac block co .conv va .bn
        if hasattr(module, 'conv') and hasattr(module, 'bn'):
            if isinstance(module.conv, nn.Conv2d) and isinstance(module.bn, nn.BatchNorm2d):
                module.conv = fuse_conv_bn(module.conv, module.bn)
                module.bn = nn.Identity()
                fused_count += 1
    return model, fused_count


def extract_channel_config(model):
    """Extract per-layer config cho Phase 2 PE design.

    Output format JSON:
        {
            "input_size": 416,
            "layers": [
                {
                    "idx": 0,
                    "name": "model.0.conv",
                    "type": "Conv2d" / "DepthwiseConv2d",
                    "Cin": 3,
                    "Cout": 32,
                    "kernel": 6,
                    "stride": 2,
                    "padding": 2,
                    "groups": 1,
                    "has_bias": true,
                    "params": 3520
                },
                ...
            ]
        }
    """
    config = {
        "input_size": INPUT_SIZE,
        "input_channels": 3,
        "layers": []
    }

    idx = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            is_dw = (module.groups > 1 and module.groups == module.in_channels)
            layer_info = {
                "idx": idx,
                "name": name,
                "type": "DepthwiseConv2d" if is_dw else "Conv2d",
                "Cin": module.in_channels,
                "Cout": module.out_channels,
                "kernel": module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size,
                "stride": module.stride[0] if isinstance(module.stride, tuple) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, tuple) else module.padding,
                "groups": module.groups,
                "has_bias": module.bias is not None,
                "params": sum(p.numel() for p in module.parameters()),
            }
            config["layers"].append(layer_info)
            idx += 1

    return config


def main():
    print("=" * 60)
    print("  PHASE 1 BUOC 5: BN FUSION + EXPORT")
    print("=" * 60)

    # ----- 1. Load -----
    print(f"\n[1/6] Loading: {WEIGHTS}")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    model.eval()
    print(f"  OK")

    # Stats
    params_before = sum(p.numel() for p in model.parameters())
    print(f"  Params: {params_before/1e6:.2f}M")

    # Count BN layers truoc khi fuse
    bn_count_before = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"  BatchNorm2d layers: {bn_count_before}")

    # ----- 2. Save reference output (truoc khi fuse) -----
    print(f"\n[2/6] Generating reference output...")
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    with torch.no_grad():
        ref_out = model(example)

    if isinstance(ref_out, (list, tuple)):
        ref_tensors = [o.clone() if torch.is_tensor(o) else None for o in ref_out]
        # Lay tensor dau tien (predictions)
        ref_main = ref_out[0] if torch.is_tensor(ref_out[0]) else ref_out[0][0]
    else:
        ref_main = ref_out.clone()

    print(f"  Reference output shape: {ref_main.shape if torch.is_tensor(ref_main) else 'tuple'}")

    # ----- 3. Fuse BN -----
    print(f"\n[3/6] Fusing BatchNorm into Conv...")
    model, fused_count = fuse_model(model)
    print(f"  Fused {fused_count} Conv+BN pairs")

    bn_count_after = sum(1 for m in model.modules() if isinstance(m, nn.BatchNorm2d))
    print(f"  Remaining BatchNorm2d: {bn_count_after} (should be 0)")

    # ----- 4. Verify bit-accuracy -----
    print(f"\n[4/6] Verifying bit-accuracy (truoc/sau fuse)...")
    model.eval()
    with torch.no_grad():
        new_out = model(example)

    if isinstance(new_out, (list, tuple)):
        new_main = new_out[0] if torch.is_tensor(new_out[0]) else new_out[0][0]
    else:
        new_main = new_out

    # Compare
    if torch.is_tensor(ref_main) and torch.is_tensor(new_main):
        max_diff = (ref_main - new_main).abs().max().item()
        mean_diff = (ref_main - new_main).abs().mean().item()
        print(f"  Max diff:  {max_diff:.2e}")
        print(f"  Mean diff: {mean_diff:.2e}")

        if max_diff < 1e-4:
            print(f"  OK - Bit-accuracy verified (< 1e-4)")
        elif max_diff < 1e-2:
            print(f"  WARNING - Diff hoi cao nhung chap nhan duoc")
        else:
            print(f"  FAIL - Diff qua cao, fuse co the sai")
            return
    else:
        print(f"  WARNING - Khong so sanh duoc tensor (shape khac)")

    # Stats sau fuse
    params_after = sum(p.numel() for p in model.parameters())
    print(f"\n  Params truoc fuse: {params_before:,}")
    print(f"  Params sau fuse:   {params_after:,}")
    print(f"  Difference:        {params_before - params_after:,} (la BN params bi merge)")

    # ----- 5. Export channel config -----
    print(f"\n[5/6] Exporting channel config...")
    config = extract_channel_config(model)

    with open(CONFIG_JSON, "w") as f:
        json.dump(config, f, indent=2)

    # In summary
    n_layers = len(config["layers"])
    n_dw = sum(1 for l in config["layers"] if l["type"] == "DepthwiseConv2d")
    n_regular = n_layers - n_dw
    n_round8 = sum(1 for l in config["layers"] if l["Cout"] % 8 == 0)

    print(f"  Total Conv2d layers: {n_layers}")
    print(f"    - Regular Conv:    {n_regular}")
    print(f"    - Depthwise Conv:  {n_dw}")
    print(f"  Layers Cout%8==0:    {n_round8}/{n_layers}")
    print(f"  Saved: {CONFIG_JSON}")

    # ----- 6. Save fused model + Export ONNX -----
    print(f"\n[6/6] Saving fused model + ONNX export...")

    # Save fused .pt
    ckpt["model"] = model
    torch.save(ckpt, FUSED_PT)
    print(f"  PyTorch: {FUSED_PT}")

    # Export ONNX
    print(f"  Exporting ONNX...")
    try:
        # Dat model ve eval va loai bo Detect post-processing cho ONNX clean
        model.eval()

        # Disable Detect inplace=True (gay van de voi ONNX)
        for m in model.modules():
            if isinstance(m, Detect):
                m.inplace = False
                m.dynamic = False
                m.export = True  # Export mode: only return raw outputs

        with contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(
                model,
                example,
                str(ONNX_PATH),
                verbose=False,
                opset_version=12,
                input_names=["images"],
                output_names=["output"],  # [ĐÃ SỬA]: Gom thành 1 output duy nhất
                dynamic_axes=None,
            )
        print(f"  ONNX:    {ONNX_PATH}")
        print(f"  Open in: https://netron.app/")
    except Exception as e:
        print(f"  ONNX export failed: {e}")
        print(f"  Khong critical - .pt file van OK cho cac buoc sau")

    # ----- Summary -----
    print("\n" + "=" * 60)
    print("  PHASE 1 BUOC 5 COMPLETED")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  1. {FUSED_PT}")
    print(f"  2. {CONFIG_JSON}")
    print(f"  3. {ONNX_PATH}")
    print(f"\nNext: Phase 2 - Custom PE Design (HLS + RTL)")
    print(f"  Input cho Phase 2: model_config.json + pruned_fused.pt")


if __name__ == "__main__":
    main()