"""
convert_to_quant.py - Convert YOLOv5s-Lite-Pruned sang Brevitas QuantModel
============================================================================

FIX VERSION 2:
- Save state_dict thay vi model object (vi Brevitas Injector classes
  khong pickle duoc)
- Soften PTQ rel error warning (cao la expected vi activation scales
  chua duoc calibrate, QAT se fix)

QUAN TRONG: Cac script sau muon load lai quant model thi phai:
1. Load FP32 fused model goc (pruned_fused.pt)
2. Goi convert_model() de build quant structure
3. Load state_dict vao

Cach chay:
    cd D:\\yolov5-fpga\\yolov5
    python convert_to_quant.py
"""
import torch
import torch.nn as nn
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from models.yolo import Detect

import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerChannelFloat, Int8ActPerTensorFloat

# ========================== CONFIG ==========================
INPUT_WEIGHTS = "runs/export/pruned_fused.pt"
OUTPUT_DIR = Path("runs/quant")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# QUAN TRONG: Save 2 files
OUTPUT_STATE_DICT = OUTPUT_DIR / "quant_state_dict.pt"  # Quant scales + weights
OUTPUT_INFO = OUTPUT_DIR / "quant_info.txt"             # Metadata

INPUT_SIZE = 416
# ============================================================


def find_detect_convs(model):
    """Tim Conv2d trong Detect.m de quantize (FIX 2: quantize detect head)."""
    detect_convs = set()
    for m in model.modules():
        if isinstance(m, Detect):
            for conv in m.m:
                detect_convs.add(id(conv))
            break
    return detect_convs


def get_parent_module(model, dotted_name):
    """Tu dotted name -> (parent_module, attr_name)."""
    parts = dotted_name.split('.')
    parent = model
    for p in parts[:-1]:
        if p.isdigit():
            parent = parent[int(p)]
        else:
            parent = getattr(parent, p)
    return parent, parts[-1]


def replace_conv_with_quant(parent, attr_name, old_conv):
    """Tao QuantConv2d voi cung config va copy weights."""
    new_conv = qnn.QuantConv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=old_conv.bias is not None,
        weight_quant=Int8WeightPerChannelFloat,
        input_quant=Int8ActPerTensorFloat,
        return_quant_tensor=False,
    )
    
    with torch.no_grad():
        new_conv.weight.data = old_conv.weight.data.clone()
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
    
    if attr_name.isdigit():
        parent[int(attr_name)] = new_conv
    else:
        setattr(parent, attr_name, new_conv)
    
    return new_conv


def count_conv_types(model):
    """Dem chinh xac so nn.Conv2d (NOT QuantConv2d) va QuantConv2d."""
    n_pure_conv = sum(1 for m in model.modules() if type(m) is nn.Conv2d)
    n_quant_conv = sum(1 for m in model.modules() if isinstance(m, qnn.QuantConv2d))
    return n_pure_conv, n_quant_conv


def convert_model(model):
    """
    Convert tat ca nn.Conv2d sang qnn.QuantConv2d, INCLUDE Detect head (FIX 2).

    Function nay duoc EXPORT de cac script sau import va dung:
        from convert_to_quant import convert_model
    """
    detect_conv_ids = find_detect_convs(model)

    # IMPORTANT: dung type(m) is nn.Conv2d, KHONG phai isinstance
    # vi qnn.QuantConv2d ke thua tu nn.Conv2d
    # FIX 2: DO NOT skip detect head anymore - quantize tat ca!
    conv_list = []
    for name, m in model.named_modules():
        if type(m) is nn.Conv2d:  # QUANTIZE tat ca, including Detect head
            conv_list.append((name, m))

    converted = 0
    failed = 0
    dw_count = 0
    detect_converted = 0

    for name, old_conv in conv_list:
        try:
            parent, attr_name = get_parent_module(model, name)
            replace_conv_with_quant(parent, attr_name, old_conv)
            converted += 1

            if id(old_conv) in detect_conv_ids:
                detect_converted += 1

            if old_conv.groups > 1 and old_conv.groups == old_conv.in_channels:
                dw_count += 1

        except Exception as e:
            failed += 1
            print(f"  FAIL: {name} - {str(e)[:80]}")

    return converted, failed, dw_count, detect_converted


def main():
    print("=" * 60)
    print("  CONVERT MODEL TO BREVITAS QUANTMODEL (FIX V2)")
    print("=" * 60)
    
    # ----- 1. Load fused model -----
    print(f"\n[1] Loading: {INPUT_WEIGHTS}")
    if not Path(INPUT_WEIGHTS).exists():
        print(f"  FAIL - Khong tim thay {INPUT_WEIGHTS}")
        return
    
    ckpt = torch.load(INPUT_WEIGHTS, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    model.eval()
    print(f"  OK - Loaded")
    
    n_pure, _ = count_conv_types(model)
    print(f"  Before: nn.Conv2d={n_pure}")
    
    # ----- 2. FP32 forward pass baseline -----
    print(f"\n[2] FP32 forward pass...")
    example = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    with torch.no_grad():
        output_fp32 = model(example)
    
    if isinstance(output_fp32, tuple):
        ref_output = output_fp32[0]
    elif isinstance(output_fp32, list):
        ref_output = output_fp32[0]
    else:
        ref_output = output_fp32
    
    if isinstance(ref_output, torch.Tensor):
        print(f"  Output shape: {ref_output.shape}")
        print(f"  Output range: [{ref_output.min():.4f}, {ref_output.max():.4f}]")
        fp32_ref = ref_output.clone()
    else:
        fp32_ref = None
    
    # ----- 3. Convert -----
    print(f"\n[3] Converting to QuantModel (FIX 2: INCLUDING Detect head)...")
    converted, failed, dw_count, detect_converted = convert_model(model)

    print(f"  Detect head convs CONVERTED: {detect_converted} (layers 59/60/61)")
    print(f"  Total converted: {converted} (including {dw_count} depthwise)")
    print(f"  Failed:    {failed}")
    
    if failed > 0:
        print(f"  WARNING: {failed} layers failed")
        return
    
    n_pure, n_quant = count_conv_types(model)
    print(f"\n  After conversion:")
    print(f"    Pure nn.Conv2d:   {n_pure}  (should be 0 - all quantized!)")
    print(f"    qnn.QuantConv2d:  {n_quant}  (includes detect head now)")
    
    # ----- 4. Verify forward pass after conversion -----
    print(f"\n[4] Verify forward pass after conversion...")
    model.eval()
    try:
        with torch.no_grad():
            output_quant = model(example)
        
        if isinstance(output_quant, tuple):
            quant_ref = output_quant[0]
        elif isinstance(output_quant, list):
            quant_ref = output_quant[0]
        else:
            quant_ref = output_quant
        
        print(f"  OK - Quant output shape: {quant_ref.shape}")
    except Exception as e:
        print(f"  FAIL: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ----- 5. Compare FP32 vs Quant (PTQ baseline - JUST INFO) -----
    print(f"\n[5] PTQ baseline check (informational only)...")
    if fp32_ref is not None and isinstance(quant_ref, torch.Tensor):
        if fp32_ref.shape == quant_ref.shape:
            diff = (fp32_ref - quant_ref).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"  Max diff:   {max_diff:.6f}")
            print(f"  Mean diff:  {mean_diff:.6f}")
            print(f"\n  NOTE: PTQ rel error CAO la BINH THUONG vi:")
            print(f"  - Activation scales (input_quant) chua duoc calibrate")
            print(f"  - Brevitas init scales bang random/default values")
            print(f"  - QAT training se LEARN dung scales -> recover accuracy")
            print(f"  - Sub-step 1.4 (QAT) se fix van de nay")
    
    # ----- 6. Save STATE_DICT (KHONG save model object!) -----
    print(f"\n[6] Saving QUANT STATE_DICT...")
    print(f"  (KHONG save model object vi Brevitas Injector khong pickle)")
    
    # Save state dict only - approach nay work voi Brevitas
    state = {
        'state_dict': model.state_dict(),
        'epoch': -1,
        'best_fitness': 0.0,
        'training_results': None,
        'date': None,
    }
    
    torch.save(state, OUTPUT_STATE_DICT)
    print(f"  Saved state_dict: {OUTPUT_STATE_DICT}")
    print(f"  Total keys: {len(state['state_dict'])}")
    
    # ----- 7. Write info file -----
    with open(OUTPUT_INFO, "w") as f:
        f.write("BREVITAS QUANT MODEL - INFO\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Source FP32 model: {INPUT_WEIGHTS}\n")
        f.write(f"Quant state_dict:  {OUTPUT_STATE_DICT}\n\n")
        f.write(f"CONVERSION:\n")
        f.write(f"  Total Conv2d before: 62\n")
        f.write(f"  Detect head (converted): 3\n")
        f.write(f"  Converted to Quant:  {converted}\n")
        f.write(f"  Including depthwise: {dw_count}\n\n")
        f.write(f"AFTER:\n")
        f.write(f"  Pure nn.Conv2d:   {n_pure}\n")
        f.write(f"  qnn.QuantConv2d:  {n_quant}\n\n")
        f.write(f"HOW TO LOAD QUANT MODEL IN OTHER SCRIPTS:\n")
        f.write(f"  1. Import convert_model from convert_to_quant\n")
        f.write(f"     from convert_to_quant import convert_model\n")
        f.write(f"  2. Load FP32 fused model:\n")
        f.write(f"     ckpt = torch.load('{INPUT_WEIGHTS}', weights_only=False)\n")
        f.write(f"     model = ckpt['model'].float()\n")
        f.write(f"  3. Convert to quant structure:\n")
        f.write(f"     convert_model(model)\n")
        f.write(f"  4. Load state_dict:\n")
        f.write(f"     state = torch.load('{OUTPUT_STATE_DICT}', weights_only=True)\n")
        f.write(f"     model.load_state_dict(state['state_dict'], strict=False)\n")
    
    print(f"  Info file: {OUTPUT_INFO}")
    
    # ----- 8. Print samples -----
    print(f"\n[7] Sample QuantConv2d layers:")
    count = 0
    for name, m in model.named_modules():
        if isinstance(m, qnn.QuantConv2d):
            is_dw = m.groups > 1 and m.groups == m.in_channels
            tag = " (DW)" if is_dw else ""
            print(f"  {name}: in={m.in_channels} out={m.out_channels} k={m.kernel_size[0]}{tag}")
            count += 1
            if count >= 5:
                break
    
    print(f"\n[8] Verify Detect head now QUANTIZED (FIX 2):")
    for m in model.modules():
        if isinstance(m, Detect):
            for i, conv in enumerate(m.m):
                ctype = type(conv).__name__
                ok = "OK" if ctype == "QuantConv2d" else "WRONG"
                print(f"  Detect.m[{i}]: {ctype} [{ok}]")
            break
    
    print("\n" + "=" * 60)
    print("  CONVERSION COMPLETED")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  {OUTPUT_STATE_DICT}")
    print(f"  {OUTPUT_INFO}")
    print(f"\nNext step: Sub-step 1.4 (QAT Training)")
    print(f"QAT se LEARN activation scales -> recover accuracy")


if __name__ == "__main__":
    main()