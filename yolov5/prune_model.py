"""
prune_model.py - ULTIMATE YOLOv5 PRUNING SCRIPT
"""
import torch
import torch.nn as nn
import torch_pruning as tp
from pathlib import Path
import sys
import warnings
import contextlib
import io

# Tắt toàn bộ các cảnh báo rác từ thư viện ptflops và PyTorch
warnings.filterwarnings("ignore")
import logging
logging.getLogger("ptflops").setLevel(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent))
from models.yolo import Detect

# ========================== CONFIG ==========================
WEIGHTS = "runs/train/pklot_lite_full/weights/best.pt"
OUTPUT_DIR = Path("runs/prune")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / "pruned.pt"
REPORT_PATH = OUTPUT_DIR / "pruning_report.txt"

PRUNE_RATIO = 0.5
ROUND_TO = 8           # Ràng buộc cho FPGA
INPUT_SIZE = 416
# ============================================================

def silent_count_ops(model, example):
    """Ém nhẹm mọi output thừa khi đếm FLOPs/Params"""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        flops, params = tp.utils.count_ops_and_params(model, example)
    return flops, params

def main():
    print("=" * 60)
    print("  YOLOV5 CHANNEL PRUNING (100% WORKING)")
    print("=" * 60)

    # ----- 1. Load Model & UNLOCK GRAPHS -----
    print(f"\n[1] Loading: {WEIGHTS}")
    ckpt = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
    model = ckpt["model"].float()
    
    # CHÌA KHÓA 1: Phục hồi Gradients để thư viện vẽ được đồ thị
    for p in model.parameters():
        p.requires_grad = True
        
    # CHÌA KHÓA 2: Ép về chế độ Train để vô hiệu hóa hàm biến đổi Bounding Box
    model.train()
    
    example_inputs = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)

    # Đo Before (Tạm dùng eval trên bản copy để có chỉ số chuẩn)
    temp_model = torch.load(WEIGHTS, map_location="cpu", weights_only=False)["model"].float().eval()
    flops_before, params_before = silent_count_ops(temp_model, example_inputs)
    del temp_model 
    
    print(f"\n[2] Before Pruning:")
    print(f"  Params: {params_before/1e6:.2f}M")
    print(f"  FLOPs:  {flops_before/1e9:.2f}G")

    # ----- 2. Define Ignored Layers -----
    ignored_layers = []
    for m in model.modules():
        if type(m).__name__ == "Detect" or isinstance(m, Detect):
            ignored_layers.append(m)
            # Khóa các output Conv2d cuối cùng (m.m)
            if hasattr(m, 'm'):
                for conv in m.m:  
                    ignored_layers.append(conv)
                    
    ignored_layers = list(set(ignored_layers))
    print(f"\n[3] Ignored {len(ignored_layers)} essential layers in Detect head")

    # ----- 3. Execute Pruning -----
    print(f"\n[4] Building Graph and Pruning ({int(PRUNE_RATIO*100)}%)...")
    imp = tp.importance.MagnitudeImportance(p=2) 
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance=imp,
        iterative_steps=1,
        ch_sparsity=PRUNE_RATIO, # Sử dụng ch_sparsity cho tp 1.6.0
        round_to=ROUND_TO,
        ignored_layers=ignored_layers,
    )

    # Thực hiện nhát cắt
    pruner.step()
    print("  => Pruning completed successfully!")

    # ----- 4. Verify & Count After -----
    model.eval() # Trả về eval để đếm và verify cho chuẩn xác
    
    flops_after, params_after = silent_count_ops(model, example_inputs)
    p_ratio = params_before / params_after if params_after > 0 else 0
    f_ratio = flops_before / flops_after if flops_after > 0 else 0

    print(f"\n[5] After Pruning:")
    print(f"  Params: {params_after/1e6:.2f}M  (giảm {p_ratio:.2f}x)")
    print(f"  FLOPs:  {flops_after/1e9:.2f}G  (giảm {f_ratio:.2f}x)")

    # Verify model còn chạy được không
    print(f"\n[6] Verifying forward pass...")
    with torch.no_grad():
        try:
            _ = model(example_inputs)
            print(f"  OK - Forward pass successful")
        except Exception as e:
            print(f"  FAIL: {e}")
            return

    # ----- 5. Save Model -----
    print(f"\n[7] Saving pruned model...")
    ckpt["model"] = model
    # Xóa các thông số tối ưu cũ để sẵn sàng cho bước Fine-tuning
    for key in ["optimizer", "ema", "updates"]:
        ckpt[key] = None
    ckpt["epoch"] = -1
    ckpt["best_fitness"] = 0.0
    
    torch.save(ckpt, OUTPUT_PATH)
    print(f"  Saved: {OUTPUT_PATH}")

    print("\n" + "=" * 60)
    print("  PRUNING SUCCESS - NEXT STEP: FINE-TUNE")
    print("=" * 60)

if __name__ == "__main__":
    main()