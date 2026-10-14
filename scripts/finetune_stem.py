"""
Fine-tune sau khi thay stem 6x6 → 2x Conv3x3 (yolov5s-lite.yaml)
Chạy: python scripts/finetune_stem.py

Quy trình:
  1. Load checkpoint cũ (step3_pruned_finetuned_fp32.pt)
  2. Build model mới với stem mới từ yolov5s-lite.yaml
  3. Copy tất cả weights cũ sang model mới (trừ stem layer_00)
  4. Fine-tune 15-20 epochs trên PKLot
  5. Save: checkpoints/yolov5s_pruned_newstem.pt
"""
import sys
import torch
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict

ROOT = Path(__file__).resolve().parent.parent / "yolov5"
sys.path.insert(0, str(ROOT))

from models.yolo import Model
from utils.general import check_yaml
from utils.torch_utils import select_device

# ========================== CONFIG ==========================
OLD_CKPT   = Path(__file__).parent.parent / "checkpoints/step3_pruned_finetuned_fp32.pt"
NEW_CFG    = ROOT / "models/yolov5s-lite.yaml"
DATA_YAML  = ROOT / "data/pklot.yaml"
OUT_CKPT   = Path(__file__).parent.parent / "checkpoints/yolov5s_pruned_newstem.pt"
EPOCHS     = 20
BATCH_SIZE = 32
IMG_SIZE   = 416
DEVICE     = "0"   # GPU index hoặc "cpu"
# ============================================================


def copy_weights_except_stem(old_state: dict, new_model: nn.Module) -> int:
    """Copy weights từ old checkpoint sang new model, bỏ qua layer 0 (stem_a mới)."""
    new_state = new_model.state_dict()
    copied = 0
    skipped = []

    for new_key in new_state:
        # stem_a (layer 0) và stem_b (layer 1) là mới — không copy
        if new_key.startswith("model.0.") or new_key.startswith("model.1."):
            skipped.append(new_key)
            continue

        # Map layer index: new model có thêm 1 layer (stem_b) so với old
        # old model.1.* → new model.2.*  (dịch +1 từ layer 1 trở đi)
        parts = new_key.split(".")
        if len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                old_key = ".".join(["model", str(layer_idx - 1)] + parts[2:])
                if old_key in old_state and old_state[old_key].shape == new_state[new_key].shape:
                    new_state[new_key] = old_state[old_key]
                    copied += 1
                    continue
            except ValueError:
                pass

        # Thử map trực tiếp (cho các key không theo pattern model.N.*)
        if new_key in old_state and old_state[new_key].shape == new_state[new_key].shape:
            new_state[new_key] = old_state[new_key]
            copied += 1

    new_model.load_state_dict(new_state, strict=False)
    print(f"Copied {copied} tensors, skipped stem: {skipped[:4]}...")
    return copied


def main():
    device = select_device(DEVICE)
    print(f"Device: {device}")

    # 1. Load old checkpoint
    print(f"\nLoading old checkpoint: {OLD_CKPT}")
    ckpt = torch.load(OLD_CKPT, map_location="cpu", weights_only=False)
    old_state = ckpt["model"].float().state_dict() if hasattr(ckpt.get("model", None), "state_dict") else ckpt.get("model", ckpt)
    if hasattr(old_state, "state_dict"):
        old_state = old_state.state_dict()

    # 2. Build new model với stem mới
    print(f"Building new model from: {NEW_CFG}")
    new_model = Model(str(NEW_CFG), ch=3, nc=2).to(device)
    print(f"New model layers: {len(list(new_model.model))}")

    # 3. Copy weights (trừ stem)
    print("Copying weights (excluding stem)...")
    copy_weights_except_stem(old_state, new_model)

    # 4. Fine-tune bằng train.py (subprocess để tận dụng toàn bộ pipeline)
    import subprocess

    # Save model mới vào file tạm để train.py load qua --weights
    tmp_weights = ROOT / "runs/finetune_newstem_init.pt"
    tmp_weights.parent.mkdir(parents=True, exist_ok=True)
    save_ckpt = {"model": new_model, "epoch": -1, "best_fitness": 0.0}
    torch.save(save_ckpt, tmp_weights)

    cmd = [
        sys.executable, str(ROOT / "train.py"),
        "--weights",    str(tmp_weights),
        "--data",       str(DATA_YAML),
        "--epochs",     str(EPOCHS),
        "--batch-size", str(BATCH_SIZE),
        "--img",        str(IMG_SIZE),
        "--device",     DEVICE,
        "--hyp",        str(ROOT / "data/hyps/hyp.scratch-low.yaml"),
        "--name",       "finetune_newstem2",
        "--workers",    "0",   # tắt pin_memory / tránh lỗi DataLoader trên Windows
        # không truyền --cache → mặc định None (không cache RAM/disk)
    ]

    print(f"\nStarting fine-tune ({EPOCHS} epochs)...")
    print("Command:", " ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=str(ROOT))

    if result.returncode == 0:
        best = ROOT / "runs/train/finetune_newstem2/weights/best.pt"
        if best.exists():
            import shutil
            shutil.copy(best, OUT_CKPT)
            print(f"\nSaved to: {OUT_CKPT}")
        else:
            print(f"WARNING: best.pt not found at {best}")
    else:
        print(f"Training exited with code {result.returncode}")


if __name__ == "__main__":
    main()
