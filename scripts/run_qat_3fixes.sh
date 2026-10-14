#!/bin/bash

# QAT Fine-tune with ALL 3 FIXES:
# Fix 1: Extended epochs (50 more from checkpoint)
# Fix 2: Quantize detect head layers 59/60/61 (INT8 instead of float32)
# Fix 3: Increase cls loss weight: 0.5 → 1.0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR/../yolov5" || exit 1

echo "=========================================="
echo "QAT Fine-tune with 3 Fixes Combined"
echo "=========================================="
echo "Fix 1: Extended epochs (50 more)"
echo "Fix 2: Quantize detect head (layers 59-61)"
echo "Fix 3: Increased cls loss weight (0.5→1.0)"
echo ""
echo "Starting training..."
echo ""

python train.py \
  --weights runs/quant/quant_state_dict.pt \
  --data data/pklot.yaml \
  --epochs 50 \
  --batch-size 8 \
  --img 416 \
  --device 0 \
  --hyp data/hyps/hyp.qat.yaml \
  --name qat_3fixes \
  --workers 0 \
  2>&1 | tee run_qat_3fixes.log

echo ""
echo "=========================================="
echo "Training complete!"
echo "Next: Extract weights with updated convert_to_quant.py"
echo "  python extract_quant_data.py"
echo "=========================================="
