#!/bin/bash
# pack_c3_sparse.sh -- Rev150 deploy: PTQ + 3 packer cho model sparse-c3.
# Chạy SAU khi finetune sparse-c3 xong (sparse_c3_best.pt) VÀ build150 xong.
# Output: extracted_c3/weights_unified_c3_ns12/ (c1x1 + dw + c3-sparse) sẵn scp lên board.
set -e
cd "$(dirname "$0")/.."
PY=venv/Scripts/python.exe
MODEL=yolov5/runs/sparse_c3/sparse_c3_best.pt
EXTR=yolov5/runs/quant/extracted_c3
IDXMAP="$(pwd)/yolov5/runs/sparse_c3/c3_idx_map.json"
OUT=weights_unified_c3_ns12

echo "==== [1/4] PTQ sparse (calibrate INT8 cho model c3-sparse) ===="
cd yolov5
$PY ptq_sparse.py --model runs/sparse_c3/sparse_c3_best.pt --out runs/quant/extracted_c3 --ncalib 32
cd ..

echo "==== [2/4] pack conv1x1 (NSPEC=12 NCFG_AUTO per-layer N) ===="
SPARSE_EXTRACTED=$EXTR OUT_NAME=$OUT NSPEC=12 NCFG_AUTO=1 $PY tools/make_unified_weights.py | tail -3

echo "==== [3/4] pack depthwise (NSPEC=12 — RTL N_SPEC_ACT=12, entry 49B/WT_BLK 480) ===="
SPARSE_EXTRACTED=$EXTR OUT_NAME=$OUT NSPEC=12 $PY tools/make_swc_weights.py | tail -3

echo "==== [4/4] pack conv3x3 SPARSE (C3_IDXMAP) ===="
SPARSE_EXTRACTED=$EXTR OUT_NAME=$OUT C3_IDXMAP="$IDXMAP" $PY tools/make_c3_weights.py | grep -E "SPARSE|dense|Total"

echo "==== DONE. Blobs: $EXTR/$OUT/ ===="
ls "$EXTR/$OUT/" | head
echo "scp lên board: rsync -az $EXTR/ kria:/home/ubuntu/yolov5/runs/quant/extracted_c3/"
