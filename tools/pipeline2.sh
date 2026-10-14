#!/bin/bash
# pipeline2.sh <frames> — 2 process lệch pha, chia vùng DMA, khoá HW per-layer
cd ~/rtl_bd
E='CONF=0.15 C3V2=1 BI=1 FK=1 C3=1 C3W=1 DWBI=1 W11=1 RR4=1 RQD=1 OW=1 WDW=1 FR4=1 SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 UNI_NAME=weights_unified_c3_ns12 PIPE=1'
N=${1:-6}
S=$(date +%s.%N)
sudo -E taskset -c 0,1 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $E PIPE_ACT_OFF=0x0000000 PIPE_OUT_OFF=0x0C00000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames $N >/tmp/p1.log 2>&1 &
A=$!
sudo -E taskset -c 2,3 env OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 $E PIPE_ACT_OFF=0x2000000 PIPE_OUT_OFF=0x2400000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames $N >/tmp/p2.log 2>&1 &
B=$!
wait $A $B
T=$(echo "$(date +%s.%N) - $S" | bc)
F=$((2*N))
echo "WALL=${T}s FRAMES=$F THROUGHPUT=$(echo "scale=3; $F/$T" | bc) fps (gồm cold-start)"
grep -a 'phát hiện' /tmp/p1.log | tail -1; grep -a 'phát hiện' /tmp/p2.log | tail -1
grep -a 'FPS=' /tmp/p1.log | tail -2; grep -a 'FPS=' /tmp/p2.log | tail -2
