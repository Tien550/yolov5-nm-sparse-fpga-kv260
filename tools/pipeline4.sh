#!/bin/bash
cd ~/rtl_bd
E='CONF=0.15 C3V2=1 BI=1 FK=1 C3=1 C3W=1 DWBI=1 W11=1 RR4=1 RQD=1 OW=1 WDW=1 FR4=1 SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 UNI_NAME=weights_unified_c3_ns12 PIPE=1'
N=${1:-8}
sudo -E taskset -c 3 env OPENBLAS_NUM_THREADS=1 $E PIPE_ACT_OFF=0x3000000 PIPE_OUT_OFF=0x3400000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames ${1:-8} >/tmp/p4.log 2>&1 &
D=$!
S=$(date +%s.%N)
sudo -E taskset -c 0 env OPENBLAS_NUM_THREADS=1 $E PIPE_ACT_OFF=0x0000000 PIPE_OUT_OFF=0x0C00000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames $N >/tmp/p1.log 2>&1 &
A=$!
sudo -E taskset -c 1 env OPENBLAS_NUM_THREADS=1 $E PIPE_ACT_OFF=0x2000000 PIPE_OUT_OFF=0x2400000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames $N >/tmp/p2.log 2>&1 &
B=$!
sudo -E taskset -c 2 env OPENBLAS_NUM_THREADS=1 $E PIPE_ACT_OFF=0x2800000 PIPE_OUT_OFF=0x2C00000 python3 inference_unified_v4.py --image /home/ubuntu/parking106.jpg --frames $N >/tmp/p3.log 2>&1 &
C=$!
wait $A $B $C $D
T=$(echo "$(date +%s.%N) - $S" | bc)
F=$((4*N))
echo "WALL=${T}s FRAMES=$F THROUGHPUT=$(echo "scale=3; $F/$T" | bc) fps"
grep -a 'phát hiện' /tmp/p3.log | tail -1
