# How-FPGA — Deploy & Run YOLOv5s-Lite N:M Accelerator on KV260
**Bản chốt production:** `bd_wrapper_rev174_resident_200.bit` — **VERIFIED 2026-07-23: 106 det · 0.77 s/frame · 1.30 FPS** (@200 MHz, WNS +0.004).
Toàn bộ lệnh dưới đây đã chạy thật trên board, không phải giả định.

---

## 0. Điều kiện & vị trí file (board đã có sẵn, KHÔNG cần copy nếu còn nguyên)
| Thành phần | Đường dẫn trên board |
|---|---|
| Bitstream | `~/rtl_bd/bd_wrapper_rev174_resident_200.bit` |
| Script host | `~/rtl_bd/inference_unified_v4.py`, `inference_cpu.py`, `cpu_conv.py` |
| Clock script | `~/rtl_bd/set_pl0.sh` |
| Driver DMA (prebuilt `.ko`) | `~/dma_region/dma_region.ko` |
| Weights INT8 N:M | `~/yolov5/runs/quant/extracted_c3/` (dir con: `weights_unified_c3_ns12c`) |
| Ảnh regression | `~/parking106.jpg` |
| Golden input (mặc định) | `~/golden/L00_in_CHW.bin` + `layer_shapes.json` |

Giao tiếp phần cứng: register qua `/dev/mem` (AXI base `0xA0000000`), DMA qua `/dev/dma_region` (64 MB). **Chạy bằng `sudo`** (cần `/dev/mem` + auto-load numpy NEON `~/.local`).

---

## A. Kết nối board (trên PC)
```bash
ssh kria                 # alias SSH trong ~/.ssh/config (tailnet, key-auth). Phải in hostname 'kria'.
```
Nếu timeout → bật nguồn board, chờ 60–90 s. LAN thay thế: `ssh ubuntu@<board-ip>`.

---

## B. Nạp bitstream + clock + driver (mỗi lần board reset)
```bash
sudo fpgautil -b ~/rtl_bd/bd_wrapper_rev174_resident_200.bit -f Full
sudo bash ~/rtl_bd/set_pl0.sh 200        # PL0_REF_CTRL = 0x01010500 → 200 MHz
sudo rmmod dma_region 2>/dev/null; sudo insmod ~/dma_region/dma_region.ko size_mb=64
sudo chmod 666 /dev/dma_region
```
Kiểm tra nhanh:
```bash
lsmod | grep dma_region                  # phải thấy dma_region
cat /sys/class/fpga_manager/fpga0/state   # = operating
```
⚠️ `set_pl0 200` **VOLATILE** — reboot board sẽ về 100 MHz (chậm 2×), phải chạy lại.

---

## C.1. Chạy inference production (LỆNH ĐẦY ĐỦ — verified 106 det / 1.30 FPS)
```bash
cd ~/rtl_bd && sudo env \
  CONF=0.15 C3V2=1 BI=1 FK=1 C3=1 C3W=1 DWBI=1 W11=1 RR4=1 RQD=1 OW=1 WDW=1 FR4=1 \
  INTWAIT=1 DWASYNC=1 RESACT=2 C3FAST=1 HEADQ=1 \
  SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 \
  UNI_NAME=weights_unified_c3_ns12c \
  python3 inference_unified_v4.py --image ~/parking106.jpg --frames 3
```
**Kỳ vọng:** `[*] 106 phát hiện` · `forward 0.77 s` (warm) · `FPS=1.30`.
Breakdown warm: conv1×1 kick 0.15 s (46 layer) · c3×3 kick 0.17 s HW (8 layer) · DW HW · head INT8 (HEADQ).

### Ý nghĩa bộ cờ (đừng bỏ cái nào)
- **Bộ NỀN (luôn bật)** — nếu thiếu là chậm/sai:
  `CONF=0.15` (ngưỡng → 106 det phủ hết ô đỗ) · `BI FK W11 RR4 RQD OW` (tăng tốc conv1×1: nếu thiếu kick 0.15→0.87 s) ·
  `C3 C3W` (conv3×3 chạy HW đúng: thiếu **C3W** thì C3 một mình ra **59 det SAI** + chậm 1.36 s) · `C3V2 DWBI WDW FR4` (biến thể c3/DW/requant).
- **Bộ LATENCY:** `INTWAIT` (nhả GIL cho DW-thread overlap) · `DWASYNC` (DW ‖ HW) · `RESACT=2` (resident-act L00→L02) · `C3FAST` (c3 seq không drain) · `HEADQ` (head L42/49/56 → INT8 HW).
- **Ảnh:** `--image ~/parking106.jpg`. Nếu KHÔNG có `--image` → dùng `golden/L00_in_CHW.bin` (ra 118 det@CONF0.15, KHÁC ảnh parking).

---

## C.2. Verify bit-exact HW vs CPU (tuỳ chọn)
```bash
cd ~/rtl_bd && sudo env \
  CONF=0.15 C3V2=1 BI=1 FK=1 C3=1 C3W=1 DWBI=1 W11=1 RR4=1 RQD=1 OW=1 WDW=1 FR4=1 HEADQ=1 \
  SPARSE_EXTRACTED=/home/ubuntu/yolov5/runs/quant/extracted_c3 \
  UNI_NAME=weights_unified_c3_ns12c \
  python3 inference_unified_v4.py --image ~/parking106.jpg --verify
```
Kỳ vọng: mọi lớp `max=0` (HW = CPU đúng từng bit). *(Không kèm `RESACT/DWASYNC/INTWAIT` khi `--verify` — script tự loại RESACT lúc verify.)*

---

## D. Xử lý sự cố
- **Kết quả sai/flaky sau khi nạp** mà reload driver/bit KHÔNG cứu → **`sudo reboot`** board (chờ 60–90 s), rồi làm lại mục B. (RTL state có thể corrupt DDR persist qua reload — chỉ reboot clear.)
- **`No such file /dev/dma_region`** → chưa `insmod` driver (mục B).
- **FPS ~0.4 thay vì 1.3** → thiếu bộ cờ NỀN (mục C.1). **59 det** → thiếu `C3W`. **118 det** → quên `--image` (đang dùng golden).
- **Chậm 2×** sau reboot → quên `set_pl0 200` (PL về 100 MHz).
- numpy NEON: script tự chèn `~/.local` khi chạy `sudo` (forward 2.2→0.8 s). Nếu chậm bất thường, kiểm `python3 -c "import numpy; print(numpy.__file__)"` → phải trỏ `~/.local/...`.

---

## E. (Chỉ khi board bị wipe) Copy file từ PC
```bash
cd /d/yolov5-fpga
scp rev76_build/bd_wrapper_rev174_resident_200.bit  kria:~/rtl_bd/
scp -r yolov5/runs/quant/extracted_c3               kria:~/yolov5/runs/quant/
scp tools/inference_unified_v4.py tools/inference_cpu.py tools/cpu_conv.py  kria:~/rtl_bd/
scp -r drivers/dma_region                           kria:~/dma_region     # rồi build: cd ~/dma_region && make
scp golden/L00_in_CHW.bin golden/layer_shapes.json  kria:~/golden/
scp test_images/input_image.jpg                     kria:~/parking106.jpg
```

---

## F. Ghi chú thiết kế (không cần build lại)
- Chỉ cần build lại Vivado nếu **đổi RTL** — hiện dùng rev174 sẵn (WNS +0.004 @200 MHz, 801 DSP, 768 MAC, 4.375 W).
- Layer census: 63 conv tổng → 59 INT8 xuống HW (43 conv1×1 + 8 conv3×3 + 8 DW), 4 lớp FP32 output (42/49/56/62) trên CPU.
- Nguồn lệnh gốc (đã mất bản cũ): dựng lại từ `DEBUG_LOG.md` + memory rev150. Bản này là bản VERIFY-lại 2026-07-23.
