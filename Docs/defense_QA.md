# Q&A Phòng Thủ Luận Văn — Câu hỏi hội đồng & Trả lời chi tiết
Đề tài: Thiết kế bộ tăng tốc phần cứng YOLOv5 trên FPGA dùng cắt tỉa thưa cấu trúc N:M.
Mẹo: câu **in đậm** ở cuối mỗi trả lời là câu chốt nên nói ra miệng.

---

## NHÓM A — MÔ HÌNH & NÉN

### A1. mAP giữ 98.7% nhưng IoU (dense→INT8) chỉ 0.600 — có mâu thuẫn không?
Không. Đây là **hai phép đo khác nhau**:
- **mAP 0.987** = độ chính xác trên tập test, so với **nhãn thật (ground truth)**, ngưỡng IoU 0.5. Nó trả lời "có phát hiện đúng ô đỗ xe không". Model sau nén vẫn phát hiện gần như trọn vẹn.
- **IoU 0.600** = so **hộp-với-hộp** giữa model nén và model FP32-dense **trên cùng một ảnh** (không phải nhãn thật). Sau nén, các hộp **dịch nhẹ** và **số hộp tăng (84→93)**, nên IoU box-to-box giảm — nhưng hộp vẫn nằm đúng trên xe, nên độ chính xác (mAP) vẫn cao.
- **Chốt: "Nén làm hộp xê dịch chút ít, chứ không làm sai phát hiện. Và vì FPGA bit-exact với CPU, phần cứng không thêm bất kỳ sai số nào."**

### A2. "40× nén" đến từ đâu? Kiến trúc chỉ giảm ~1.4×?
Đúng, và nhóm nói rõ điều này. 40× là **tích của 3 tầng**:
- Kiến trúc Lite: **~1.4× tham số / ~1.5× FLOP** (thay conv bằng depthwise-separable, giảm depth, đổi activation).
- Lượng tử hoá INT8: **~2.3×** (32-bit → 8-bit + gộp BN).
- Cắt tỉa N:M 71.9%: **~3.56×**.
- Nhân lại: file trọng số **14.4 MB → 0.36 MB ≈ 40×**.
- **Chốt: "Phần nén lớn đến từ lượng tử hoá + cắt tỉa; kiến trúc chủ yếu để thân thiện phần cứng, không phải để nhỏ."**

### A3. Vì sao chọn N:M mà không cắt tỉa không cấu trúc (nén cao hơn)?
Cắt tỉa không cấu trúc cho nén cao nhất nhưng trọng số nằm **rải rác ngẫu nhiên** → phần cứng phải dùng bảng chỉ số để tìm, overhead ăn hết tốc độ. N:M giữ N trọng số mạnh nhất mỗi khối 16 → **vừa mịn vừa đều đặn**: phần cứng biết trước số phép nhân, không rẽ nhánh, ánh xạ thẳng vào PE.
- **Chốt: "N:M là điểm cân bằng giữa độ mịn của không-cấu-trúc và tính đều đặn thân thiện phần cứng."**

### A4. DominoSearch hoạt động thế nào? Vì sao mỗi lớp một N?
Không lớp nào chịu cắt tỉa như nhau. DominoSearch bắt đầu từ mạng **dày đặc (N=16)**, rồi **hạ N từng bước**, mỗi bước chọn lớp "chịu đựng tốt nhất", cho tới khi đạt ngân sách phần cứng. Kết quả: lớp mong manh (model.1/2/4 — downsample sớm) giữ N=16, lớp khoẻ hạ tới N=2. Chỉ dùng thống kê trọng số, **không train lại từ đầu**.
- **Chốt: "Nó bắt đầu dày đặc rồi chỉ cắt ở nơi mạng chịu được — như xô đổ domino từng quân một."**

### A5. Vì sao INT8 mà không INT4 (nhanh gấp đôi)?
INT8 giữ gần trọn độ chính xác (**0.9943 vs FP32 0.9945**, mất 0.02 điểm). INT4 có thể tăng tốc nhưng rủi ro rớt chính xác cao, và **2 bài báo chuẩn của đề tài đều INT8 N:M**. INT4 là hướng tương lai, ngoài phạm vi.
- **Chốt: "INT8 là điểm ngọt độ-chính-xác/hiệu-năng; INT4 để dành tương lai."**

### A6. Trên slide sensitivity, "dense = 0.1934" — sao không phải 0.9945?
Đó là baseline trên **một tập con val nhỏ**, chỉ để **xếp hạng độ nhạy** các lớp (lớp nào rớt nhiều khi cắt = nhạy). Nó **không phải** mAP toàn tập. mAP thật của model dày đặc là **0.9945**.
- **Chốt: "0.1934 chỉ dùng để so tương đối giữa các lớp, không phải điểm accuracy."**

### A7. QAT khác PTQ chỗ nào?
- **PTQ** = lượng tử hoá sau khi train xong (đơn giản, rớt chính xác — mAP@.5:.95 chỉ 0.7421).
- **QAT** = train tiếp trong khi model "biết" mình đang bị làm tròn, học bù sai số → mAP hồi về **0.9943**.
- **Chốt: "QAT dạy mạng sống chung với việc làm tròn, nên hồi 99.98% độ chính xác."**

---

## NHÓM B — PHẦN CỨNG

### B1. Vì sao 768 MAC mà không dùng hết 1.248 DSP?
Nút thắt **không phải DSP mà là bộ nhớ trong chip**: BRAM 94%, URAM 88%. 768 MAC (**801 DSP** kể cả requant) là cấu hình lớn nhất còn vừa với BRAM/URAM. Thêm MAC cũng không có bộ nhớ nuôi dữ liệu.
- **Chốt: "Giới hạn là bộ nhớ on-chip, không phải DSP — 768 MAC là điểm bão hoà thực tế."**

### B2. Sparsity được khai thác thế nào (không phí chu kỳ)?
Bộ giải mã chỉ **bung các trọng số khác 0**. Nhiều accelerator vẫn "bước qua" ô 0 (tốn 1 chu kỳ dù nhân 0). Của nhóm thì **số 0 bị loại trước khi vào engine** — nên 768 bộ nhân không bao giờ nhân 0.
- **Chốt: "71.9% trọng số bị cắt = 71.9% công việc engine không phải làm — đó là chỗ cắt tỉa biến thành tốc độ thật."**

### B3. COO và Bitmap khác gì, vì sao 2 định dạng?
- **COO**: lưu cặp (vị trí, giá trị) — rẻ khi **ít** trọng số. Ví dụ N=3: 3×12 = 36 bit.
- **Bitmap**: lưu mặt nạ 16-bit + giá trị — rẻ khi **nhiều** trọng số. N=3: 16+24 = 40 bit.
- Điểm hoà: **N=4 (48 bit = 48 bit)**. Decoder tự chọn cái nhỏ hơn: **use_bitmap khi N≥5**. Cả hai cho **cùng một đầu ra** nên engine phía sau không cần biết.
- **Chốt: "Hai định dạng để luôn chọn cách nén rẻ nhất, nhưng chỉ một giao diện đầu ra."**

### B4. Vì sao dùng LeakyReLU thay SiLU?
SiLU là hàm cong, phần cứng phải xấp xỉ tốn kém. LeakyReLU(0.1) chỉ là **1 phép nhân hằng + dịch bit** (≈ ×205 rồi >>11), tốn **1 DSP**, sai số ~0.1%. Sau QAT gần như không mất chính xác.
- **Chốt: "Đổi hàm cong tốn kém lấy hai đoạn thẳng — một DSP là xong."**

### B5. Vì sao 60% công suất là PS (ARM)? Accelerator không làm gì nhiều?
4.375 W tổng gồm cả **ARM (2.425 W)** chạy Linux + điều phối + head FP32 + NMS. Riêng **IP accelerator (PL) chỉ ~1.633 W**. PS cao vì đang gánh phần host + detect head; đẩy head xuống FPGA (tương lai) sẽ giảm phần PS. Con số hiệu quả năng lượng nhóm báo là **cho toàn board** (khắt khe hơn, không "ăn gian").
- **Chốt: "Riêng phần cứng tăng tốc chỉ 1.6 W; 60% là ARM đang làm việc điều phối và head."**

### B6. "Kiến trúc thống nhất (unified)" nghĩa là gì?
**Một** datapath xử lý cả conv1×1, conv3×3, depthwise — chọn bằng thanh ghi **LAYER_TYPE**. Thay vì 3 mạch riêng (tốn tài nguyên), làm 1 engine linh hoạt → đủ nhỏ để vừa board (23 module, ~7.000 dòng Verilog).
- **Chốt: "Một engine cho cả ba loại conv, chọn bằng một thanh ghi — đó là cách để vừa board."**

### B7. N biến thiên theo lớp thì phần cứng cố định xử lý sao?
Engine thiết kế cho **N tối đa (16)**; decoder phát đúng N phần tử mỗi khối theo `n_cfg`. Khi N nhỏ, engine **xong sớm (N chu kỳ)** → tiết kiệm. Không cần mạch riêng cho mỗi N.
- **Chốt: "Một mạch, N nạp từ config; N nhỏ thì chạy nhanh hơn."**

---

## NHÓM C — KẾT QUẢ & KIỂM CHỨNG

### C1. "Bit-exact" nghĩa là gì và chứng minh thế nào?
FPGA cho ra **đúng từng bit** giống model INT8 trên máy tính. Nhóm viết **14 testbench**, so từng phần tử của **cả 59 lớp** với reference Python vàng — khớp **100%, 0 lỗi LSB**. Nghĩa là phần cứng không có bug số học ẩn nào.
- **Chốt: "Chúng em so từng con số của 59 lớp — lệch một bit là bắt được ngay; kết quả không lệch bit nào."**

### C2. Vì sao FPS chỉ 1.30 — có phải real-time không?
Thẳng thắn: 1.30 FPS **chưa real-time** cho video. Lý do: **~55% thời gian là di chuyển dữ liệu** (DMA + host điều phối từng lớp), và detect head còn chạy CPU. Bản thân engine tính nhanh (153.6 GOPS đỉnh). **Nút thắt là dữ liệu, không phải tính toán.** Hướng khắc phục đã có thiết kế: double-buffering (chồng lấp nạp dữ liệu với tính) + runtime C++ → cắt 30–50% thời gian chờ. Đề tài ưu tiên chứng minh **kiến trúc đúng (bit-exact)** trước.
- **Chốt: "Engine không chậm — dữ liệu chậm. Double-buffering là bước tiếp theo đã có sẵn thiết kế."**

### C3. Số phát hiện tăng (84→93) — nén mà tăng detection?
Nén không nhất thiết giảm số hộp — nó làm dịch/nhân đôi vài hộp. 84 là model FP32-dense cha, 93 là INT8-pruned. Vài hộp tăng do ngưỡng/độ nhạy đổi sau lượng tử hoá. Quan trọng: **mAP vẫn 0.987**, tức phát hiện vẫn đúng.
- **Chốt: "Số hộp thay đổi nhẹ, nhưng phát hiện vẫn đúng — mAP vẫn 0.987."**

### C4. Timing WNS chỉ +0.004 ns — sát quá, rủi ro không?
+0.004 ns là **dương** → **đóng timing ở 200 MHz**, thiết kế hợp lệ, board chạy thật ổn định. Sát nhưng đủ; có bản dự phòng tần số thấp hơn nếu cần biên an toàn lớn hơn.
- **Chốt: "Dương là đạt — và board đã chạy 200 MHz bit-exact trên thực tế."**

### C5. Vì sao 63 conv nhưng chỉ 59 xuống FPGA?
63 conv tổng. **59 lớp INT8** (43 conv1×1 + 8 conv3×3 + 8 depthwise) xuống FPGA. **4 lớp output FP32** (detect head) chạy CPU vì cần dấu phẩy động + decode phức tạp.
- **Chốt: "59 lớp nặng xuống FPGA; 4 lớp head nhẹ nhưng phức tạp nên để CPU."**

---

## NHÓM D — HẠN CHẾ & PHÁT TRIỂN

### D1. Chỉ 2 lớp trên 1 dataset — làm sao biết tổng quát?
Đúng, hạn chế đã nêu rõ. PKLot 2 lớp là bài toán **được định nghĩa rõ để kiểm chứng phần cứng sạch sẽ**. Đóng góp của nhóm là **phần cứng** (bit-exact, tiết kiệm điện), không phải độ chính xác. Engine **không phụ thuộc số lớp** — mở rộng đa lớp (COCO/VOC) chỉ cần đổi head, engine giữ nguyên.
- **Chốt: "Bài toán 2 lớp để kiểm chứng phần cứng sạch; kiến trúc không giới hạn số lớp."**

### D2. BRAM 94% — sắp hết bộ nhớ, mở rộng model lớn hơn được không?
Đúng, bộ nhớ on-chip là nút thắt (lý do chọn 768 MAC). Model lớn hơn cần **streaming/tiling activation** thay vì giữ toàn bộ on-chip — đó là hướng mở rộng. Trên KV260 hiện tại, thiết kế đã tối ưu để vừa.
- **Chốt: "Model lớn hơn cần chiến lược streaming activation — hướng phát triển rõ ràng."**

### D3. Detect head chạy CPU — vì sao không xuống FPGA?
Head chỉ 4 lớp FP32 output, tính nhẹ nhưng cần **dấu phẩy động + logic decode/NMS phức tạp** → chạy CPU đơn giản, ít tốn tài nguyên FPGA. 59 lớp **nặng** (conv) đã xuống FPGA.
- **Chốt: "Phần nặng đã xuống FPGA; head nhẹ để CPU cho đơn giản — đẩy nốt là bước sau."**

---

## NHÓM E — PHẢN BIỆN / KHÓ

### E1. So sánh benchmark khác model, khác board — có ý nghĩa gì?
Nhóm **thừa nhận ngay trên slide**: không phải so sánh công bằng tuyệt đối (ResNet-50 vs YOLOv5s-Lite; ZCU102 274K LUT vs KV260 117K LUT). Trình bày như **mốc tham chiếu**, không phải "thắng". Điểm mạnh rõ ràng: **hiệu quả năng lượng (~94 vs 13.7 GOPS/W)** và tiết kiệm tài nguyên (62% vs 84% LUT) — hai chỉ số quan trọng nhất ở edge.
- **Chốt: "Đây là mốc tham chiếu; điểm mạnh không chối cãi là hiệu quả năng lượng và tài nguyên."**

### E2. Đóng góp thực sự của nhóm là gì (so với 2 bài báo)?
2 bài (DominoSearch + Xie 2023) đưa ra **thuật toán và ý tưởng** N:M. Đóng góp của nhóm là **hiện thực hoá phần cứng hoàn chỉnh, đầu-cuối, trên silicon thật**:
1. Decoder N:M **kép COO/Bitmap tự chọn** theo lớp.
2. Engine **hợp nhất** 3 loại conv với **N biến thiên** theo lớp.
3. Triển khai **bit-exact verified** trên KV260 thật, đạt 200 MHz, 4.375 W.
- **Chốt: "Chúng em biến lý thuyết N:M thành một mạch chạy được, kiểm chứng đúng từng bit trên board thật."**

### E3. Ứng dụng thực tế là gì?
Bài toán đỗ xe là ví dụ **kiểm chứng**. Đích hướng tới là **latency-critical**, ví dụ kiểm tra lỗi bo mạch PCB — cần độ trễ thấp, ổn định, điện thấp trên thiết bị nhúng. FPGA phù hợp vì **độ trễ tất định + tiết kiệm điện**.
- **Chốt: "Đích là bài toán độ-trễ-thấp như kiểm lỗi PCB, nơi FPGA hơn hẳn GPU/CPU về độ trễ ổn định và điện năng."**

### E4. Nếu chỉ 1.6 W cho PL mà FPS thấp, hiệu quả thật sự có tốt không?
Hiệu quả **năng lượng trên mỗi phép tính** (GOPS/W) rất tốt vì engine chạy đúng và tiết kiệm. FPS thấp là do **overhead di chuyển dữ liệu ở mức hệ thống**, không phải engine kém hiệu quả. Khi bỏ nút thắt dữ liệu (double-buffering), cùng công suất sẽ cho FPS cao hơn nhiều.
- **Chốt: "Hiệu quả engine cao; FPS bị chặn bởi khâu dữ liệu, không phải khâu tính."**

### E5. Làm sao chắc số 1.30 FPS và bit-exact là thật (không phải mô phỏng)?
Cả hai **đo trên board KV260 thật**: bitstream + kernel module + trọng số nạp qua SCP, chạy inference thực, so kết quả với reference. Bit-exact được kiểm bằng 14 testbench (mô phỏng) **và** đối chiếu kết quả board với CPU-INT8 (IoU = 1.00).
- **Chốt: "Số liệu đo trên silicon thật, không phải mô phỏng — và khớp với reference đúng từng bit."**

---

## 3 CÂU DỄ BỊ HỎI NHẤT — học thuộc kỹ
1. **FPS thấp** → "dữ liệu là nút thắt, không phải tính toán; double-buffering đã có thiết kế."
2. **IoU 0.600 vs mAP 0.987** → "hai phép đo khác nhau: một so nhãn thật (accuracy cao), một so hộp-với-hộp model gốc (nén làm dịch hộp); phần cứng bit-exact không thêm sai số."
3. **Đóng góp** → "biến thuật toán N:M thành mạch phần cứng đầu-cuối, kiểm chứng bit-exact trên board thật, tiết kiệm điện."
