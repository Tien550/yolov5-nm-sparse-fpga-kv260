# Tài liệu thiết kế — LEVEL B (I-SPEC paper-faithful, KV260)

> Trạng thái: **PRE-CODE**. Dò kiến trúc tới từng dòng 2026-07-02. Chờ "go" để code.
> Nguồn: bài báo I-SPEC (Micromachines 2023, 14, 528). Build hiện tại = Rev135.
> Ràng buộc tuyệt đối: **BỎ SYSTOLIC** — chỉ dùng đường `unified_cnn_top` (i_spec).

---

## 0. TÓM TẮT ĐIỀU HÀNH

| Mục | Giá trị |
|---|---|
| FPS hiện tại (416×416, warm) | **0.14** (7.12 s/frame), 156 detections |
| Nút cổ chai | HW kick 4.90 s = 69% → **DMA-bound** (ACT đọc lại N_OCB lần; on-chip mem trống 99%) |
| Đòn bẩy | B-i ACT-stationary + B-w wider array + **B-c3 conv3×3 HW** + clock 166 MHz |
| FPS mục tiêu | B-i+B-w ~0.33 (c3×3 CPU thành bottleneck) → **+B-c3 ~0.5** → +166MHz ~0.65 → ~5 (nếu input 160²) |
| Chiến lược build | Feature-flag, **2 build** (125 MHz verify từng cờ → 166/200 MHz) |
| Quyết định user | ✅ **Option C**: 12 SPEC + prune R1 + **retrain conv3×3 → sparse N:16** (paper-faithful, mọi layer sparse, 1 mảng chung). 166 MHz (thử 200), feature-flag. Thêm ML track (retrain, risk mAP, golden mới). |

---

## 1. HIỆN TRẠNG (Rev135, đã verify bằng report impl thực)

### 1.1 Build top thực = `unified_cnn_top` (KHÔNG phải cnn_acc_top)
- `component.xml` → `modelName = unified_cnn_top` = `dma_engine` + `unified_acc_top` + `layer_sequencer`.
- BD cell tên "cnn_acc_top" (giữ ổn định) nhưng RTL top là `unified_cnn_top`.
- `cnn_acc_top.v` = systolic top CŨ (rev114, 113 det/0.8s), **KHÔNG build** — bỏ qua.
- Fingerprint xác nhận: BRAM 0.35% (in_bram_wide = LUT-RAM) chỉ khớp unified, không khớp systolic.

### 1.2 Tài nguyên thực (impl_1/bd_wrapper_utilization_placed.rpt, build 1/7)
| Tài nguyên | Dùng | Tổng | % | Ghi chú |
|---|---|---|---|---|
| DSP | 545 | 1248 | 43.7% | dư 703 |
| BRAM | 0.5 | 144 | 0.35% | gần như trống |
| URAM | 4 | 64 | 6.25% | gần như trống |
| LUT | 45466 | 117120 | 38.8% | dư 61% |
| FF | 35423 | 234240 | 15.1% | dư 85% |
| **WNS** | **−0.046 ns** | | | **đang FAIL** ở clock hiện tại |

- Clock hiện tại = **125 MHz** (`rebuild_bitstream.tcl:50` PL_FREQ_MHZ 125).
- Phân rã DSP: **256** (i_spec×4 MAC) + **256** (conv_tile_r1 = R1 conv3×3, DORMANT) + **~33** (rr/misc).

### 1.3 Breakdown forward 7.12 s (warm)
| Phần | Thời gian | % |
|---|---|---|
| HW conv1×1 kick | 4.90 s | 69% |
| CPU conv3×3 (vectorized) | 1.45 s | 20% |
| HW DW (kick+pack+deint) | 0.43 s | 6% |
| FP32 (L42/49/56/62) + khác | ~0.34 s | 5% |

### 1.4 Gốc rễ DMA-bound (dò `layer_sequencer.v`)
Vòng lặp **oc-OUTER** (L157-169): với MỖI oc-block, sweep tất cả pos, mỗi pos DMA lại toàn bộ N_ICB ACT chunk từ DDR. conv1×1 ACT **không phụ thuộc oc** → **ACT bị đọc lại N_OCB lần** (Cout=512 → 128×). On-chip mem trống 99% ⇒ mọi tile round-trip DDR. Đây là nguyên nhân kick 4.90 s.

---

## 2. BẢN ĐỒ KIẾN TRÚC (verified line-by-line)

```
unified_cnn_top
├─ dma_engine (426d)   AXI4 master 128b. tgt 2-bit {0=ACT→dma_in_w128(6b addr), 1=WT→dma_wt_w128
│                       (+ls_wt_uram_off cộng ngoài), 2=OUT(wide out128), 3=DESC}. Burst tới 4KB.
├─ unified_acc_top (~1400d)
│   ├─ axi_lite_slave   reg 9-bit (AXI_OFF_*), slot 16-aligned ĐẦY tới 0x1E0 (0x1F0 chết)
│   ├─ config_decoder   conv_type[1:0]=1x1/3x3/dw → swc_en/eff_poc/eff_accu_mode/dw_mode (comb)
│   ├─ sliding_win_ctrl u_swc  (chỉ 3x3/dw)
│   ├─ i_spec u_spec ×4 (generate si<4 HARD-CODED)  →  mỗi SPEC = 64 DSP
│   │     = 4×weight_decoder(COO idx CHUNG, val RIÊNG/group=poc) + depth_shuffle(512FF)
│   │       + mul_group_array(64×mac_unit=64 DSP) + psum_array(0 DSP, 64 acc FF)
│   ├─ accu_top u_accu (158d)  adder 1/2/4-in, COMB 0 DSP, ports PHẲNG 4-SPEC. (1x1 KHÔNG dùng)
│   ├─ rr_unit u_rr ×1  requant psum×M0>>sh + leaky0.1, ~3 DSP, 3-stage. WRITEBACK NỐI TIẾP.
│   ├─ in_bram_wide     1 tile 1024B, LUT-RAM (→BRAM~0), ping-pong (pp_in_sel). rd8 + rd128.
│   ├─ wt_uram          WT-stationary URAM 128KB (8192×128b), wide-wr + byte-rd RD_LAT=2, +ls_wt_uram_off
│   ├─ bram_ping_pong u_bram_out  OUT 784×8b, PINGPONG=0, RD_LAT=1
│   └─ conv_kernel_r1 → conv_tile_r1  = 256-MAC (256 DSP) R1 conv3x3. DORMANT khi conv1x1 (r1_mode=reg_pad[10]=0)
└─ layer_sequencer (176d)  vòng oc→pos→ic on-chip; WT nạp 1×/oc (LS_WTU); ACT re-DMA mỗi oc ← GỐC RỄ
```

### 2.1 DSP/module (mac_unit = 1 DSP, verified)
| Module | DSP | Ghi chú |
|---|---|---|
| mac_unit | 1 | multiply-only (use_dsp), accumulate ở psum FF |
| mul_group_array | 64 | G4×P16, weight broadcast/group (paper Fig 2b) |
| psum_array | 0 | 64 accumulator FF |
| i_spec ×4 | 256 | MAC |
| conv_tile_r1 (R1) | 256 | **dormant** → prune được |
| rr + misc | ~33 | |

### 2.2 Driver dma.ko (`/home/ubuntu/dma_region/dma_region.c`, 209d)
CMA 1GB + `dma_alloc_pages` + remap **Write-Combine**. Read ~400 MB/s, Write ~1200 MB/s. IOCTL SYNC_TO/FROM (no-op WC). `/dev/dma_region`, `phys_base` qua `/sys/module/dma_region/parameters/phys_base`, `insmod size_mb=128`.

### 2.3 Register map (cnn_defines.vh)
- Offset 16-aligned dùng hết 0x000→0x1E0 (AP×4, CIN..ACT_CHUNK). **0x1F0 = slot trống duy nhất nhưng chết** (netseq addr-wall).
- Params: `M=16 G=4 P=16 N_SPEC_KV=4`; **có sẵn `N_SPEC_ZCU=16`**.
- `CONV_TYPE_1X1/3X3/DW = 0/1/2`.

---

## 3. CHIẾN LƯỢC THỰC THI

### 3.1 Feature-flag → 2 build (hòa giải "song song" + "chống-race")
Cả 4 tính năng viết song song, **gate bằng `REG_FEAT_EN`** (mặc định OFF = Rev135 verified). Verify cô lập từng cờ ở 125 MHz (không race), rồi 1 build đổi clock.
- **REG_FEAT_EN = bit[4:1] của `AXI_OFF_LAYER_MODE (0x190)`** (hiện chỉ bit0) — KHÔNG cần offset mới.
  - bit1 = B-i (ACT-stationary), bit2 = B-w-runtime (nếu cần), bit3/4 dự phòng.

### 3.2 Thứ tự (theo rủi ro/lợi ích) — Option C (paper-faithful đầy đủ)
1. **B-i ACT-stationary** — an toàn nhất, trị bệnh DMA-bound; phục vụ cả conv1×1 + conv3×3.
2. **B-w 12 SPEC + prune R1** — conv1×1 3×; verify model hiện tại (~801 DSP 64%).
3. **[ML track song song] retrain conv3×3 → N:16 sparse** — model + golden mới, verify mAP.
4. **B-c3 conv3×3-HW qua i_spec swc sparse** — rework swc (WT-stat + act_uram + SHIFT) + pack 9-tap COO; verify model sparse mới.
5. **Clock 166 MHz** (thử 200) — làm CUỐI, cô lập.
6. **STOP** — graph/FP32/decode (~0.25s) giữ CPU.

### 3.3 Quyết định đã chốt (user)
- B-w = **12 SPEC** (~801 DSP = 64%, thoải mái sau prune R1).
- Clock = **166 MHz** mốc chắc, thử 200 MHz.
- **feature-flag, 2 build**.

---

## 4. THIẾT KẾ B-i — ACT-STATIONARY ON-CHIP

### 4.1 Nguyên lý
Nạp **toàn bộ ACT của layer** vào on-chip URAM **1 lần**, đọc lại cho mọi oc-block. Cắt ACT DMA từ `N_OCB×N_POS×N_ICB` xuống `N_POS×N_ICB` (**giảm tới 128×**). **Nhân bản y hệt cơ chế WT-stationary đã board-verified.**

### 4.2 So khớp WT-stationary (đã có) ↔ ACT-stationary (mới)
| | WT-stationary (CÓ) | ACT-stationary (B-i) |
|---|---|---|
| Store | `wt_uram u_bram_wt` | **`act_uram` (mới)** |
| Offset | `ls_wt_uram_off = ic*WT_BLK` (seq L82-83) | **`ls_act_uram_off = act_idx*256`** |
| Read | `{5'd0,wt_rd}+ls_wt_uram_off` (L229) | **`in_bram_rd_addr + ls_act_uram_off`** |
| Nạp 1× | LS_WTU loop load_ic (L120-128) | **LS_ACTL loop (pos,ic)** |

### 4.3 Phân tích fit — KHÔNG cần tiling
ACT packed = `N_POS × N_ICB × 256B`. Lớn nhất = **692 KB** (104² Cin=64: L03/04/05; và 52² Cin=256: L40/41). URAM trống = 2.1 MB (60/64). → dùng ~21 URAM, dư 3×. **Mọi conv1×1 fit → B-i độc lập, không cần B-ii tiling.**

### 4.4 Danh sách thay đổi (file:vùng)
1. **`act_uram.v` (MỚI)** — clone `wt_uram.v`, `DEPTH≈49152` (768 KB), `WAW=16`, `BAW=20`, RD_LAT=2, `ram_style="ultra"`.
2. **`layer_sequencer.v`**
   - Thêm state `LS_ACTL/LS_ACTLW` + counter `la_pos/la_ic`.
   - `LS_IDLE`: `ls <= feat_en0 ? LS_ACTL : LS_WTU`.
   - `LS_ACTL`: DMA tile (la_pos,la_ic) → act_uram (TGT_ACT), set `ls_act_uram_off` load-phase; loop hết (pos,ic) → LS_WTU.
   - Trong vòng oc: khi `feat_en0` **bỏ LS_ACT/LS_ACTW** (ACT đã on-chip) → LS_KICK thẳng.
   - Thêm output `ls_act_uram_off` (byte) = `act_idx*256` (act_idx đã tính L89), phân theo phase (load: la_*, kick: pos/ic).
3. **`unified_acc_top.v`**
   - Instantiate `act_uram u_bram_act`.
   - Đường đọc ACT (L1157, cổng rd8) → mux: `feat_en0 && !dw_mode ? act_uram(in_bram_rd_addr+ls_act_uram_off) : in_bram_wide`.
   - DMA-in wide (`dma_in_w128`) ghi act_uram tại `dma_in_w128_addr + ls_act_uram_off[MSB:4]` khi feat_en0 (mirror wt L228).
   - Thêm input port `ls_act_uram_off`; thêm reg `reg_feat_en` + output `reg_feat_en_out` từ axi_lite_slave.
4. **`axi_lite_slave.v`** — trong WR_EXEC `AXI_OFF_LAYER_MODE`: thêm `reg_feat_en <= wr_data_lat[4:1]` (cạnh `reg_layer_mode <= wr_data_lat[0]`). Thêm port `reg_feat_en`.
5. **`unified_cnn_top.v`** — wire `ls_act_uram_off` (layer_sequencer → unified_acc_top) + `reg_feat_en` (unified_acc_top → layer_sequencer), giống `ls_wt_uram_off_w`.
6. **`inference_unified_v4.py`** — ghi `REG_LAYER_MODE = 0x1 | (feat<<1)` (bit0 layer_mode, bit1 B-i). Layout ACT DMA **không đổi** (tile 256B).

### 4.5 An toàn / verify
- **DW không đổi**: gate `!dw_mode` (DW act phụ thuộc oc, không stationary).
- `feat_en0=0` → đường `in_bram_wide` cũ y nguyên → Build-1 baseline pass.
- `dma_engine` **KHÔNG sửa** (TGT_ACT tái dùng).
- Verify: bit-exact **156 det** + kick giảm mạnh (kỳ vọng cắt tới N_OCB× ACT DMA).

---

## 5. THIẾT KẾ B-w — 12 SPEC (WIDER ARRAY)

### 5.1 Nguyên lý + prune R1
Scale i_spec 4→12 (compute ~3×). **Prune `conv_kernel_r1`/`conv_tile_r1` (256 DSP dormant, an toàn vì conv3×3 vẫn CPU) → free 256 DSP.** DSP cuối ~801 (64%).

### 5.2 Kết quả deep-dive 3 vùng (2 điểm làm NHẸ hơn)
- **accu_top**: 🟢 conv1×1 **KHÔNG dùng** accu_top (đọc `psum_spec[flush_spec][g][p]` từng SPEC riêng, L664/678). accu_top chỉ 3×3/dw. → không bắt buộc mở 12-in cho conv1×1 (chỉ cần nếu muốn SUM >4 SPEC — lựa chọn thiết kế).
- **Weight distribution**: FSM nạp 256 reg `wt_rd_s{spec}g{grp}[0:15]` từ WT-URAM → `spec_wg{grp}[spec][nz]` → i_spec. B-w: **768 reg**, `spec_wg[0:11]`, WT blob/M-tile ×3, FSM load loop dài hơn.
- **Requant/writeback**: 🟢 `rq_m0/sh/bias[0:255]` đã 256 OC → 12 SPEC = 48 OC/kick « 256 → **không đổi mảng**. 🔴 **CHỈ 1 rr_unit, writeback nối tiếp** (4 SPEC=256 cyc; 12 SPEC=768 cyc) → phải **song song hóa rr** (1 rr/nhóm-4-SPEC = 3 rr) để hiện tốc độ.

### 5.3 Danh sách thay đổi
1. `unified_acc_top.v`: generate `g_spec si<12`; `ps[0:11]`; `psum_spec si2<12`; weight reg 256→768 (`wt_rd_s{0..11}g{0..3}`) + `spec_wg[0:11]`; FSM ST_WT_LOAD nạp 12-SPEC blob; `flush_spec` mở rộng (0..11, 4-bit); **song song 3 rr_unit** (flush theo nhóm).
2. `router.v`: 4→12 (sel rộng hơn) — nếu dùng router path.
3. Prune `conv_kernel_r1`: tie `r1_mode=0`/bỏ instantiate để Vivado loại → free 256 DSP.
4. Feed BW 12×: **bank** in_bram/wt_uram/act_uram để đọc song song 12 SPEC.
5. `inference_unified_v4.py`: pack WT blob 12-SPEC; điều chỉnh N_ICB/kick nếu đổi scheme reduction.
6. (Tùy chọn) param hóa `N_SPEC` (dùng define sẵn N_SPEC_ZCU=16) thay hard-code 4/12.

### 5.4 DSP budget (sau prune R1)
| | DSP | % 1248 |
|---|---|---|
| 12 SPEC MAC | 768 | |
| 3 rr song song | ~9 | |
| misc | ~24 | |
| **Tổng** | **~801** | **64%** |
| (16 SPEC nếu muốn) | ~1057 | 85% |

### 5.5 Verify
Bit-exact 156 det (bật B-i+B-w) + kick giảm ~3× so với B-i-only.

> ⚠️ **Quyết định prune-R1 xem lại ở §5b**: conv3×3 là DENSE → cần R1 làm engine conv3×3 → KHÔNG prune R1 (trừ Option C retrain). B-w có thể phải giảm 12→8 SPEC để chừa DSP cho R1.

---

## 5b. THIẾT KẾ B-c3 — conv3×3 LÊN HW

### 5b.1 Phát hiện quyết định: conv3×3 là DENSE (đo weights 2026-07-02)
NZ/16-block = **15.75–15.86/16** (0% block ≤3) cho L01/02/09/16/23/46/53. conv3×3 **KHÔNG sparse** ("N=3 toàn bộ" chỉ đúng conv1×1). → i_spec sparse fabric **KÉM** cho conv3×3 (n_cfg=16 dense, không có sparse gain). Engine dense đúng = **R1 systolic** (conv_tile_r1, 256-MAC dense).

### 5b.2 conv3×3 hiện tại
CPU 1.45 s (8 layer, dense BLAS — vốn hiệu quả). Sau B-i/B-w kick c1×1 giảm mạnh → **conv3×3 CPU thành bottleneck** (~48% forward ~3s).

### 5b.3 Đường HW đúng = R1 systolic (KHÔNG phải i_spec swc)
- **swc path (i_spec) tồn tại nhưng KÉM cho dense**: WT reload per-tile (L1067), ACT `SHUF_LOAD` per-tap (L1174, bỏ SHIFT), drain n_cfg=16 → chậm (khớp memory "conv3x3 bottleneck/ISPEC chậm").
- **R1 (conv_tile_r1)**: 256-MAC dense, tile-once, board-verified rev cũ, đang instantiate dormant trong unified_acc_top (`kr1_*`, `r1_mode=reg_pad[10]`). 2 kém-hiệu-quả của swc KHÔNG áp cho R1.

### 5b.4 TENSION với B-w (R1 = 256 DSP) — bảng lựa chọn (cập nhật)
| Option | conv1×1 | conv3×3 | DSP | Ghi chú |
|---|---|---|---|---|
| **A ✅ near-term** | **8 SPEC (2×)** | **R1 dense HW** | ~801 (64%) | an toàn, KHÔNG retrain; **ĐỔI B-w 12→8 SPEC** để chừa 256 DSP cho R1 |
| B aggressive | 12 SPEC (3×) | R1 dense HW | ~1057 (85%) | timing 166 MHz rủi ro cao (congestion) |
| C endgame paper | 12 SPEC (3×) | i_spec sparse | ~801 (64%) | **RETRAIN** conv3×3 → N:16 + risk mAP; prune R1 |
| D | 8-12 SPEC | CPU 1.45s | | conv3×3 = bottleneck |

→ ✅ **USER CHỐT: Option C** (12 SPEC + prune R1 + retrain conv3×3 sparse) — paper-faithful đầy đủ, mọi layer sparse trên 1 mảng chung.

### 5b.4b Hệ quả Option C (đã chốt)
- **Thêm ML track:** prune conv3×3 → N:16 sparse (như conv1×1 N:3 đã làm), QAT/retrain, **verify mAP giữ ~0.7976** → **model MỚI + golden MỚI**. Đây là **prerequisite của B-c3 verify** (không cản B-i/B-w). Risk: mAP sụt.
- conv3×3 chạy **i_spec swc với n_cfg nhỏ** (sparse) → fabric hiệu quả (drain N thay vì 16).
- **Prune R1** (free 256 DSP) → 12 SPEC vừa (~801, 64%).
- **swc rework bắt buộc** (dù sparse): WT-stationary conv3×3 (nạp 9-tap 1×/oc-block) + ride act_uram (B-i) + bật depth_shuffle SHIFT (bỏ SHUF_LOAD per-tap) — để bỏ 2 kém-hiệu-quả.

### 5b.5 Bring-up conv3×3-HW qua i_spec swc sparse (Option C)
1. **ML: prune conv3×3 → N:16 sparse** (dùng pipeline đã prune conv1×1 N:3), retrain QAT, verify mAP ~0.7976. → model + golden mới. (Song song RTL.)
2. **Pack WT 9-tap COO** cho conv3×3 (idx+val per tap, per M-block) — mở rộng make_swc_weights.py.
3. **WT-stationary conv3×3**: nạp 9-tap weights 1×/oc-block vào wt_uram (bỏ reload per-tile L1067).
4. **ACT-stationary**: swc đọc từ `act_uram` (B-i) — ride, có halo 3×3.
5. **Bật depth_shuffle SHIFT** (bỏ SHUF_LOAD per-tap L1174) — tái dùng cửa sổ giữa các tap.
6. **layer_sequencer** thêm conv3×3 mode (hiện conv1×1+DW): sweep output-tile, 9-tap, halo.
7. Gate bit REG_FEAT_EN mới (B-c3).
8. **Nuance**: L00 (Cin=3 RGB) không sparse-được (chỉ 3 kênh) → giữ CPU (178ms); L01/02 + deep chạy HW.

### 5b.6 Verify
Bit-exact vs golden MỚI (model sparse) + forward conv3×3 giảm (~1.45s → mục tiêu ~0.4-0.6s). ⚠️ mAP verify trên tập val (không chỉ det count).

---

## 6. CLOCK 166/200 MHz

### 6.1 Blocker (đã xác định)
- WNS hiện **−0.046 ns** ở 125 MHz (đang fail nhẹ). 200 MHz (5 ns) cần cắt ~3 ns.
- Đường xấu (tcl comment): **route URAM→DSP 6.1 ns** + smartconnect clock-skew.

### 6.2 Kế hoạch
1. Chèn input-reg cắt route URAM→DSP (bù 1 cycle latency, re-tune drain như conv1x1 drain=4).
2. HPM 32-bit (đã có), direct DMA (rs_dma đã gỡ).
3. `phys_opt` + `post_route_phys_opt` (đã bật).
4. B-w mảng to → congestion tăng → nếu WNS âm nặng: hạ 166 MHz hoặc giảm SPEC.
5. Mục tiêu **166 MHz** (1.33×) chắc; thử 200 nếu dư.

---

## 7. KẾ HOẠCH BUILD (2 build)

### Build 1 @ 125 MHz — verify cô lập từng cờ
| Bước | REG_FEAT_EN | Verify |
|---|---|---|
| 1.0 | 0 (OFF) | baseline 156 det bit-exact |
| 1.1 | B-i | 156 det + kick giảm (cắt ACT re-DMA) |
| 1.2 | B-i + B-w (12 SPEC, prune R1) | 156 det + kick c1×1 giảm ~3× |
| 1.3 | B-i + B-w + B-c3 (conv3×3 sparse HW) | verify golden MỚI (model sparse) + mAP; conv3×3 chuyển HW (~1.45s→~0.5s). Cần model retrain xong. |

Mỗi bước: **warm trước golden** (bài học warm-up false-neg), so bit-exact.

### Build 2 @ 166/200 MHz
Chỉ đổi `PL_FREQ_MHZ` + input-reg pipeline. Verify timing + correctness ở tốc độ + đo FPS cuối.

---

## 8. RISK REGISTER

| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| 200 MHz (WNS đã âm) | 🔴 Cao | Mốc 166 MHz; pipeline URAM→DSP; làm cuối cô lập |
| rr writeback nghẽn (768 cyc @12 SPEC) | 🟠 TB | Song song 3 rr_unit |
| B-w × timing (array to → congestion) | 🟠 TB | 12 (không 16) SPEC; phys_opt |
| Refactor 4→12 SPEC vỡ correctness | 🟠 TB | Gate cờ, verify cô lập 125 MHz |
| **Retrain conv3×3 sparse sụt mAP** | 🔴 Cao (Option C) | Prune N:16 (lỏng hơn conv1×1 N:3); verify mAP val; fallback giữ 1 phần dense/CPU nếu sụt |
| B-c3 swc rework (WT-stat+SHIFT) vỡ | 🟠 TB | Gate cờ; verify bit-exact golden mới |
| Prune R1 → 12 SPEC | 🟢 OK | Đúng Option C; free 256 DSP; conv3×3 dùng i_spec swc sparse |
| act_uram fit | 🟢 Thấp | 692 KB « 2.1 MB, không cần tiling |

---

## 9. PAPER MAPPING (I-SPEC, Micromachines 2023)

| Workstream | Mục bài báo | Cơ chế |
|---|---|---|
| B-i ACT-stationary | Sec 4.3 channel-first [22] | activation on-chip, bỏ re-fetch/deint host |
| B-w 8/12 SPEC | Sec 4.1 + kết quả ISA | scale SPEC (ISA dùng 1024 DSP) |
| **B-c3 conv3×3 HW (R1)** | conv dense datapath | 256-MAC systolic dense, WT/ACT-stationary |
| rr song song | RR Fig 8 | requant+ReLU song song |
| 166/200 MHz | kết quả ISA @200 MHz | — |
| (sau) B-ii tiling | Fig 10 | oc/h/w loop cho layer > on-chip |
| (endgame) conv3×3 sparse N:16 | Sec 4.2 COO | RETRAIN → i_spec dùng chung, sparse gain |

---

## 10. FILE ĐỘNG TỚI (tổng hợp)

| File | B-i | B-w | B-c3 | Clock |
|---|---|---|---|---|
| `act_uram.v` (mới) | ✓ | | (R1 ride) | |
| `layer_sequencer.v` | ✓ | | ✓ (c3×3 mode) | |
| `unified_acc_top.v` | ✓ | ✓ | ✓ (R1 WT-stat, un-dormant) | ✓ (input-reg) |
| `axi_lite_slave.v` | ✓ (REG_FEAT_EN) | | ✓ (bit B-c3) | |
| `unified_cnn_top.v` | ✓ | | ✓ | |
| `router.v` | | ✓ | | |
| `cnn_defines.vh` | | ✓ (N_SPEC) | | |
| `conv_kernel_r1.v`/`conv_tile_r1.v` | | GIỮ (không prune) | ✓ (bring-up) | |
| `rebuild_bitstream.tcl` | | | | ✓ (PL_FREQ_MHZ) |
| `inference_unified_v4.py` | ✓ | ✓ | ✓ (pack 9-tap WT, route c3×3) | |
| **KHÔNG đụng** | `dma_engine.v`, `cnn_acc_top.v`, i_spec leaf (depth_shuffle/weight_decoder/coo/bitmap/pch_gen), sliding_win_ctrl, tile_ctrl | | | |

---

## 11. THỨ TỰ CODE ĐỀ XUẤT (B-i)
1. `act_uram.v` — tự chứa (clone wt_uram, enlarge). Sim-check được.
2. `layer_sequencer.v` — LS_ACTL + ls_act_uram_off. Sim-check FSM.
3. `unified_acc_top.v` — instantiate act_uram + mux đọc ACT + reg_feat_en.
4. `axi_lite_slave.v` — REG_FEAT_EN (0x190 bit[4:1]).
5. `unified_cnn_top.v` — wire ls_act_uram_off + reg_feat_en.
6. `inference_unified_v4.py` — set bit.
→ Build 1 (125 MHz), verify 1.0 → 1.1.

---
*Memory liên quan: project-rtl-module-map, project-bi-act-stationary, project-levelb-exec-flow, project-ispec-levelb-arch, project-netseq-axi-addr-wall.*

---

## 9. ADDENDUM Rev139 (2026-07-02) — 3-trong-1 THỰC TẾ ĐÃ CODE (thay thế kế hoạch §5b/§6/§7)

**Bối cảnh đảo thiết kế:** (a) gate prune conv3×3 NO-GO (mAP sụp 39pp) → Option C sparse-c3 CHẾT;
(b) dò weights: **7/8 conv3×3 là stride-2** → R1 systolic (sweep stride-1 cứng) vô dụng → **prune R1**.

### 9.1 B-c3 = 9-pass conv1×1 (đường thay thế §5b)
conv3×3 dense = Σ 9 conv1×1 tap-shifted, ride engine 12-SPEC đã verify:
- ACT = raw CHW plane resident trong act_uram (1.5MB, chứa L01/L02 1.32MB). L00 (Cin=3) giữ CPU.
- Tap-loop ở **layer_sequencer**: vòng ic = 9×NICC kick (tap-major), keep_psum/no_flush cơ chế cũ.
- LS cấp per-kick: pixel-base (rowpix+colpix+tapoff — thuần adder) + pad masks (zero/left/right+last_pos).
- Engine: addr = base + idx*HW (accumulator, idx==flush_k dense-ascending) + pos*stride; mask tại capture.
- WT wide-load mới (ST_WT_LOAD_W): entry 64B-aligned → 16B/cycle, routing cố định 4 phase.
- Blob: make_c3_weights.py — chunk (ob,tap,icc) = 1360B; wt_uram 256KB (max 191KB/ob).

### 9.2 Fastkick (feat bit3) — win bất ngờ cho CẢ 1×1
RQ (336B @48oc) + writeback (768 cyc) chạy mọi kick dù middle-kick bị đè → skip RQ khi keep_psum,
skip WB khi no_flush. Với n_cfg=3 sparse, RQ là thành phần LỚN NHẤT của kick 1×1 → dự kiến ~2×.

### 9.3 Clock 166MHz + RD_LAT 2→3
+1 output-reg ở act_uram/wt_uram/in_bram (cắt route URAM→DSP 6.1ns). Retune +1 TOÀN BỘ capture
offsets (WT_LOAD/ACT_LOAD/LOAD_ACT_1X1/LOAD_RQ). KHÔNG flag được — baseline verify bắt regression.
Fallback ladder: 166.67 → 150 → 125 (sửa FREQ trong set_166mhz.tcl).

### 9.4 Feature-flag map (REG_LAYER_MODE 0x190 data bits)
bit0=lseq, bit1=B-i, bit2=B-c3, bit3=fastkick. Env host: BI=1 / C3=1 / FK=1.
Reg mới: 0x170=C3_CFG, 0x180=ACT_TOTAL (⚠️ BI=1 giờ BẮT BUỘC ghi 0x180 = npos*nicb).

### 9.5 Verify ladder (board): ns12 baseline → +BI → +FK → +C3 --verify → FPS full
⚠️ Bit NS=12 CHỈ chạy weights_unified_v4_ns12 (ESZ=49). Dự kiến ~0.6 FPS @166 (frame ~1.6s).

---

## 10. ĐÒN 4 — THIẾT KẾ CHI TIẾT (rev 2, 2026-07-03 sáng, sau khi soi RTL từng dòng)

**Nguồn:** paper Micromachines 2024,15,1164 (dense 4-bit, ZCU102, 79.1 FPS @512², 49% MAC eff)
+ đo đạc thật Rev139c trên board + dò `unified_acc_top.v` từng dòng.

### 10.0 Cycle đi đâu? (nền tảng của mọi quyết định dưới đây)

Rev139c @142.86MHz, frame warm 2.00s, phân rã THẬT:

| Thành phần | s | Bản chất |
|---|---|---|
| Kick HW c1×1 | 0.33 | busy thật của engine |
| Kick HW DW | 0.28 | busy thật |
| Kick HW c3 (9-pass) | 0.66 | busy thật (139d wide-ACT → ~0.35) |
| Python quanh kick (setup reg + poll ×47 layer) | 0.32 | **host glue → Đòn 4b xóa** |
| CPU L00 (Cin=3, 416²) | 0.18 | **→ 10.4 đưa lên HW** |
| FP32 detect-head + decode | 0.10 | giữ CPU (quyết định cũ) |
| Graph glue (resid/cat/route) | 0.09 | Đòn 4b xóa phần lớn |

**Phát hiện quyết định (dò RTL sáng 2026-07-03):** bên TRONG kick c1×1, mỗi NZ entry
(COO idx) nạp 16 pixel ACT bằng **20 cycle byte-serial** (`ST_LOAD_ACT_1X1`,
unified_acc_top.v L1153-1170: act_ld_cnt 0→19, đọc 1 byte/cycle qua rd8). Nhưng 16 byte
đó **thẳng hàng đúng 1 word 128-bit** của act_uram (addr = {icc, idx, act_ld_cnt[3:0]},
16B-aligned) — port word `rd_wsel` (Rev139d) đọc được trong 1 lần (RD_LAT=2).
Với n_cfg=3: feed 63 cyc/M-block trong khi MAC chỉ cần 3 cyc → **MAC eff ~5% do feed
serialization, không phải do engine**. `wt_rd_idx[0:15]` là mảng CHUNG cho cả 12 SPEC
(L332) → 1 word read phục vụ toàn mảng. DW cùng bệnh (feed byte-serial).

⇒ Hai lớp overhead tách bạch: **(1) trong-kick feed 20×** (fix = Đòn 4a, thuần RTL,
tái dùng port word 139d), **(2) host glue giữa layer 0.4s** (fix = Đòn 4b scheduler).
Bản nháp cũ chỉ nhìn thấy lớp (2).

### 10.1 Hai đường (cập nhật trần FPS trung thực)

| | Đường A: I-SPEC + dataflow 2024 | Đường B: full paper 2024 (dense 4-bit) |
|---|---|---|
| Engine | Giữ i_spec 12-SPEC INT8 (đã verify) | PE array mới: DSP packing 4×4-bit + Parity Adder Tree, feed II=1 |
| Retrain | KHÔNG | CÓ: QAT 4-bit (M4bQ+SCQ+OPReLU; paper mAP −1.5%) |
| Trần FPS @416 | **~2-3** (xem waterfall 10.5 — bản nháp cũ ghi 5-8 là lạc quan quá) | **~10-20** (cần eff ≥30% + 200MHz, xem 10.6) |
| Khối lượng | 4a ~2-4 ngày; 4b ~1 tuần | Lớn: engine mới + QAT + RQ 4-bit + golden mới |
| Rủi ro | Thấp (4a) / Trung (4b) | Trung-cao (mAP + bring-up) |
| Vai trò | Bước đệm; **toàn bộ 4a+4b tái dùng cho B** | Đích 10 FPS |

### 10.2 Đòn 4a — WIDE-ACT FEED cho conv1×1 + DW (đòn bẩy đơn lẻ lớn nhất, không retrain)

Nhân rộng đúng trick wide-ACT c3 của Rev139d sang 2 đường feed còn lại:
1. **conv1×1** (`ST_LOAD_ACT_1X1`): thay vòng 20-cycle byte-read bằng 1 word-read qua
   `rd_wsel/rd_waddr` (word addr = (ls_act_uram_off + {icc, idx, 4'b0}) >> 4). Barrel
   không cần (16B thẳng lane): `act_g0..g3[p] <= rd_wdata[p*8 +: 8]`. FSM: 3-4 cycle/entry
   (issue + RD_LAT2 + capture) thay 20 → **kick c1×1 0.33 → ~0.10s**.
2. **DW**: cùng xử lý cho đường nạp window DW (đang byte-serial qua act_uram sau 139d B-i)
   → **kick DW 0.28 → ~0.10-0.12s**.
3. Gate: `REG_LAYER_MODE` data **bit5 = W11** (wide-1×1), **bit6 = WDW** (wide-DW), env
   `W11=1 WDW=1` — giữ nguyên kỷ luật feature-flag + A/B trên board.
4. Điều kiện tiên quyết: ✅ đã có sẵn trong 139d — port word time-mux act_uram, mẫu FSM
   capture (aw_cnt), RD_LAT=3 retune đã xong. KHÔNG cần reg AXI mới, KHÔNG đổi blob.
5. Verify: ladder quen — baseline 154 det → +W11 bit-exact → +WDW bit-exact → FPS.

### 10.3 Đòn 4b — TASK SCHEDULER on-PL + ping-pong (cốt lõi paper 2024)

**Hồi sinh `network_seq.v` (Rev132 descriptor-mode ĐÃ board-PASS)** — không viết mới.
Lý do NO-GO cũ (Stage-1 net-negative −21%) KHÔNG còn áp dụng: hồi đó giữa 2 layer host
phải deint/repack (sequencer phải mô phỏng lại = overhead); nay sau B-i/B-w/B-c3 mọi
layout ACT/WT/OUT đã HW-native end-to-end, giữa layer không còn bước host nào — savings
= nguyên 0.32s glue + poll, overhead sequencer +0.27ms/layer × 47 ≈ 13ms. **Thắng chắc.**

1. **Descriptor v2 = 128B/layer** trong DDR (TGT_DESC=3 đã proven): toàn bộ reg per-layer
   hiện Python ghi (dims, base ACT/WT/OUT, C3_CFG, ACT_TOTAL, RQ base, REG_STRIDE, flags,
   layer_mode bits) + 2 trường mới: `src_sel` (ACT từ DDR hay đã resident act_uram) +
   `post_op` (none / eltwise-add / spill-only).
2. **Reg-mux**: khi netseq active, giá trị reg lấy từ descriptor thay AXI (mẫu Rev132).
   AXI giữ config toàn cục + kick 1 lần/segment + poll done 1 lần/segment.
3. **Chuỗi cắt SEGMENT tại điểm glue CPU**: SPPF maxpool ×1, upsample ×2, vài concat
   đầu-neck → **5-6 segment/frame** = 5-6 kick Python thay ~47 → glue 0.41s → ~0.02s.
   Eltwise-add resid trong C3 block: post_op add on-PL (adder 16B/beat trên đường
   writeback — RTL nhỏ); concat = thuần addressing (chọn OUT base kề nhau trong DDR, 0 RTL).
4. **Output dual-write**: writeback vừa ghi act_uram (vùng ACT layer k+1, khi fit) vừa
   DMA ra DDR (spill cho branch/golden — write WC 1200MB/s, ~20ms/frame cho toàn mạng, rẻ).
   **Ràng buộc URAM thật:** act_uram 1.5MB; L01/L02 plane 1.32MB + output 0.66MB > 1.5MB
   → các layer sớm (L01-L05) vẫn round-trip DDR (scheduler tự làm, không host); từ L06↓
   (≤52²) chuỗi on-chip trọn. KHÔNG cần CG-tiling ở giai đoạn này (khác paper — họ buffer
   nhỏ hơn nên phải tile; ta URAM to nên tile theo LAYER là đủ).
5. **WT prefetch ping-pong**: wt_uram 256KB chia đôi; trong lúc engine compute layer k
   (DMA idle sau B-i), netseq nạp WT layer k+1 vào nửa kia — che trọn WT DMA.
6. Verify ladder: netseq 2-layer bit-exact (mẫu test_netseq Rev132, warm trước golden!)
   → 1 C3-block segment → full backbone segment → 5-6 segment full frame 154 det → FPS.

### 10.4 L00 lên HW (xóa nốt 0.18s CPU)

L00 = conv3×3 s2 Cin=3 @416² — chạy được NGAY trên đường c3 9-pass hiện có: plane
416²×3 = 519KB fit act_uram, NICC=1 (pad 3→16 kênh, 13 kênh zero = n_cfg vẫn 16 dense
taps nhưng chỉ 1 icc), masks/stride2 đã có (7/8 layer c3 là s2). Việc phải làm: packer
thêm L00 vào blob c3 + host bỏ nhánh CPU L00. Không RTL mới. (Lý do cũ giữ CPU là
"Cin=3 không sparse" — không còn liên quan vì c3 9-pass là dense sẵn.)

### 10.5 WATERFALL FPS — DỰ ĐOÁN vs THỰC ĐO (cập nhật 2026-07-04, TRUNG THỰC)

**DỰ ĐOÁN cũ (đã sai 2 chỗ, giữ để đối chiếu):**
| Mốc | Frame dự | FPS dự |
|---|---|---|
| Rev139c | 2.00 | 0.50 |
| + 4a wide-feed (W11/WDW/RR4/RQD/OW/FR4) | ~1.21 | ~0.85 |
| + 4b scheduler | ~0.86 | ~1.2 |
| + L00 HW | ~0.73 | ~1.4 |
| + 166MHz | ~0.64 | ~1.6 |

**THỰC ĐO (board, model DominoSearch 0.8383):**
| Mốc | Frame nóng | FPS | Ghi chú |
|---|---|---|---|
| Rev139c | 2.00 | 0.50 | |
| Rev143 (4a đủ: W11+RR4+RQD+OW+WDW+FR4) | 1.47 | **0.68** | ✅ khớp gần dự đoán 4a |
| + 4b NS scheduler | 1.47 | **0.68** | ❌ **NET-NEUTRAL** — dự đoán SAI |
| + L00 HW (Rev147, C3V2) | 1.37 | **0.73** | ✅ gain thật +0.05 |
| + 166MHz | (đang build) | ~0.80? | nếu timing close |

**2 SAI LẦM của dự đoán:**
1. **"Glue 0.41s" SAI** — glue THỰC đo chỉ **0.09s** ("Giữa các conv"). §10.2/§10.3 tưởng scheduler
   cắt 0.41→0.03 = +0.35s. Thực: conv1×1 xen kẽ c3/DW/concat → segment ngắn 1-3 layer. NS cắt
   conv1×1-kick 0.34→0.15 NHƯNG chuyển NGUYÊN sang seg_flush + build-descriptor → **net-zero**.
   LẶP LẠI đúng bài học Stage-1 sequencer (net-negative) — [[project-c3block-sequencer-design]].
2. **HW-busy floor cao hơn tưởng**: kick c1×1 0.18 + DW 0.20 + c3 0.41 = **0.79s SÀN** không đổi
   bởi scheduler. Scheduler chỉ xáo host-overhead, không chạm HW-busy.

⇒ **Đường A THỰC TẾ trần ~0.8-0.9 FPS** (0.73 + 166MHz), KHÔNG phải 2-2.5. NS scheduler = ngõ cụt
cho FPS (code giữ, tắt mặc định). Muốn cao hơn: (a) c3 engine sâu hơn (kick 0.41 = lớn nhất),
(b) DW pack 0.09 host → on-chip, (c) **Đường B 4-bit** (đích thật cho ≥2 FPS — packing 4mul/DSP
hạ HW-busy floor ~4×, là đòn bẩy DUY NHẤT chạm HW-busy). Glue/scheduler KHÔNG phải đòn bẩy.

### 10.6 Đường B (Giai đoạn 3) — điều kiện đủ cho 10 FPS

Toán: sau prune còn ~768 DSP conv × 4 mul (packing 4-bit) × 2 op × 200MHz = **1.2 TOPS
peak**; cần eff ~30% → 370 GOPS → conv 5.2 GOP = **14ms/frame** + head 0.10s → **~8-10 FPS**
(@320² thì 15-20). Điều kiện đủ:
1. QAT 4-bit (M4bQ + SCQ + OPReLU per paper; GPU sẵn) — mAP đích ≥0.76 (paper −1.5%).
2. PE packing DSP48E2 (Fig 8: 2×4-bit chung port A/D + Correction Number precomputed).
3. **Feed II=1** — đây mới là "49% eff" của paper, không phải packing: Local Buffer
   BRAM (630KB trống) 2 port cấp PE array mỗi cycle, tái dùng weight khi trượt x.
   Kiến trúc 4a/4b (word-feed + scheduler + ping-pong) là bản INT8 của đúng cơ chế này
   — **tái dùng nguyên, chỉ thay PE core + RQ 4-bit**.
4. Golden pipeline mới (quant 4-bit) + verify mAP val, không chỉ det count.

### 10.7 Thứ tự thực thi đề xuất

1. **Rev139d verify + FPS** (đang build @150MHz) — nền cho mọi thứ sau.
2. **Đòn 4a** (W11+WDW, ~2-4 ngày, 1 build): mốc ~0.85-0.9 FPS. Rủi ro thấp nhất/gain lớn nhất.
3. **Đòn 4b + L00** (~1 tuần, 1-2 build): mốc ~1.4 FPS. Trong lúc đó ML track khởi động QAT 4-bit.
4. **Build 166MHz**: ~1.6 FPS. Quyết định GO/NO-GO Đường B dựa trên mAP QAT.
5. **Đường B** (~2 tuần): PE 4-bit + RQ 4-bit trên dataflow sẵn → 8-10+ FPS @200MHz.

Ràng buộc board (đã phát hiện, không quên): PL0 = 1000MHz/DIV (100/125/142.86/166.67/200);
bit phải TIME ở tần số tương ứng; `set_pl0.sh` volatile — chạy lại sau mỗi reboot; warm
kick trước mọi golden; golden hiện hành = **154 det** (leaky-trước-clip).

### 10.8 ĐÒN 4B — SPEC THI CÔNG (chốt 2026-07-03 chiều, bắt đầu code)

**Nguyên tắc integration: netseq_v2 = "host on PL" — GHI VÀO REG-FILE, không mux từng reg.**
axi_lite_slave thêm cổng ghi thứ 2 (`ns_wr_en/ns_wr_addr[8:0]/ns_wr_data[31:0]`) đi chung
decode WR_EXEC → mọi consumer (LS, engine, unified) không đổi MỘT DÒNG. Race-free: netseq chỉ
ghi khi host không ghi (host kick 1 lần/segment rồi poll).

**Descriptor v2 = 128B/layer (8 beat×16B; dma_engine desc buffer 4→8 beat, addr [1:0]→[2:0]):**
w0..w23 = giá trị 32-bit GHI THẲNG vào reg offset tương ứng (bảng offset cố định trong netseq
ROM nhỏ: CIN,COUT,HIN,WIN,NCFG,POC,ACCU,RELU,STRIDE,PAD,LTYPE,ACT_CHUNK,WT_BLK,WT_BYTES,
OUT_BYTES,NOCB,NPOS,NICB,ACT_TOTAL,C3_CFG,LAYER_MODE,IN/WT/OUT_ADDR lo+hi). w30=flags
(bit0=last-of-segment, bit1=eltwise-add postop [phase 2]), w31=magic 0x4B44 (guard).

**FSM netseq_v2:** IDLE → FETCH_DESC (TGT_DESC 8 beat) → CHECK magic → WRITE_REGS (24 nhịp,
1 reg/cyc qua ns_wr) → KICK_LS (pulse ap_start_ns, mux OR với ap_start axi) → WAIT ls_done →
(last? DONE_LATCH : next layer, desc_addr += 128). Host: ghi desc table 1 lần/frame (~47×128B
= 6KB), 5-6 kick/segment/frame (cắt tại SPPF/upsample/concat-glue như §10.3), poll done.

**Phase 1 (build kế): chuỗi conv thẳng** — mọi output vẫn DDR round-trip (scheduler tự làm,
không Python). KHÔNG dual-write act_uram, KHÔNG eltwise on-PL (giữ CPU tại điểm cắt segment).
Kỳ vọng: glue 0.41s → ~0.05s → FPS 0.68 → ~0.9-1.0. Phase 2: dual-write + eltwise + L00.
Gate: LAYER_TYPE=14 giữ nguyên convention (is_netseq) hoặc feat bit11 — chốt khi code.
