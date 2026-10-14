# Debug Log — yolov5-fpga

## 2026-07-12 — DỌN REPO + TÀI LIỆU BẢN ĐỒ DỰ ÁN (Docs/PROJECT_MAP.md)

- **Tạo `Docs/PROJECT_MAP.md`**: bản đồ toàn dự án cho người mới — từng folder/file: vai trò,
  cách chạy, input/output; flow 5 giai đoạn; lineage model đã trace bằng grep
  (quant_state_dict → qat_3fixes13 → extracted → sparse_lw_best → sparse_c3_best → extracted_c3).
- **ĐÃ XÓA (an toàn, không thuộc flow)**: tools/__pycache__, 2 file .bak host
  (bak_dwasync, bak_pipe), tools/nms_experiment.py (fix đã vào code),
  ip_repo/src/dma_engine.v.bak_widen, .Xil ×2, build173/174 backup jou+log,
  bd_cnn_acc/vivado_*.backup.*, dfx_runtime.txt ×3, runs/train run hỏng
  (qat_finetune* ×11, qat_3fixes cũ ×10 — GIỮ qat_3fixes13, pklot_lite_full).
- **XÓA ĐỢT 2 (user duyệt "ngoài flow xóa hết", thu hồi ~381M — runs 840M→459M)**:
  checkpoint era cũ (runs/sparse trừ dense_parent.pt; sparse_lw trừ sparse_lw_best.pt+idx),
  extracted_lw/_lw50/_newanchors/_sparse/_sparse_ep41, quant *_newanchors, train/pklot_qat_newanchors,
  3 script gate lịch sử (_res_sweep, gate_nm_finetune, eval_structured_prune).
  Sau dọn, lineage tối thiểu còn nguyên: dense_parent → sparse_lw_best → sparse_c3_best → extracted_c3.
- GIỮ nguyên: golden/, host_backup+src_backup_rev169b (không có git — bảo hiểm),
  zcu102_build/ (hướng đang chờ part), venv, datasets.

## 2026-07-11 — 🏁 CHỐT PRODUCTION (user): rev174 + lệnh chuẩn = 0.77s / 1.30 FPS / 106 det

User chốt "phiên bản 1.5FPS ban đầu, thử nghiệm vậy đủ rồi" → đóng giai đoạn thử nghiệm.
- **Bản chốt: `bd_wrapper_rev174_resident_200.bit`** + lệnh chuẩn How-fpga C.1 (mọi tính năng
  thử nghiệm gated TẮT → hành vi = bản 1.30 FPS baseline, nhưng **WNS +0.004 ĐÓNG 200MHz** —
  timing tốt nhất mọi bản; verify đủ bộ 2026-07-11: regression + --verify max=0 + Gate D).
- IC_INTERNAL (streaming-kick) DỪNG Ở THIẾT KẾ (đã chốt trong log 2026-07-11, chưa implement).
  WT ping-pong c3 không làm. Các mỏ FPS còn lại để ngỏ có thiết kế + số ước lượng.
- Dọn: xóa bit rev172/173 (trung gian; 173 chứa bug scatter) + build172/173 tcl/log/jou.
  GIỮ: rev174 (production) + rev170/rev169b (fallback) + build169b/170/174.tcl.
- Docs/How-fpga.md → rev174 toàn bộ.
- Hồ sơ thành quả cuối phiên thử nghiệm (đều board-verified): 1.30 FPS runtime ổn định;
  Dense A/B (sparsity ×3.95 kick, bit-exact 106 cả hai); netseq YIELD/resume; Gate D
  (resident chain + graphop on-chip bit-exact 0/173056); phase-breakdown kinh tế học
  resident; timing closure +0.004. Đủ nguyên liệu chương thiết kế + kiểm chứng luận văn.

## 2026-07-11 — MỎ FPS: NS-DDR ĐÓNG (đo); STREAMING-KICK (IC_INTERNAL) — THIẾT KẾ CHỐT

**(i) NS-DDR đo trên board**: NS=1 + best config = forward 0.79s (chậm hơn 0.77 baseline) + **228 det
(SAI — NS viết trước HEADQ, LazyOut không mang qscale)**. → ĐÓNG mỏ (i) bằng số đo; lịch sử không
đưa NS vào best config là có lý do. KHÔNG debug thêm (kể cả fix đúng cũng chậm hơn).

**(iii) STREAMING-KICK = IC_INTERNAL mode (feat bit17, 0x40000) — thiết kế sau khi đọc sâu engine:**
Khung nội bộ CÒN SỐNG: `n_ic_blks_1x1=ceil(reg_cin/16)`, `ic_blk_cnt` loop (unified_acc_top:1725
→ ST_WAIT_DMA), acc_clr/RQ chỉ ở blk0, LS giữ k_dma_in_done → ST_WAIT_DMA pass ngay. Mục tiêu:
1 kick/(oc,pos) thay vì N_ICB kicks → bỏ ~7cyc overhead/block × (N_ICB-1) → conv1x1 kick 0.150
→ ~0.11s (~+0.35 FPS lý thuyết một mình nó).
**3 RÀNG BUỘC NGẦM phát hiện (đọc RTL, chưa code):**
1. `l_nicb` dùng KÉP trong LS (đếm LS_WTU load chunks & đếm kick-loop) và `l_actc` dùng KÉP
   (ACTL DMA chunk ≤1024B & stride act per (pos,ic)). → Host giữ N_ICB=16/ACT_CHUNK=256 như cũ;
   LS thêm input `ic_internal`: kick-loop bound = ic_internal ? 1 : l_nicb (WTU giữ l_nicb).
2. **Blob format**: mode nội bộ nguyên thủy kỳ vọng entries mọi M-block LIỀN MẠCH + 1 RQ tail
   (wt_bram_rd_addr chỉ +1 xuyên block: dòng 1358). Blob ns12c = per-chunk [entries+RQ+pad
   WT_BLK]. → Engine fix: tại blk mới (ST_WAIT_DMA else-branch), wt addr base = ic_blk_cnt ×
   reg_wt_blk_bytes (chunk base trong wt_uram) + wt_bram_rd_addr RESET 0 (thay vì +1 tiếp).
   Cần đọc biểu thức wt_uram read addr để chèn base (đường ls_wt_uram_off=0 khi LS ic đóng băng).
3. RQ offset: `wt_nz_bytes_1x1 = n_ic_blks × ncfg_esz` (dòng 1184) → ic_internal phải dùng
   1×ncfg_esz (RQ tail của CHUNK 0). Gate theo ic_internal.
4. ACT addr: cộng `ic_blk_cnt×256` (1x1 chunk cố định 256B) — GẤP vào register ls_act_off_eff
   (đã có, timing-safe: ic_blk_cnt đổi ở FLUSH→WAIT_DMA, act đọc sau ≥16cyc qua WT_LOAD).
5. c3 LOẠI TRỪ (swc_en → tap-loop ở LS, không nội bộ hoá) — gate ic_internal & ~swc_en & bi_act.
6. feat_en 17→18 bit: readback 0x190 co padding 9'd0→8'd0.
Verify: elaborate + board A/B bit-exact per-layer (bit gated — regression an toàn kiểu rev170).
Ước sau (iii): 0.77 → ~0.72s. (ii) WT ping-pong c3 (shadow regs +6K FF) xếp SAU (iii).

## 2026-07-11 — PHASE BREAKDOWN chuỗi resident (block3 @26²) — KINH TẾ HỌC LỘ DIỆN

Đo tách segment trên rev174 (tổng ~121ms vs per-layer path ~35-45ms cùng khối việc):
| Pha | ms | Đối chiếu per-layer | Chẩn đoán |
|---|---|---|---|
| d0+d1 L17+L18 (scat) | 28.4 | 11ms | desc LM_BASE TẮT cờ tối ưu (W11/FK/RR4/RQD/OW) → conv chậm 2.5× |
| d2 L20 (resin→DDR) | 7.6 | 3.3ms | như trên |
| host yield (detile+DW21+pack) | 33.9 | ~0 nhìn thấy (DWASYNC ẩn) | **YIELD ĐỒNG BỘ HOÁ DW** — mất overlap |
| d3 L22 | 7.2 | 3.5ms | cờ |
| d4 GRAPHOP | 16.7 | resid_cat CPU ~4ms | **graphop byte-serial ~19cyc/byte = chậm hơn CPU 4×** |
| d5 L19 | 27.7 | 7.1ms | cờ |

**3 kết luận:**
1. Cờ tối ưu bật lại trong desc = fix dễ (~−40ms).
2. graphop v1 (tuần-tự-triệt-để, đúng-trước) chậm hơn CPU numpy ~4× tại op — muốn dùng cho FPS
   cần **graphop v3 pipelined ~1B/cyc** (RTL vừa: overlap read/rr/pack; resid 2cyc/B) → ~1.5ms/block.
3. **DW-CPU là nút thắt KIẾN TRÚC của chuỗi resident**: yield đồng bộ 34ms/block mà per-layer path
   ẩn hoàn toàn sau DWASYNC; dependency mdw chặn ẩn trong-block; graph YOLOv5 không cho xen block.
   → Sau MỌI fix nhanh (cờ + graphop v3): chuỗi NS block3 ≈ 52ms vs per-layer 40ms — **VẪN THUA**.
   DW on-PL in-chain cần window-formatter on-chip (RTL mới, to) — không đáng lúc này.

**⇒ Resident-chain KHÔNG phải mỏ FPS trên KV260 chừng nào DW còn trên CPU.** Gate D giữ giá trị
= proof-of-architecture (đủ luận văn). Mỏ FPS thật còn lại (từ scan 2026-07-10): NS-DDR gộp thủ tục
per-layer (+~0.07-0.1 FPS, rẻ); WT ping-pong c3 (19-30% c3 kick); streaming-kick (IDLE+overhead
7/19 conv1x1). Ước cộng dồn 1.3 → ~1.7-1.9 FPS KV260. Mốc 2.2+ cần phá thêm (ZU7EV/DW-PL).

## 2026-07-11 — 🎉 REV174 BOARD: GATE D PASS bit-exact + WNS +0.004 (timing ĐÓNG)

- **WNS = +0.004** (rev172 −0.157 → rev173 −0.266 → rev174 **+0.004**): register-hoá 3 tổng
  địa chỉ tĩnh-trong-tile trúng critical path. Bit: `bd_wrapper_rev174_resident_200.bit` (board).
- **Gate A regression**: 106 det / 0.78s — superset sạch (NS/resident off = hành vi rev170).
- **GATE D PASS bit-exact** (`test_ns_resident_block.py`): chuỗi block3 @26² 6 desc/1 yield —
  L17→scatA, L18→scatC, L20←A[RESIN]→DDR+YIELD, host DW21, L22→scatB, GRAPHOP(A,B,C→DST),
  L19←DST[RESIN]→out. mc1 mismatch=0, L19 cuối **mismatch=0/173056 max=0**. chain=112ms
  (busy-poll NS + host detile/pack quanh yield — chưa tối ưu; đây là gate ĐÚNG, chưa phải FPS).
- CHỨNG MINH TRÊN SILICON cùng lúc: multi-region residency + RESINBASE + multi-ob tiled scatter
  (L17 N_OCB=3 + ock-clamp) + graphop end-to-end resid_cat + YIELD/resume giữa chuỗi resident.
  = mọi viên gạch của host-minimum. Root-cause lag_en_out (hôm qua) ĐÚNG đích.
- CÒN LẠI = production orchestration: desc-builder cả mạng (block 2 @52² fit, block1 @104²
  breakpoint vì storage-wall, neck/SPPF) + gộp segment (giảm host round-trip quanh yield) →
  ĐO FPS thật (kỳ vọng 1.3→2.2-2.8). Chain 112ms/block hiện cao do overhead poll — sẽ giảm khi
  gộp desc + INTWAIT.

## 2026-07-11 — GATE D FAIL → DEBUG CÓ HỆ THỐNG → ROOT CAUSE + TIMING FIX (build174 chạy)

**Gate D (rev173) FAIL**: mc1 sai 80% max=29 ngay checkpoint đầu (trước graphop). Chuỗi debug:
1. Ladder: T1 (desc cơ bản) PASS, T2 (resident-in @0) PASS, T4 (scatter→A + RESINBASE) FAIL.
2. Hypothesis-sim (5 mô hình layout) → KHÔNG cái nào khớp → nghi mô hình sai tầng sâu hơn.
3. Probe "x6@0 còn nguyên sau scatter?" (L17-scat → L18 đọc lại @0): PASS ×2 dữ liệu khác nhau
   → scatter không phá vùng thấp.
4. **Dump vùng A bằng conv IDENTITY** (blob tự chế W=I, m0=2^15 sh=15, relu off, RESINBASE=A)
   → A sai ~96% đều, tile0 60%.
5. **Fit tham số layout từ dump**: khớp **98.7%** với `stride_pt=48, ob_off=48×43, KHÔNG reset`
   = hành vi Rev200-cũ (H10).
6. Đối chiếu RTL → **ROOT CAUSE: fix "lag=c3-only" Rev173 chỉ vá `lag_en_c` NỘI BỘ, SÓT
   `lag_en_out`** (cấp cho LS!). Hệ quả chuỗi: LS l_lag=1 → l_tiled=0 (scat_cpad rơi về 48)
   + DSCATW đi nhánh DRAINW (không có ob-reset) = đúng H10; drain-writes bị mux lag_en_c=0
   chặn nên x6 nguyên (khớp probe 3). MỌI triệu chứng giải thích trọn.
   → FIX: `lag_en_out &= reg_feat_en[1]` (feat[1]=C3 flag; c3 RESACT giữ nguyên).
   BÀI HỌC: một semantic đổi phải grep MỌI nơi tính cùng biểu thức (lag_en_c VÀ lag_en_out).

**TIMING (user yêu cầu đóng 200MHz — WNS rev172 −0.157, rev173 −0.266):**
- Path: ls_reg(state) → mux act_load_phase → +resin_off(21b) → +in_bram_rd_addr(21b) → mux
  graphop → URAM CAS_ADDR. Stack Rev200 chất 1 adder + 1 mux lên path vốn +0.019 (rev170).
- FIX (register hoá tổng TĨNH-TRONG-TILE, an toàn vì uoff pipe 2 tầng + kick đến sau ≥2 cyc
  → trễ 1 nhịp không bao giờ bị đọc): (a) `ls_act_off_eff` wire→REG; (b) 2 tổng base scatter
  `scat_tbase_r/scat_chw_r` REG (oc_deint latch base ở start, cách lần cập nhật cả 1 kick).
- Lỗi phụ đã né: always dùng scat_base_w trước khai báo → dời luôn-block xuống sau.

**Build 174 relaunch** (12-core sẵn trong tcl dòng 14): 4 fix Rev173 + lag_en_out root-cause
+ 3 register timing + glob ipshared 3-tầng (ipcache gotcha: `ip_user_files/bd/bd/bd/` không
được refresh — cây ea59 giữ bản cũ; đã xoá + sửa glob). Elab PASS. CHỜ WNS + Gate D lại.

## 2026-07-10 (đêm) — REV173: 3 BUG SCATTER BẮT TRÊN GIẤY khi thiết kế Gate D + build chạy

Thiết kế Gate D (block 3 resident + graphop in-chain) trước khi viết test → design-review bắt 3 gap
của Rev200 tiled scatter (kill build 2 lần để vá — rẻ hơn 3 vòng debug board):
1. **Multi-ob cpad+offset**: LS tính cpad từ OCK(48) & không có ob-offset → chỉ đúng Cout≤48; mọi
   lớp cv thật (64-256) sai layout. FIX: cpad THẬT từ REG_C3_CFG[9:0] (host cấp; c3_cfg rảnh với
   conv1x1, gate l_tiled) + `ob_base_s` (ob*OCK, wrap oc-block: tbase=ob_base mới).
2. **lag gating**: lag_en_c = bit12&~bit14 → conv1x1 RESOUT-không-INBANK rơi nhầm LAG mode.
   FIX: `& c3_act` (lag = c3-only đúng bản chất Rev169). c3 RESACT cũ giữ nguyên hành vi.
3. **ock clamp ob cuối**: N_OCB*48 > cpad (vd Cout=128, ob2 có 16 kênh pad) → scatter pad TRÀN
   word sang tile kế = đè rác lên ob trước. FIX: `deint_ock = min(OCK, cpad - ob_base)` khi l_tiled.
(+ mang theo fix dbase-resume từ chiều). Elaborate PASS ×4 (2 lần đầu dính lỗi khai-báo-trước-dùng
— xvlog strict; nhớ: reg phải khai TRƯỚC wire dùng nó). coreRev **173**, build173 đang chạy (lần 3).

**tools/test_ns_resident_block.py (GATE D) viết sẵn**: block 3 @26² = 6 desc/1 yield:
L17(ACTL x6@0 →scat A, N_OCB=3 test clamp!) → L18(x6 resident →scat C) → L20(RESINBASE A →DDR,
YIELD) → host DW21 → L22(ACTL mdw →scat B) → GRAPHOP(A,B→resid ‖ C →DST) → L19(RESINBASE DST
→DDR, LAST). Map act_uram: X=0/A=0x2B000/B=0x40800/C=0x56000/DST=0x6B800 (<1.5MB, đều /256).
So bit-exact với per-layer chuẩn. PASS = multi-region residency + graphop e2e CHỨNG MINH TRÊN SILICON.

## 2026-07-10 — REV172 BOARD-VERIFIED: regression + verify max=0 + NETSEQ YIELD/RESUME PASS

**Build172**: bitstream OK, WNS **−0.157** (path: u_ls/ls_reg → act_uram CAS_IN_ADDR — mux addr
thêm nhánh graphop/scat_tiled). Xử theo tiền lệ Rev165 (âm nhẹ → trọng tài = silicon):
- **Gate A**: regression NS-off = 106 det / 0.77s / 1.30 FPS — y hệt rev170 ✓
- **Gate A+**: `--verify` = **max=0 TOÀN MẠNG** (mọi lớp HW bit-exact CPU) → WNS âm vô hại,
  rev172 nghiệm thu. Bit: `bd_wrapper_rev172_nsyield_200.bit` (board ĐANG NẠP).
- NỢ TIMING (không chặn): restructure mux act_uram addr hoặc multicycle cho scatter-base
  (chỉ dùng lúc start pulse) — làm cùng build sau.

**Gate B/C — test_ns_yield.py: ALL PASS (bit-exact)**
- desc0 L26 qua netseq: mismatch=0. YIELD dừng đúng, host CPU-op thật (detile+repack), kick
  resume → desc1 L30 (LAST): mismatch=0 vs golden 2-kick rời.
- **BUG board-caught + fix 2 tầng**: desc conv GHI ĐÈ REG_IN_ADDR (w10=act addr) → resume latch
  desc_base từ reg bẩn → fetch desc từ vùng ACT → magic err → done rỗng (o30 toàn 0, không
  timeout). Fix host: ghi lại desc_base trước MỌI kick resume (chạy ngay, không cần build).
  Fix RTL (build sau): `dbase <= resume_pend ? dbase : desc_base` (đã vào source).
- Ý nghĩa: **cơ chế breakpoint/resume của kiến trúc host-minimum HOẠT ĐỘNG trên silicon** —
  nền tảng để rút host khỏi vòng lặp per-layer. Còn lại: desc table cả mạng (production
  orchestration: resident blocks + graphop-in-chain + DW yield) → đo FPS.

## 2026-07-10 — REV172 IMPLEMENT: netseq YIELD + LS graphop passthrough — elab PASS, BUILD RUNNING

**3 sửa RTL (thiết kế đã chốt lượt trước):**
1. `unified_acc_top.v`: `k_done_pulse_out = (state==ST_DONE) | (graphop_mode & graphop_done)`
   (graphop mode FSM chính đứng ST_IDLE → LS treo nếu không có; graphop_done = pulse 1-cyc engine)
   + port mới `lt_graphop_out = graphop_mode`.
2. `layer_sequencer.v`: input `lt_graphop` + reg `l_gop`. LS_IDLE nhánh mới (TRƯỚC nhánh thường):
   ap_start && layer_mode && lt_graphop → l_gop=1, thẳng LS_KICK (không latch config loop, không
   WTU/ACTL). LS_RUN: k_ap_done && l_gop → LS_DONE thẳng (bỏ vòng ic/OUT). FSM 16 state ĐÃ HẾT
   → tái dùng KICK/RUN với cờ l_gop (không thêm state). Các gán c3/k_keep trong LS_KICK vô hại
   (kernel chính idle trong graphop mode).
3. `network_seq_v2` (unified_cnn_top.v): **w30 bit1 = YIELD** — xong layer → resume_pend=1,
   resume_li=li+1, ns_done (→ap_done/IRQ, host làm CPU-op); ns_go kế: li = resume_pend?resume_li:0.
   LAST clear resume_pend. Backward-compat: desc cũ w30 bit1=0 → hành vi y nguyên Rev144.
- Warning elab `reg_conv_type 4b→2b config_decoder`: VÔ HẠI (LT=4 decode như 1x1 nhưng kernel chính
  idle trong graphop mode; rev170 board-verified với đúng warning này). KHÔNG sửa module proven.
- **Elaborate PASS** (xvlog+xelab unified_cnn_top). **build172 ĐANG CHẠY** (impl_1, ~2h).
- **Host test viết sẵn**: `tools/test_ns_yield.py` — gate board: golden 2-kick rời vs NS segment
  2-desc có YIELD giữa (L26 yield → host stage input L30 = "CPU-op thật" → kick resume → L30 LAST)
  → so bit-exact. Đây là pattern DW-breakpoint thật thu nhỏ.
- Bring-up plan sau build: (A) NS off regression 106 det → (B/C) test_ns_yield → (D) graphop-in-chain
  (desc LT=4 + producer RESOUT/SCATOUT + consumer RESINBASE) → (E) orchestration production.

## 2026-07-10 — KV260 KILL-HOST bắt đầu: BACKPORT Rev200 PASS ×2 gate + netseq_v2 scoped

**Backport zcu102→rev76_build (KV260)**: 5 RTL (axi_lite_slave, graphop_engine, layer_sequencer,
unified_acc_top, unified_cnn_top) + 2 sim. GIỮ act_uram/wt_uram bản KV260 (URAM "ultra" 1.5MB —
diff xác nhận zcu102 chỉ khác ram_style/DEPTH). coreRevision 171→**172**.
- **GATE 1 PASS**: tb_graphop (tiled in/out) trên act_uram 1.5MB KV260 → TB_GRAPHOP_ALL_PASS.
- **GATE 2 PASS**: xvlog+xelab unified_cnn_top (31 file). → KV260 giờ có TRỌN stack: graphop tiled,
  SCATOUT (bit15, scat_base=reg_out_addr[20:4] tái dùng), RESINBASE (bit16), LS tiled scatter.

**netseq_v2 scoped (đọc trọn — module trong unified_cnn_top.v:414, KHÔNG phải network_seq.v Rev132)**:
Rev144-147 đã hoàn chỉnh + 3 fix board-caught (edge-detect ns_go, data-arrival gate beat8, busy-seen
gate) — fetch desc 128B → ghi 27 reg (0x040-0x1E0) qua ns_wr → kick LS → chờ ls_done → LAST.
**Mở rộng cần cho paper-direct (thiết kế chốt):**
1. **w30 bit1 = YIELD** (breakpoint): xong layer → ns_done (ap_done→IRQ như cũ) + resume_pending
   giữ li+1; ns_go kế tiếp tục từ resume_li (không reset 0). LAST clear resume; !ns_mode = reset cứng.
2. **LS graphop passthrough**: input mới lt_graphop (=graphop_mode có sẵn); LS_IDLE thấy
   ap_start&&layer_mode&&lt_graphop → LS_KICK (pulse k_ap_start) → LS_RUN chờ k_ap_done → LS_DONE
   (skip WTU/ACTL/OUT). ĐƯỜNG KICK ĐÃ THÔNG: ap_start_eff = mode_layer ? ls_k_ap_start : ap_start_w
   (unified_acc_top:872) → graphop nhận kick từ LS tự nhiên (:1278). CẦN KIỂM khi implement:
   k_ap_done pulse về LS trong graphop mode (graphop_done → k_done).
3. Desc 27 word ĐÃ PHỦ đủ mọi reg graphop tái dùng + out_addr (=scat_base SCATOUT) → format GIỮ NGUYÊN.
4. Host: desc-builder python (layer_params + requant → bảng 128B/layer; M0/shift graphop =
   find_best_shift; thứ tự block1-fix L3→L4→L6→DW7→L8→graphop→L5 với DW=YIELD breakpoint).
- Housekeeping treo: network_seq.v (Rev132) nghi dead (chỉ dma_engine nhắc tên — kiểm comment hay code).
- CÒN LẠI: implement 1+2 (RTL nhỏ) → tb netseq (reg-trace + yield) → host builder → build 172 → board.

## 2026-07-10 — SCAN HEADROOM sâu (board, GPROF + PERLAYER + STPROF) — bản đồ nạc còn lại

**GPROF (graph-ops CPU, per frame):** resid_cat 29ms, cat 24ms, maxpool 16ms, upsample 4.5ms
(= 73.5ms của bucket between 102ms; còn ~30ms Python glue). NMS 29ms + decode_pre 14ms (1 lần/ảnh).
→ graphop engine phủ 53ms (resid_cat+cat); upsample CPU rẻ hơn tưởng (4.5ms — engine hạ ưu tiên,
breakpoint đủ); maxpool 16ms breakpoint OK.

**PERLAYER (frame nóng):** early layers ăn hết — c3: L01 53ms + L02 47ms + L09 34ms + L00 32ms
= 166/202ms c3_hw. conv1×1 block1 @104²: L05 25 + L04 20 + L03 17 + L06 14 + L08 14 ≈ 90/340ms.
Lớp 13² chỉ 2-5ms/lớp. → mọi tối ưu nên nhắm 104²/208²/52².

**STPROF c3 (phân bố FSM, 8 lớp):** LD_ACT_W **41-48%**, WT_LOAD_W **19-30%**, SHUF+KW ~13%,
WRITEBACK 3-10%, IDLE 3-14% (lớp nhỏ IDLE cao = per-kick overhead), COMPUTE ~0% (ẩn trong pipeline).
→ c3 kick ≈ **75% thuần nạp dữ liệu**: (a) WT ping-pong che được 19-30%; (b) c3 1-pass sliding-window
giết re-read ×9 của LD_ACT_W 45%; (c) IDLE lớp nhỏ = đất của streaming-kick.

**BẢN ĐỒ NẠC (số đo, frame 0.77s):** host 0.45 → ~0.10 (residency+graphop+net_seq, đã có RTL);
c3 HW 0.20 → ~0.05 (ping-pong WT + 1-pass); c1x1 HW 0.15 → ~0.10 (streaming-kick bớt 7cyc/19cyc
overhead). ⇒ trần KV260 sau tất cả: ~0.2-0.25s ≈ **4-5 FPS** (từ 1.29). Ưu tiên đổi: upsample_engine
HẠ (CPU 4.5ms); WT ping-pong NÂNG (STPROF lộ 19-30%, cao hơn ước cũ).

## 2026-07-10 — THÍ NGHIỆM DENSE-ENCODED A/B (đo cổ tức sparsity trên silicon) — SẴN SÀNG, CHỜ BOARD

Mục đích: pack CHÍNH model sparse hiện tại nhưng ép N=16 (dense encode, slot 0 giữ nguyên)
→ kết quả bit-exact 106 det, CHỈ timing đổi → cô lập đóng góp drain-N. Dự đoán: kick ×~3-3.7
(cost model 4N+7: 19→71 cyc/block), forward 0.77→~1.4-1.7s (host 0.42s không đổi = luận điểm
dataflow). ĐO Ở `PROF[kick]+c3_kick` (KHÔNG phải FPS — FPS bị host nén).
- **make_unified_weights.py**: thêm env `FORCE_N` (ép N_CFG mọi lớp, tắt NCFG_AUTO).
- **make_c3_weights.py**: KHÔNG cần sửa — bỏ env C3_IDXMAP là mọi lớp dense 16 (đường có sẵn).
- Packed `extracted_c3/weights_unified_dense16/` (66 file như ns12c; dw copy nguyên): conv1x1
  46 lớp WT_BLK=1120B (4.30MB), c3 8 lớp WT_BLK=1360B (4.24MB, **max per-ob 191KB < wt_uram
  256KB ✓**), tổng blob 9.1MB (sparse 4.1MB). N_OCB từng lớp == ns12c ✓ (chỉ N_CFG/WT_BLK đổi).
- Board user bật lại (reboot → reload đủ: rev170.bit + set_pl0 200 + dma_region). Tar timestamp
  warning = đồng hồ board lệch sau reboot, vô hại.
- ✅ **KẾT QUẢ A/B (2026-07-10, board KV260 rev170, frame nóng, cùng lệnh chuẩn + PROFDUMP):**

  | Metric | Sparse ns12c | Dense16 | Tỷ lệ |
  |---|---|---|---|
  | kick conv1×1 (46 lớp) | 0.150s | 0.593s | **×3.95** |
  | c3_kick (8 lớp) | 0.166s | 0.218s | ×1.31 |
  | HW kick tổng | 0.316s | 0.811s | **×2.57** |
  | forward | 0.77s | 1.20s | ×1.56 |
  | FPS | 1.29 | 0.84 | — |
  | **Detections** | **106** | **106** | **BIT-EXACT** ✓ |
  | Host buckets (pack/deint/between) | 0.06/0.10/0.10 | y hệt | KHÔNG đổi ✓ |

  3 KẾT LUẬN (thesis-grade):
  1. **Flexible drain-N verified trên silicon**: cùng weights, N=2..6 vs N=16 → 106 det giống hệt.
  2. **Cổ tức sparsity đo được**: conv1×1 ×3.95 — KHỚP cost model 4N+7 (N_avg≈3: 71/19=3.7;
     L26 N=2: 4.7). c3 chỉ ×1.31 → suy ngược N_avg_c3 ≈ 12 (c3 giữ N cao vì accuracy — đúng scheme).
  3. **Coupling chứng minh bằng đo**: compute ×2.57 nhưng end-to-end chỉ ×1.56 vì host 0.42s
     KHÔNG đổi → sparsity chỉ hiện đủ thành FPS khi hệ compute-bound → motivation dataflow/net_seq.

## 2026-07-10 — Dọn 20 file RTL chết khỏi rev76_build/ip_repo/src (KV260) — elab PASS

Phương pháp: script reachability BFS từ top THẬT `unified_cnn_top` (modelName trong component.xml)
→ 29 file sống / 22 không với tới. Xóa 20, GIỮ 2 tb của module sống:
- **XÓA (19 RTL chết)**: cnn_acc_top (top cũ 3-kernel: c1_copy, rebank), systolic path
  (conv1x1/conv3x3_systolic_top, systolic_seq, systolic_layer_seq, conv_kernel_r1, conv_tile_r1,
  pe_array, bram_ping_pong), spec-rời path (spec_conv1x1_top, spec_layer_seq, i_spec_mm,
  depth_shuffle_mm), dw cũ (dw_engine, dw_layer), conv3x3_layer_seq, dma_sequencer,
  deint_bram (Rev165 đã bỏ instantiation) + **tb_ispec_mm.v** (test i_spec_mm chết).
- **GIỮ**: tb_unified_acc_top.v (= sim modelName của IP trong component.xml!) + tb_i_spec_e2e.v
  (instantiate i_spec SỐNG). network_seq.v SỐNG (NS mode Rev144 trong unified_cnn_top).
- component.xml: gỡ 38 entries (19×2 filegroup), coreRevision 170→**171**.
- **GATE PASS**: xvlog bộ 31 file còn lại + xelab unified_cnn_top OK + 2 tb compile OK.
- Bitstream rev170 trên board KHÔNG ảnh hưởng (đã build). Lưu ý build sau: build*.tcl copy src
  vào ipshared nhưng KHÔNG xoá file cũ ở đó — component.xml hết list nên không được compile
  (vô hại); muốn sạch tuyệt đối thì xoá thư mục ipshared/*/src trước build 171.
- zcu102_build/ip_repo CHƯA dọn (user chỉ định KV260) — cùng danh sách chết nếu muốn đồng bộ.

## 2026-07-10 — Dọn 10 file Python legacy (user yêu cầu; verify import-graph trước khi xóa)

Xóa (không ai import — đã grep tools/+yolov5/+scripts/): `tools/inference_fpga.py`, `tools/fpga_ispec.py`,
`tools/fpga_conv1x1.py`, `tools/gen_test_golden.py`, `tools/gen_test_golden_mini.py` (cụm host 3-kernel/
I-SPEC rời cũ), `tools/gen_cnn_pkg.py` (sinh cnn_pkg.sv flow cũ — cnn_defines.vh hiện tại là file VIẾT TAY,
không liên quan), `tools/gen_nm_sparse.py` (heuristic, KHÔNG phải đường DominoSearch), `tools/
calibrate_detect_head.py` (thay bằng ptq_head.py; nơi khai sinh find_best_shift — hàm đã sống trong
extract_quant_data.py), `tools/debug_swc_pingpong.py` (one-off debug đã đóng), `yolov5/
extract_sparse_weights.py` (đường cũ — thay bằng create_sparse_extracted.py).
- **GIỮ `tools/fpga_conv3x3.py`** dù gốc legacy: bench_kick.py + test_netseq.py đang import (ref_conv3x3
  + c3seq helpers). GIỮ sanity_check/nms_experiment/draw_dets (bằng chứng thesis + hữu dụng).
- Tham chiếu còn lại chỉ trong comment/docstring (inspect_coo, compare_qat_single, ptq_sparse) — vô hại.
- tools/ còn 19 file, tất cả thuộc pipeline chính / verify / đo đạc (mô tả đầy đủ từng file: chat 2026-07-10).

## 2026-07-10 (tiếp) — PAPER-DIRECT chốt + chuẩn TILED khép kín (sim PASS + elab PASS)

User chốt: BÁM BÀI BÁO NGAY (host-minimum, không thang deploy). ARCH_PLAN.md VIẾT LẠI =
bản spec paper-direct: chuẩn TILED cho mọi plane resident, LAYER_TYPE 5/6 (pool5/upsample),
**net_seq descriptor walker** (desc_bram 64×128B, format 32-word, TUẦN TỰ NGHIÊM NGẶT +
BREAKPOINT yield-host — chống race-wall T2-4 by construction), host cuối = nạp ảnh + kick
+ đọc 3 head + NMS. SPPF pool5 v1 = breakpoint (byte-serial RTL 97ms > CPU — không đáng;
word-based 2.5ms để sau).
- **LS tiled scatter (Rev200)**: deint_tbase = pos*cpad tăng dần (+= cpad = (OCK+15)&~15 tại
  LS_DSCATW), port mới qua unified_cnn_top → unified_acc_top; oc_deint base mux 3 mode:
  lag / tiled (RESOUT & ~c3: base=tbase+scat_base_w, stride=1) / CHW (c3: pos,npos — Y CŨ).
  conv1x1+RESOUT chưa từng dùng trên KV260 → không đổi hành vi flow cũ.
- **graphop INPUT TILED (Rev200)**: addr đọc = base + (p[15:4]*cpad_src + ch)*16 + p[3:0],
  cpad_src RIÊNG A/B vs C. Golden gen xuất input tiled (pad kênh riêng từng nguồn: 20ch→1024B,
  12ch→512B), tb addr A=0/B=1024/C=2048/DST=2560 (chia hết 256).
  **Sim: `=== TB_GRAPHOP_ALL_PASS ===` (resid_cat + cat, 1024 byte bit-exact)** → chuẩn TILED
  khép kín: producer scatter tiled → graphop tiled → consumer conv1x1 tiled.
- Elaborate unified_cnn_top PASS (lần 3) với toàn bộ Rev200.
- CÒN LẠI (spec §6): upsample_engine (LT=6) → net_seq + desc_bram + virtual-reg mux + tb
  reg-trace → host desc-builder + inference_zcu102.py → build (chờ part) → board.

## 2026-07-10 — ZCU102 FULL FLOW khởi động: workspace + Rev200 RTL (elaborate PASS ×2)

User có board ZCU102 → triển khai FULL flow bài báo 2023 (4 nguồn tốc độ: SPEC✅ + channel-first
+ ping-pong + full-network). Kế hoạch chi tiết: `zcu102_build/ARCH_PLAN.md`. KV260 giữ nguyên.
- Workspace `zcu102_build/` (copy ip_repo rev170 gồm graphop TILED; KHÔNG mang src_backup).
- **Rev200 RTL (elaborate PASS)**: act_uram ultra→block + DEPTH 98304→131072 (**2MB** — khít
  WAW17/BAW21, zero bus ripple; phá storage wall L5 1690KB<2048KB); wt_uram ultra→block;
  reg_feat_en [14:0]→[16:0]: **bit15 SCATOUT** (scatter base=reg_out_addr[20:4], tái dụng reg rảnh
  của layer RESOUT) + **bit16 RESINBASE** (ls_act_off_eff=ls_act_uram_off+reg_in_addr[20:0], vá
  3 đường đọc: byte-1x1/W11[20:4]/DW aw_base; c3 paths GIỮ vùng 0). Cả hai GATED — flow cũ y nguyên.
  component.xml coreRev **200**.
- **build_zcu102.tcl**: tạo project+BD TỪ ĐẦU (extract connectivity từ KV260 project bằng
  write_bd_tcl; đơn giản hoá bỏ HPM1). PS preset ZCU102 (nếu board files) + delta HPM0/HP0/IRQ0/
  PL0=200. Global synth + dont_touch_wt.xdc (copy) + AggressiveExplore. Tự check part.
- Part `xczu9eg-ffvb1156-2-e` CHƯA cài trong Vivado (PARTS: 0) — user đang cài. Build chạy sau.
- Thứ tự Phase 1 orchestration QUAN TRỌNG (tránh đè): L3→**L4**→L6→DW7(CPU)→L8→graphop→L5
  (L4 trước L8 vì L8 ACTL@0 đè x2). Bản đồ vùng act_uram trong ARCH_PLAN §2.

## 2026-07-09 — On-chip graph-op engine (graphop) — sim PASS + integrated + build170 running

### graphop_engine.v (MỚI) — CAT / RESID_CAT trên FPGA
- Đọc/ghi act_uram trực tiếp, tái dùng rr_unit rescale. LAYER_TYPE=4 (CONV_TYPE_GRAPHOP).
- Math: `rescale_int8(x,fs,ts)=round(x*fs/ts)clip`; `cat=concat(rescale(a),rescale(b))`;
  `resid_cat=concat(clip(rescale(cv1)+rescale(mc2)),rescale(cv2))`.
- **Verify: `tb_graphop_engine.v` + `gen_graphop_tb.py` → `=== TB_GRAPHOP_ALL_PASS ===` (bit-exact)**.
- Bug bắt qua sim + đã fix: (1) RR_LAT=4 sai→5; (2) counter chung RD_LAT/RR_LAT; (3) pipeline chồng
  (viết lại tuần tự, counter riêng mỗi latency); (4) `wrbase<=21'd0` quên addr_dst → ghi đè source A
  (fix `wrbase<=addr_dst`, phase C: `addr_dst+ch_ab*n_pos`).
- Bug testbench: multi-driver au_rd_baddr (X) → tách host_* + wire mux; thiếu `#1` sau posedge (DUT
  không thấy start); ADDR_DST=300 lệch word (act_uram ghi word 128b, phải chia hết 16 → 192).

### Tích hợp unified_acc_top.v — elaborate PASS (elab170), gate an toàn conv_type=4
- `reg_conv_type [1:0]→[3:0]` (2-bit truncation aliasing LAYER_TYPE 4-15→0-3, bug đã fix).
- graphop chạy FSM ĐỘC LẬP (16 state FSM chính đã dùng hết, không chỗ thêm state) — conv_type=4 thì
  FSM chính đứng ST_IDLE, graphop_engine tự chạy. Register mapping tái dùng in/wt/out_addr, stride,
  win, n_ocb, n_icb, pad, cin/cout/hin (m0/sh đóng gói 16b+5b).
- Mux act_uram gate `graphop_active`; conv_type≠4 = hành vi cũ y nguyên (regression 106 det phải giữ).
- component.xml: +graphop_engine.v (×2 file-group), coreRevision 169→170. build170.tcl launched.

### PHÁT HIỆN quan trọng cho chuỗi end-to-end (build 171)
- **act_uram KHÔNG có đường DMA ra host** → không test graphop độc lập; phải test trong chuỗi thật
  (graphop ghi act_uram → conv kế đọc act_uram → out_bram → DMA ra).
- **Scatter đã tổng quát**: `oc_deint` (feat bit12 RESOUT) scatter out_bram→act_uram cho MỌI conv
  (kể cả conv1×1). KHÔNG cần module RTL mới → mảng #2 chủ yếu là HOST orchestration.
- **GAP nhỏ cần build 171**: scatter hardcode `base=deint_pos` (luôn vùng act_uram ~0). graphop cần
  3 tensor A/B/C ở 3 vùng khác nhau đồng thời → thêm **scatter base-offset per-layer** (host set,
  tái dùng reg_out_addr — layer RESOUT không DMA out nên out_addr rảnh). Đổi ~3 dòng ở oc_deint base:
  `base = {1'b0, deint_pos} + reg_scat_base`. Backward-compat khi reg_scat_base=0.
- Host TODO build171: set RESOUT bit L03/L04/L08 (scatter về 3 vùng); kick graphop conv_type=4 với
  addr_a/b/c trỏ 3 vùng, addr_dst=vùng L09 mong đợi; set RESIN (bit14) L09 đọc act_uram.
- Đánh giá payoff: graphop chỉ dời ~14 graph-op CAT/RESID_CAT khỏi host (một phần của 0.29s
  deint/graph/pack) + né DMA round-trip các transition → ước 0.81s→~0.7s (~1.4 FPS), TĂNG KHIÊM TỐN.
  Prize thật (streaming dataflow né memory-wall) BỊ storage-wall chặn — graphop KHÔNG mở khoá nó.

## 2026-07-09 — REVIVAL graphop end-to-end, STEP 1: graphop OUTPUT tiled position-major — sim PASS

User chốt hồi sinh graphop end-to-end (build 171). Step 1 = sửa graphop xuất tiled cho conv1×1 consumer.
- **graphop_engine.v**: store address CHW `wrbase+ch_i*n_pos+pos` → **TILED** `dstw + pt*cpad + C_global`
  (pt=pos_i>>4; C_global=(phase_c?ch_ab:0)+ch_i; cpad=ceil((ch_ab+ch_c)/16)*16; dstw=addr_dst>>4).
  Bỏ wrbase (per-group CHW base); INPUT vẫn đọc CHW (host stage liền mạch). Khớp cách conv1×1 BI đọc
  act_uram: byte=(pt*N_IC+ic)*256+ch*16+sp.
- **Sim re-verify**: gen_graphop_tb.py N_POS=16→**32** (2 position-tile), CH_AB/C=3/2→**20/12** (Cout=32
  = 2 channel M-block, bội 16 không padding); golden xuất qua to_tiled(). tb addr A=0/B=640/C=1280/DST=1664
  (đều /16); timeout 200us→2ms (test lớn hơn ~450us). **`=== TB_GRAPHOP_ALL_PASS ===`** (resid_cat +
  cat, 1024 byte bit-exact). Write pattern kiểm bằng mắt: waddr 104,136,105,137 = dstw+pt·32+C ✓.
- BUG quy trình: sửa tb rồi chỉ chạy xelab (dùng lib cũ) → timeout vẫn 200us. PHẢI xvlog LẠI tb trước xelab.
- Còn lại: step 2 host staging A/B/C→act_uram + kick graphop + L5 đọc act_uram; build 171; board.

## 2026-07-09 — REVIVAL STEP 2: STORAGE WALL chặn full-resident graphop (transition lớn)

Điều tra staging path (full-resident vs host-staged) → chạm storage wall — đúng giới hạn đã biết:
- Spatial L3/L4/L8 = **104²=10816** (trace: L0 s2→208, L1 s1, L2 s2→104, L3-L8 s1). N_OCB=1 cả 3 (RESOUT OK).
- **Transition L5 full-resident**: A=L3(32×10816=338KB) + B=L8(338KB) + C=L4(338KB) + out=L5in(64×10816
  =676KB) = **1690KB > act_uram 1536KB → TRÀN ~154KB**. Naive full-resident KHÔNG FIT transition lớn nhất.
- 2-pass (AB-phase rồi C-phase, giải phóng A/B nạp C): peak = A+B+out = 338+338+676 = 1352KB < 1536 (fit)
  nhưng cần 2-kick + host quản lý vùng act_uram (mong manh) + scatter-base-offset/load-only RTL + board debug.
- Host-staged KHÔNG né được wall: 3 input + output vẫn phải đồng cư act_uram (trừ khi 2-pass như trên).
- Transition NHỎ hơn fit dễ (L12 52²=2704, L19 26²=676; 13²=169 KHÔNG chia hết 16 → loại) nhưng payoff
  càng nhỏ (graph-op ở đó chỉ vài % của 0.29s host). 
- **KẾT LUẬN**: step 1 (graphop tiled-output) = cải tiến RTL thật, verified, GIỮ. End-to-end resident =
  chạm storage wall (fundamental, = lý do paper dùng streaming tile-buffer). Đường thật = streaming/tiling
  dataflow (redesign lớn), KHÔNG phải revival nhanh. Báo user quyết.

## 2026-07-09 — Build 171 SCOPE CORRECTION: layout mismatch graphop(CHW) ↔ conv1×1(tiled)

Điều tra trước khi code build 171 → phát hiện ước tính "~3 dòng RTL + host" SAI:
- **Consumer mọi graph-op (resid_cat/cat) đều conv1×1** (L05/12/19/26/35/31... trong layer_params.json).
  conv1×1 BI đọc act_uram TILED `(N_POS,N_IC,16ch,16sp)`; graphop đọc/ghi CHW-flat (channel-major).
- Chuỗi resident hiện có (RESACT=2 L00→L01→L02) TOÀN conv3×3 (0,1,2 trong c3_layer_params) — conv3×3
  đọc CHW plane nên scatter→CHW→conv3×3 khớp. Không có tiền lệ CHW→conv1×1 resident.
- **Build 171 thật cần đồng thiết kế layout xuyên producer→scatter→graphop→consumer**: graphop xuất
  TILED position-major (đổi store addr `C·npos+pos`→`pos·Cpad+C`) + scatter cấp đúng input layout +
  re-sim + build + board debug. NHIỀU VÒNG, không phải 1 build nhanh. Payoff vẫn ~1.4 FPS (khiêm tốn).
- Đã báo user re-decide. graphop milestone (sim bit-exact + timing +0.019 + board-regression 106det)
  ĐÃ ĐỦ cho luận văn độc lập với con số FPS end-to-end.

## Phase 0 — Offline Pipeline — PASS — 2026-06-03

### Trạng thái artifact
| Artifact | Path | Kết quả |
|---|---|---|
| QAT checkpoint | `checkpoints/yolov5s_pruned_newstem.pt` | ✅ 10.11 MB |
| fused base | `yolov5/runs/export/pruned_fused.pt` | ✅ tồn tại |
| QAT best.pt | `yolov5/runs/train/qat_3fixes13/weights/best.pt` | ✅ tồn tại |
| metadata.json | `yolov5/runs/quant/extracted/metadata.json` | ✅ 63 layers |
| requant_params.json | `yolov5/runs/quant/extracted/requant_params.json` | ✅ 59 entries |
| weights_bin/ | `yolov5/runs/quant/extracted/weights_bin/` | ✅ 63 files |
| bias_bin/ | `yolov5/runs/quant/extracted/bias_bin/` | ✅ 63 files |
| test image | `test_images/pklot_sample.jpg` | ✅ 640x640 PKLot |

### Ghi chú thực tế vs pipeline spec
- **63 layers** (không phải 59): model đã được update với Detect head quantized (QAT Fix 2)
- **4 fp32-output layers**: `[42, 49, 56, 62]` (không phải `[41, 48, 55]` như spec)
- Layers 60, 61, 62 = `model.25.m.0/1/2` (Detect head) đã INT8

### Step 2 — N:M Sparse Pruning (tools/gen_nm_sparse.py)
- **63 layers** encoded: 62 COO, 1 DENSE (layer 62, final classifier)
- Average sparsity: 81.25% (COO layers), 0% (dense)
- Packed size: 1.60 MB (từ 5.14 MB dense = 68.9% reduction)
- Output: `weights/` — 63 × (L{XX}_packed.bin + L{XX}_N_config.txt)

### Step 3 — Golden Vectors (tools/gen_golden.py)
- Load model: `pruned_fused.pt` + `qat_3fixes13/weights/best.pt`
  - `missing=0 unexpected=0` (load hoàn hảo)
- Forward pass: ảnh PKLot 640x640 → letterbox 416x416
- **63 input tensors** + **59 output tensors** saved (4 fp32-out skipped)
- Output: `golden/` — 122 .bin files + layer_shapes.json
- Convention: `in[i] = float_in / S_in[i]`; `out[i]` = hook Conv **wrapper** output (post-BN+Act) / S_out[i]
- Bug đã sửa: không dùng `float_in[exec_next]` làm proxy vì exec_order ≠ data-flow order trong CSP blocks

### Step 4 — Verification (tools/verify_sparse_golden.py)
- Mode: DENSE original weights, threshold 25%, toàn bộ **59/59 layers**
- Kết quả: **ALL PASS ✅** (2026-06-03, kiểm tra lần 2)
- off-by->1 range: 0.00% (Detect head) → 22.17% (L25 model.9.cv2, backbone sâu)
- Tất cả diff chỉ là d2/d3/d4 = BN rounding noise (16-bit M0 precision)
- COO encode/decode round-trip: **0 diff** trên mọi layer (L01, L07, L25 verified)

### Step 5 — cnn_pkg.sv (tools/gen_cnn_pkg.py)
- Output: `rtl/pkg/cnn_pkg.sv` (72 lines)
- Nội dung: paper params, data widths, AXI base addresses, N_CFG[63], M0_CH0[63], SHIFT_CH0[63], LAYER_CFG[63]

### Step 6 — RTL Structure
- `rtl/pkg/`, `rtl/leaf/`, `rtl/spec/`, `rtl/top/`, `tb/` — tạo xong

---

## Phase 1.0 — RTL Leaf Modules viết xong (Verilog 2001) — 2026-06-03

### Chuyển sang Verilog 2001 thuần (quyết định học thuật)
- Xóa toàn bộ `.sv` files
- `cnn_pkg.sv` → `rtl/pkg/cnn_defines.vh` (`` `define `` + AXI offsets)
- Tất cả `logic` → `wire`/`reg`, `always_ff/@(posedge clk)`, `always_comb/@(*)`
- Port arrays không dùng được → flat ports (vals_in_0..15, idx_in_0..15)
- `import cnn_pkg::*` → `` `include "cnn_defines.vh" ``

### Leaf modules (Verilog 2001)
| Module | File | CLAUDE.md spec | Ghi chú |
|---|---|---|---|
| mac_unit | `rtl/leaf/mac_unit.v` | §8.2 | 2-stage pipeline, 1 DSP48E2 |
| accumulator | `rtl/leaf/accumulator.v` | §8.11 | clr priority over en |
| rr_unit | `rtl/leaf/rr_unit.v` | §8.4 | 3-stage, LeakyReLU >>>3 |
| bitmap_decoder | `rtl/leaf/bitmap_decoder.v` | §8.5 | bm & (~bm+1), OR-tree |
| coo_decoder | `rtl/leaf/coo_decoder.v` | §8.6 | counter-based drain |
| weight_decoder | `rtl/leaf/weight_decoder.v` | §8.7 | MUX COO/Bitmap, use_bm_r |

### Testbenches (Verilog 2001)
- `tb/tb_mac_unit.v` — TC_01..06
- `tb/tb_accumulator.v` — TC_01..04
- `tb/tb_rr_unit.v` — TC_01..06 (dùng ref function với Layer-01 M0/shift thực)
- `tb/tb_bitmap_decoder.v` — TC_01..06
- `tb/tb_coo_decoder.v` — TC_01..04
- `tb/tb_weight_decoder.v` — TC_01..04
- `tb/run_all_sim.tcl` — batch XSim, `-i rtl/pkg` include path

### Phase 1 XSim Results — PASS — 2026-06-03
Tool: Vivado XSim 2022.2 (`D:\Study_sofware\Xilinx\Vivado\2022.2`)

| Module | TC count | Kết quả | Ghi chú |
|---|---|---|---|
| mac_unit | 6 | ✅ PASS | 2-stage pipeline, DSP hint |
| accumulator | 4 | ✅ PASS | clr tách riêng, negedge-based TB |
| rr_unit | 6 | ✅ PASS | 48-bit multiply, ref_rr khớp DUT |
| bitmap_decoder | 6 | ✅ PASS | bm & (-bm), OR-tree, TC_03 8-bit fix |
| coo_decoder | 4 | ✅ PASS | counter drain, n_cfg=0 edge case |
| weight_decoder | 4 | ✅ PASS | COO/Bitmap MUX, path switch |

**ALL 6/6 PASS ✅**

Bugs sửa trong quá trình sim:
- `timescale` missing → thêm vào tất cả RTL files
- UTF-8 BOM → strip từ tất cả files
- TB `do_load` thiếu `@(posedge)` → posedge không latch state
- TB `drain` advance clock trước sample → thay bằng `#1` + sample tại chỗ
- `accumulator` TB: `en=0` trước posedge cuối → bỏ sót accumulate lần 4

---

## Phase 2.0 — i_spec.v rewrite + tb_i_spec — PASS — 2026-06-03

### Lỗi kiến trúc trong i_spec.v cũ
- i_spec cũ dùng 1 `weight_decoder` duy nhất, broadcast cùng `wd_weight` cho wt_g0..g3
- Theo paper §8.10: "Different groups may have different weights" → cần G=4 decoders độc lập
- i_spec.v đã viết lại với 4 `weight_decoder` instances (u_wd_g0..g3)
  - Cùng indices (`wt_idx_0..15`) cho tất cả groups
  - Giá trị weight riêng biệt per group (`wt_val_g0_0..15` .. `wt_val_g3_0..15`)
  - Bitmap values cũng per group (`bm_val_g0..g3`)
  - addr dùng wd_addr_g0 (tất cả decoders synchronize vì cùng n_cfg + indices)

### tb_i_spec.v viết lại
- TC_01: N=2, single group active → psum_g0=10, g1..g3=0 ✅
- TC_02: acc_clr resets all PSUMs → 0 ✅
- TC_03: All 4 groups different weights → g0=27, g1=4, g2=20, g3=-14 ✅
- TC_04: Consecutive tiles without clr → psum_g0=54, g3=-28 ✅

### XSim Results — PASS — 2026-06-03
Tool: Vivado XSim 2022.2
- 9/9 test cases PASS (TC_01a, TC_01b, TC_02, TC_03a..d, TC_04a, TC_04b)
- Simulation time: 395 ns

**Phase 2 ALL PASS ✅**

---

## Phase 3.0 — Top-level modules + conv1x1_top — PASS — 2026-06-03

### Modules viết mới (rtl/top/)
| Module | File | CLAUDE.md spec | Ghi chú |
|---|---|---|---|
| bram_ping_pong | `rtl/top/bram_ping_pong.v` | §8.13 | inferred dual-port BRAM, pp_bank_sel swap |
| tile_ctrl | `rtl/top/tile_ctrl.v` | §8.14 | row-major tile counter, halo/pad output |
| accu_top | `rtl/top/accu_top.v` | §8.15 | 1-in/2-in/4-in mode, function-based MUX |
| router | `rtl/top/router.v` | §8.16 | unicast/multicast/broadcast via valid gating |
| axi_lite_slave | `rtl/top/axi_lite_slave.v` | §8.17 | AXI4-Lite slave, W1SC ap_start, W1C ap_isr, SLVERR on invalid addr |
| conv1x1_top | `rtl/top/conv1x1_top.v` | §8.18 | IP wrapper, ext_wt ports for Phase 3 TB |

### TB helpers
- `tb/helpers/axi_lite_bfm.v` — AXI4-Lite master BFM (write/read tasks)
- `tb/helpers/mem_model.v` — 64KB byte-addressed memory model

### tb_conv1x1_top.v — PASS — 2026-06-03
| TC | Description | Result |
|---|---|---|
| TC_01a | Cin readback=16 | ✅ PASS |
| TC_01b | Cout readback=4 | ✅ PASS |
| TC_01c | N_config readback=2 | ✅ PASS |
| TC_01d | ap_idle=1 at reset | ✅ PASS |
| TC_02 | irq_out asserted after 25 cycles | ✅ PASS |
| TC_02b | ap_isr latched after ap_done | ✅ PASS |
| TC_03 | Invalid wr addr → SLVERR, no hang | ✅ PASS |
| TC_04 | Invalid rd addr → SLVERR + 0xDEADBEEF | ✅ PASS |

Bugs sửa trong quá trình sim:
- `wr_resp_lat` non-blocking race: `bresp <= wr_resp_lat` lấy giá trị cũ → đổi sang blocking `wr_resp_next`
- `ap_done` 1-cycle pulse cleared trước khi TB đọc → TB check `ap_isr` thay vì `ap_ctrl[1]`
- `$deposit` không support trong XSim → thay bằng `ext_wt_*` input ports trên conv1x1_top

**Phase 3 ALL PASS ✅**

---

## Phase 4.0 — conv3x3_top + dwconv3x3_top — PASS — 2026-06-03

### Modules viết mới (rtl/top/)
| Module | File | AXI base | Ghi chú |
|---|---|---|---|
| conv3x3_top | `rtl/top/conv3x3_top.v` | 0xa0010000 | IP_TYPE=1, 3x3 kernel FSM, LOAD+SHIFT depth_shuffle |
| dwconv3x3_top | `rtl/top/dwconv3x3_top.v` | 0xa0020000 | IP_TYPE=2, depthwise, accu_mode forced 1-in |

### Điểm khác biệt vs conv1x1
| Feature | conv1x1 | conv3x3 | dwconv3x3 |
|---|---|---|---|
| AXI stride/pad | SLVERR (IP_TYPE=0) | OKAY (IP_TYPE=1) | OKAY (IP_TYPE=2) |
| depth_shuffle mode | LOAD only | LOAD (kw=0) + SHIFT (kw=1,2) | LOAD + SHIFT (same as conv3x3) |
| Kernel loop | 1 pass | 9 passes (3×3) | 9 passes per channel |
| accu_mode | from AXI reg | from AXI reg | forced 2'b00 (1-in) |
| Cin/Cout | separate | separate | C (Cin=Cout=C) |

### XSim Results — PASS — 2026-06-03

| TB | TC | Description | Result |
|---|---|---|---|
| tb_conv3x3_top | TC_01a..d | AXI regs + stride/pad | ✅ PASS |
| | TC_02a/b | 3x3 kernel irq (58 cycles) + ap_isr | ✅ PASS |
| | TC_03 | stride=2 write OKAY | ✅ PASS |
| | TC_04 | invalid addr SLVERR | ✅ PASS |
| tb_dwconv3x3_top | TC_01a..c | C=8, stride, ap_idle | ✅ PASS |
| | TC_02a/b | depthwise irq (58 cycles) + ap_isr | ✅ PASS |
| | TC_03 | Cout mirrors Cin | ✅ PASS |
| | TC_04 | invalid addr SLVERR | ✅ PASS |

**Phase 4 ALL PASS ✅**

---

## Phase 5.0 — Vivado project prep + full regression — PASS — 2026-06-03

### Artifacts viết mới
| File | Mục đích |
|---|---|
| `tb/run_all_sim.tcl` | Full regression script: Phase 1..4, tất cả 11 TB |
| `vivado_rtl/create_project.tcl` | Vivado batch TCL tạo RTL project (chưa chạy synthesis) |
| `vivado_rtl/kv260.xdc` | Timing constraints @ 200 MHz, KV260 xck26-sfvc784-2LV-c |

### Full regression kết quả — Vivado XSim 2022.2

| Phase | TB | Kết quả |
|---|---|---|
| Phase 1 | tb_mac_unit | ✅ PASS |
| Phase 1 | tb_accumulator | ✅ PASS |
| Phase 1 | tb_rr_unit | ✅ PASS |
| Phase 1 | tb_bitmap_decoder | ✅ PASS |
| Phase 1 | tb_coo_decoder | ✅ PASS |
| Phase 1 | tb_weight_decoder | ✅ PASS |
| Phase 1 | tb_depth_shuffle | ✅ PASS |
| Phase 2 | tb_i_spec | ✅ PASS |
| Phase 3 | tb_conv1x1_top | ✅ PASS |
| Phase 4 | tb_conv3x3_top | ✅ PASS |
| Phase 4 | tb_dwconv3x3_top | ✅ PASS |

**11/11 PASS — zero regression ✅**

### Trạng thái sẵn sàng synthesis
- 30 Verilog modules compile sạch (0 ERROR, 0 WARNING)
- `vivado_rtl/create_project.tcl` sẵn sàng → chờ lệnh chạy synthesis
- Dự kiến resource: ~256 DSP48E2 (64%), ~60K LUT (51%), ~40 BRAM-36 (28%) cho KV260

**Phase 5 ALL PASS ✅**

---

## Phase 5.1 — RTL Deep Review + Bug Fixes — PASS — 2026-06-04

### 7 vấn đề tìm được và sửa

| # | File | Lỗi | Severity | Fix |
|---|------|-----|----------|-----|
| 1 | `axi_lite_slave.v` | Multi-driver: `ap_start` có 2 `always` block cùng drive | 🔴 Critical | Xóa always block thừa, chỉ drive trong write FSM |
| 2 | `axi_lite_slave.v` | `WR_ACCEPT` dùng `awready/wready` làm flag "đã nhận" — sai vì ready deassert ngay chu kỳ sau | 🔴 Critical | Viết lại FSM 4-state: `WR_IDLE→WR_ADDR→WR_DATA→WR_EXEC`, latch riêng biệt |
| 3 | `conv1x1_top.v` | `SHUF_LOAD` pulse 1 cycle rồi chuyển state ngay — depth_shuffle không đủ 1 cycle để latch | 🟡 Logic | Thêm state `ST_SHUF_WAIT` giữ SHUF_LOAD đúng 1 cycle |
| 4 | `conv1x1_top.v` | BRAM latency 1 cycle không được bù — `act[load_cnt]` latch data của địa chỉ trước đó | 🔴 Critical | Dùng `load_cnt - 1` làm index latch; tăng LOAD_ACT loop lên 17 cycles |
| 5 | `conv3x3_top.v` | Double assignment `state <= ST_PIX_CLR` rồi `state <= ST_KW_LOAD` trong cùng block — khó đọc | 🟡 Logic | Xóa assignment thừa, chỉ giữ `ST_KW_LOAD` cho next-row transition |
| 6 | `conv3x3_top.v` | `kw_cnt` không reset về 0 khi vào `ST_PIX_CLR` — sau pixel đầu tiên kw có thể != 0 | 🟡 Logic | Thêm `kw_cnt <= 2'd0` trong `ST_PIX_CLR` |
| 7 | `dwconv3x3_top.v` | Thiếu `kw_cnt <= 2'd0` trong `ST_PIX_CLR` | 🟡 Logic | Thêm reset kw_cnt |

### Regression sau fix — Vivado XSim 2022.2
**11/11 PASS ✅ — zero regression**

---

## Phase 5.2 — Wire 4 orphan modules into hierarchy — PASS — 2026-06-04

### Modules đã wire vào đúng vị trí

| Module | Vị trí | Logic |
|--------|--------|-------|
| `pch_gen` | `i_spec.v` | Thay inline addr calc; tính `poc_cnt×poc+g` per group. poc=0: tất cả groups dùng wd_addr trực tiếp |
| `rr_unit` | 3 conv tops | PSUM g0p0 → INT8 → output BRAM write FSM |
| `tile_ctrl` | 3 conv tops | tc_start tại ST_IDLE; tc_next khi tile done; pp_out_sel swap khi tile_last |
| `router` | 3 conv tops | BROADCAST mode (valid_in=wt_load_r) — sẵn sàng N_SPEC=4 |

### Bug phát hiện khi wire pch_gen
- poc=0: pch_gen tính `addr+g` → group 1,2,3 đọc addr sai
- Fix: poc=0 tất cả groups dùng `wd_addr_g0` trực tiếp

### Full regression — 11/11 PASS ✅

---

## Phase 5.3 — Logic Verification + End-to-End TB — PASS — 2026-06-04

### TBs mới (logic verification với expected values tính tay)

| TB | Nội dung | Kết quả |
|----|----------|---------|
| `tb_pch_gen.v` | TC_01..09: poc=0,1,2 × cnt=0..15 → verify per-group addr | ✅ 9/9 PASS |
| `tb_i_spec_e2e.v` | TC_01..03: poc=0,1,2 với activations g*16+p, weights khác nhau per group → verify PSUM tính tay | ✅ 12/12 PASS |
| `tb_conv1x1_e2e.v` | TC_01: AXI config. TC_02: act=50,wt=3→PSUM=150→rr→75→BRAM[0] | ✅ TC_01,TC_02 PASS |

### RTL bugs tìm thêm trong quá trình e2e TB

| Fix | File | Vấn đề |
|-----|------|---------|
| `psum_ready` signal | `conv1x1_top.v` | rr_unit nhận `mac_valid` (mid-accumulation) thay vì signal sau N NZ done |
| `out_bram_wr_en` → wire | `conv1x1_top.v` | Registered write enable gây addr+data misalign với BRAM write |
| `pp_out_sel` deferred swap | `conv1x1_top.v` | Swap xảy ra đồng thời rr_valid → next layer ghi sai bank |
| `out_bram_wr_addr` reset | `conv1x1_top.v` | Reset trong ST_DONE sau rr_valid, không trong ST_WAIT_DMA |

### Verified pipeline (tính tay):
- act=50, wt=3 → PSUM=150 → rr(16384,>>15)=75 → BRAM[0]=75 ✅

### Full regression 11/11 PASS ✅

---

## Phase 5.4 — Deep RTL Review + Architectural Bug Fix — 2026-06-04

### RTL bugs phát hiện qua manual code review

| # | File | Bug | Severity | Fix |
|---|------|-----|----------|-----|
| 1 | `i_spec.v` | `mul_group_array` nhận `act_sel_g` (scalar broadcast) tới tất cả P=16 MAC ports trong cùng group — sai kiến trúc. Paper §8.10: P MACs nhận **khác activation** `act[g][p]` per position, không phải cùng 1 giá trị. Kết quả: tất cả 16 accumulators trong group g tính cùng 1 value → output chỉ có ý nghĩa tại p=0, p1..15 duplicate | 🔴 Critical | Wire trực tiếp `ds_g{g}p{p}` từ depth_shuffle vào từng MAC port — mỗi position p nhận activation riêng biệt |
| 2 | `i_spec.v` | `pch_gen` được instantiate nhưng addr output không được dùng để MUX activation — logic address mapping bị bypass hoàn toàn | 🔴 Critical | Remove pch_gen instance; kiến trúc đúng là ALL P activations feed vào P MACs (không cần address MUX per group) |
| 3 | `conv3x3_top.v` | `rr_unit.valid_in = mac_valid_s0` — fire sau mỗi MAC cycle trong 9 kernel passes, không phải sau khi PSUM đầy đủ. Output sẽ là PSUM intermediate, không phải final | 🔴 Critical | Thêm `psum_ready` signal, chỉ assert khi kh=2, kw=2, flush done |
| 4 | `dwconv3x3_top.v` | Cùng bug như #3 | 🔴 Critical | Cùng fix |
| 5 | `tb_i_spec.v` | Expected values sai — assume "all P positions = same value" dựa trên architecture cũ (broadcast act) | 🟡 TB bug | Rewrite với per-position values: psum_g0[p3]=30, psum_g0[p7]=60 |
| 6 | `tb_i_spec_e2e.v` | Expected values sai — dùng pch_gen addr mapping cũ với poc=1 | 🟡 TB bug | Rewrite với poc=0 direct wire scenario |

### Files đã sửa
- `rtl/spec/i_spec.v` — remove pch_gen, wire ds_g{g}p{p} trực tiếp tới mul_group_array
- `rtl/top/conv3x3_top.v` — thêm psum_ready, fix rr_unit valid_in
- `rtl/top/dwconv3x3_top.v` — thêm psum_ready, fix rr_unit valid_in
- `tb/tb_i_spec.v` — rewrite expected values
- `tb/tb_i_spec_e2e.v` — rewrite với scenario mới

### Regression kết quả — Vivado XSim 2022.2 GUI — 2026-06-04

| TB | TC count | Kết quả | Ghi chú |
|----|----------|---------|---------|
| tb_mac_unit | 8 | ✅ PASS | +TC_07 reset pipeline, +TC_08 sign test |
| tb_accumulator | 6 | ✅ PASS | +TC_05 en=0 hold, +TC_06 INT32 boundary |
| tb_rr_unit | 9 | ✅ PASS | +TC_07 relu boundary, +TC_08 back-to-back |
| tb_bitmap_decoder | 6 | ✅ PASS | |
| tb_coo_decoder | 6 | ✅ PASS | +TC_05 N=15 negative, +TC_06 n_cfg=4 boundary |
| tb_weight_decoder | 6 | ✅ PASS | +TC_05 COO boundary, +TC_06 Bitmap boundary |
| tb_depth_shuffle | 5 | ✅ PASS | |
| tb_pch_gen | 9 | ✅ PASS | |
| tb_i_spec | 10 | ✅ PASS | broadcast p0==p15 confirmed, pch_gen đúng |
| tb_i_spec_e2e | 9 | ✅ PASS | poc=1 pch_gen mapping: 260/290/-92/521 đúng |
| tb_conv1x1_top | 6 | ✅ PASS | +TC_05 ap_idle restore, +TC_06 stride SLVERR |
| tb_conv3x3_top | 4 | ✅ PASS | IRQ sau 58 cycles (9 kernel passes) |
| tb_dwconv3x3_top | 7 | ✅ PASS | IRQ sau 58 cycles, depthwise symmetric |
| tb_conv1x1_e2e | pass | ✅ PASS | PSUM=150 → rr → 75 → BRAM[0]=75 ✅ |

**14/14 PASS — zero regression ✅**

### Architectural fix trong phase này
- `i_spec.v`: restore `pch_gen` instance, `act_sel_g` broadcast đúng theo paper
- `conv3x3_top.v` + `dwconv3x3_top.v`: thêm `psum_ready` signal, fix `rr_unit.valid_in`

## Rev165 (2026-07-07) — XÓA deint_bram, giải phóng 512KB BRAM cho kế hoạch resident-act
- unified_acc_top.v: bỏ instance `u_deint_bram` (32768×128b = 512KB = ~128 BRAM tile, 89% BRAM);
  `dma_out_data128 = out_bram_rd128` trực tiếp (bỏ mux deint_dma_active); deint_bram_rd128 tie 0.
- GIỮ oc_deint (scatter sim-proven) để tái dụng hướng scatter→act_uram (resident activation).
- Lý do: deint-on-HW đo NET-NEGATIVE (Rev164 wash); host DEINT=0 nên đường này chết trong bit đang chạy.
- coreRevision 164→165. Build: build157.tcl (maxThreads/jobs = 10 theo yêu cầu user).

## Rev166 (2026-07-07) — FIX WNS -0.060: tách mux đọc act_uram + post-route phys_opt Aggressive
- act_uram.v: mux byte 16:1 (rd_word→rd_byte) tách thành 4:1 32b (rd_q32) + 4:1 byte, trải lên
  2 tầng reg có sẵn. RD_LAT=3 GIỮ NGUYÊN, byte chọn y hệt → bit-exact theo cấu trúc.
- Lý do: Rev165 retiming dồn URAM-cascade+mux16:1 vào 1 cycle → path fail 60ps (9 endpoints,
  toàn rd_byte_reg trong act_uram).
- build166.tcl: + STEPS.POST_ROUTE_PHYS_OPT_DESIGN.ARGS.DIRECTIVE = AggressiveExplore. 10 core.

## PIPELINE N-ẢNH (2026-07-07) — FPS 1.0 → 1.69 (+70%), HOST-ONLY, Rev166
- inference_unified_v4.py: PIPE env (PIPE=1 bật flock /tmp/hw_unified.lock quanh [reg→kick→wait]
  per layer; PIPE_ACT_OFF/PIPE_OUT_OFF = vùng DMA riêng per process). Weights share (ghi trùng).
- Vùng DMA 4 process: ACT 0/32/40/48MB + OUT 12/36/44/52MB (vùng netseq NS=0 không dùng).
- taskset pin 1 core/process + OPENBLAS_NUM_THREADS=1 (thiếu pin: 2-proc chỉ +10% vì DDR contention).
- Đo (parking106.jpg, 106 det mọi process): 1p=0.95 | 2p=1.37 | 3p=1.52 | 4p=1.65 | 4p×16f=1.694 fps.
- Bão hòa DDR: per-frame latency tăng 1.05→~2.3s; throughput +70%. Board: pipeline2/3/4.sh.

## DW-ON-CPU + OT verify (2026-07-07 chiều) — pipeline-4: 1.69 → 1.84 FPS
- OTV=1 (bit12 OT out-transpose, feat_en[11]): verified board 106 det — OUT pos-major [pos][ob][OCK][16].
  NHƯNG phân tích graph: chuỗi c1→c1 "sạch" chỉ 3 cặp (L26→30, L31→32, L35→39) vì shuffle toàn
  branch (resid_cat/cat) → zero-copy/fuse chỉ +~1%, KHÔNG làm. OTV giữ làm flag inert.
- DWCPU=1: DW 8 layer chạy CPU bằng dw_cpu_fast (9 shift-add vectorized + requant y hệt quant_conv;
  quant_conv gốc là Python-loop 55s — KHÔNG dùng). Single-frame hòa (1.06 vs 1.01) nhưng
  pipeline-4 THẮNG +9% nhờ nhả khoá HW 0.14s/frame: 1.694 → **1.842 fps (64 frames, 106 det)**.
- Board script: pipeline4dw.sh. FPS timeline: 1.00 → 1.69 (pipe4) → 1.84 (pipe4+DWCPU).

## DW-CPU v1 (2026-07-07): pad int8 + product int16 → dw_cpu 0.217→0.186s, pipeline-4 1.842→1.910 fps
- bench_dw.py (board, 8 shape DW thật, bit-exact assert): v0 int32-pad 229ms | v1 int8-pad+int16 173ms
  | v2 block-8 284ms (numpy overhead > cache win) | v2b block-32 194 | v5 einsum 424 (và einsum int16
  TRÀN 9-tap — phải int32). Winner v1 áp vào dw_cpu_fast.
- 106 det giữ nguyên. FPS: 1.00 → 1.69 (pipe4) → 1.84 (DWCPU) → 1.91 (DW v1).

## OTV3 c3-OT (2026-07-07): verified 106 det NHƯNG c3_deint 0.078s = KHÔNG nhanh hơn (cùng khối
## lượng permute). Flag OTV3 giữ inert. Kết luận: deint 0.164s tổng = trần transpose numpy trên A53.

## Rev167 WIP (2026-07-07): RESIDENT-ACT bước 1 — oc_deint tổng quát (base,stride)
- oc_deint.v: pos/n_pos → base/stride (addr = base + oc*stride). CHW cũ: base=pos, stride=n_pos
  (hành vi Y HỆT — refactor trung tính). M-block mới: base=pos*cpad (LS tính tăng dần), stride=1.
- unified_acc_top.v: instance đổi tên port tương ứng. THIẾT KẾ ĐẦY ĐỦ: memory project_resident_act.md
  (đích act_uram mux, LS skip ACTL, danh sách cặp, reg map, verify plan). Build khi xong = 10 CORE.

## USER CHỐT (2026-07-07): DW trên HW (mặc định). pipeline4.sh = 1.692 fps / 106 det.
## DWCPU=1 (pipeline4dw.sh = 1.910) giữ làm biến thể tuỳ chọn qua env, không phải mặc định.

## DWASYNC latency-mode (2026-07-07 tối) — DW-CPU thread ‖ HW cùng frame
- inference_cpu.py run_graph: reorder 8 shuffle block (cv1→mc1→DW→cv2→mc2) — thuần dataflow,
  regex 8/8 block, backup .bak_dwasync. inference_unified_v4.py: _DwFuture (thread, numpy nhả GIL)
  + dispatcher resolve future + env DWASYNC.
- Board: DWASYNC forward 1.00-1.01s | DW-HW 1.03 | DWCPU-sync 1.05 — DWASYNC TỐT NHẤT latency.
  106 det cả 3 mode (reorder không đổi kết quả ✓). Overlap thực ~25ms (cv2 nhỏ hơn DW + memory
  contention) so với lý thuyết 60ms.
- User intent: latency mode ưu tiên (ứng dụng đích PCB defect-style); parking chỉ để test model.

## L60/61 LÊN HW (2026-07-07 tối) — forward 1.00 → 0.98s, BIT-EXACT (106 det giữ nguyên)
- Nguyên nhân bị bỏ ngoài: make_unified_weights.py điều kiện `Cout%4!=0: skip` (L60/61 Cout=21).
  KHÔNG phải thiếu requant (req có đủ), không phải RTL.
- Fix packer: pad W/bias/M0/shift 21→24 (bội 4, kênh pad zero → out 0, host crop [:21]);
  layer_params ghi Cout=21 THẬT (Cout_real) để crop đúng.
- ⚠️ Bài học pack: dir ns12 CŨ pack với N_CFG bảo thủ (L03: N=4/WT_BLK 544) vs script hiện tại
  auto N=3/496 → 26 blob lệch NHƯNG đều hợp lệ (mỗi layer tự mô tả qua params). AN TOÀN = merge:
  copy dir cũ nguyên si + append 2 blob mới → weights_unified_c3_ns12b (43 layer).
- Board: L60 HW 6.2ms + L61 HW 2.7ms (từ 13.4+7.6 CPU) = tiết kiệm ~12ms. UNI_NAME MỚI = ns12b.

## HEADQ — FP32-head PTQ INT8 lên HW (2026-07-07 đêm): forward 0.99 → 0.91s, 106 det GIỮ
- ptq_head.py: M0/shift = in_sc*w_sc/out_sc per-channel, chuẩn hoá M0 về [2^15,2^16) →
  requant_params_headq.json (req gộp). Packer thêm env REQ_FILE. Pack L42/49/56 → merge dir
  weights_unified_c3_ns12c (46 layer; ns12b nguyên si).
- v4: env HEADQ=1 → (1) load req gộp; (2) force r=relu cho 42/49/56 (FP32_NO_LEAKY sẽ tắt nhầm
  leaky — float path vốn BẬT leaky); (3) sau HW: dequant out.astype(f32)*output_scale (graph
  expect float x42/x49/x56). L62 GIỮ CPU float (3ms, không scale chuẩn).
- A/B board: 106 det cả 2; lệch conf ≤0.002, box ≤3px (mức lượng tử logits) — chấp nhận được.
- LATENCY MODE BEST: DWASYNC=1 HEADQ=1 UNI_NAME=weights_unified_c3_ns12c → forward 0.91s.

## Rev167 C3FAST (2026-07-08) — 1x1-seq KHÔNG drain giữa entries (Fix A+B từ đo STPROF)
- ĐO SILICON (STPROF, sample dbg_kernel_state 0x180 — có sẵn, không cần build): COMPUTE chỉ 5%!
  LD_ACT_W 33% + WT_LOAD_W 20% + FLUSH_MAC 19% = 72% overhead. Per-entry 13 cyc khớp FSM đọc được.
- FIX: ST_KW_SHUF sau wt_load: nếu (seqfast && !swc_en && còn entry) → flush_k++/c3_idxhw mới/
  → LOAD_ACT_W ngay (bỏ COMPUTE+FLUSH giữa entries; acc cộng dồn; FLUSH 1 lần cuối — nhánh cũ
  trong FLUSH_MAC tự vô hiệu vì chỉ tới đó khi flush_k==n_cfg-1). Áp CẢ conv1x1 thường (cùng path).
- Gate A/B: feat bit13 = LAYER_MODE 0x4000 (env C3FAST=1 phía host, thêm sau khi build xong).
- Hazard đã audit: depth_shuffle đè sau ≥7cyc > mul-pipe 4; slot-0 đè sau ≥9cyc; mult idx*c3_hw
  dời từ FLUSH_MAC sang KW_SHUF (4×18b, 1 cyc — WNS sẽ nói). oc_deint(base,stride) trung tính đi kèm.
- Build build167.tcl 12 CORE (user yêu cầu). Resident-act = build KẾ TIẾP (RTL chưa viết xong).

## 2026-07-08 — DỌN FILE LỖI THỜI (chỉ thị: luận văn bám full bài 2023 N:M SPEC, không nhắc flow cũ)
- XÓA: `REFACTOR_TO_PAPER.md` (doc chuyển tiếp 3-kernel→unified, đã hoàn thành vai trò),
  `deploy_kv260.sh` + `kv260_setup.sh` (flow deploy cũ — thiếu set_pl0 200 + dma_region.ko),
  `tests/` (tham chiếu `vivado_rtl/` đã xóa, reg map 3-kernel), `scripts/gen_tb_hex.py`
  (fixture cho tb_cnn_acc_real.v không còn + encode bitmap pre-COO), `test_input.jpg` (ảnh test
  dư, chuẩn = test_images/input_image.jpg), `drivers/kv260_test/` (debug bring-up May–Jun:
  inference_bd/v2/pipeline cũ, debug_dma_hang...).
- CỨU: `drivers/kv260_test/dma_region/` → `drivers/dma_region/` (source dma_region.ko ĐANG chạy board).
- GIỮ (kiểm chứng tham chiếu trước khi quyết): `golden/` (inference_unified_v4.py:888 đọc
  layer_shapes.json+L00_in làm input mặc định), `weights/` (L*_N_config.txt = DominoSearch record
  + COO packed — đúng flow 2023), `runs/report/` (hình luận văn 5.2), `LEVEL_B_DESIGN.md`
  (thiết kế kiến trúc hiện hành), `scripts/finetune_stem.py` + `run_qat_3fixes.sh` (lineage model).

## 2026-07-08 — REV167 C3FAST BOARD-VERIFIED ✅ (WNS +0.006 @200MHz, 12 core)
- A/B board (parking106, DWASYNC+HEADQ ns12c): C3FAST=0 = y hệt Rev166 (106 det, c3_kick 0.221,
  kick 0.149, forward 0.91s warm). C3FAST=1: **106 det, c3_kick 0.221→0.170 (−23%), kick c1
  0.149→0.143, forward warm 0.91→0.85s (FPS 1.10→1.18)**.
- STPROF xác nhận cơ chế: FLUSH_MAC ~19%→0-5% (chỉ còn flush cuối tap; L00/01/02/09 biến khỏi
  top-6). Còn lại: LD_ACT_W 40-49% + WT_LOAD_W 24-29% = mục tiêu Fix B prefetch + resident-act.
- c1 chỉ −4%: kick c1 dominated bởi WT_LOAD/WRITEBACK, entry-loop ngắn hơn c3 9-pass.
- Bit: rev76_build/bd_wrapper_rev167_200_c3fast.bit (local+board). C3FAST=1 thành mặc định đề xuất.

## 2026-07-08 — REV168 RESIDENT-ACT (RTL + host, build đang chạy)
### Thiết kế (phase 1: chuỗi L00→L01, gate riêng từng bit để A/B)
- `act_bram_in.v` MỚI: BRAM bank 512KB (128 RAMB36, BRAM đang 0/144) gương act_uram 1:1
  (RD_LAT byte=3/word=2, mux 4:1+4:1 bài học Rev166) — chứa input plane L00 (519,168B).
- `unified_acc_top.v`: mux write act_uram = scatter (deint_scat_active) vs ACTL (~INBANK);
  oc_deint AW 15→17 (plane 86,528 word); read mux act_uram/bank theo feat[14]; khai báo
  ocd_*/deint_scat_active DỜI LÊN trước act_uram (bài học identifier-before-declaration).
- `layer_sequencer.v`: (1) act_total==0 → skip LS_ACTL (plane đã resident @0);
  (2) LS_DSCATW pos==last → kết thúc như LS_OUTW (KHÔNG LS_DOBDMA — deint_bram chết Rev165).
- Bit map: feat[12] (host 0x2000) = RESOUT scatter→act_uram@0; feat[14] (host 0x8000) = INBANK.
- Host: `_ResidentPlane` sentinel (chỉ .shape); L00 = 0x2000|0x8000 + skip out-DMA/de-tile;
  L01 = ACT_TOTAL=0 + skip plane-write. RESACT loại trừ --verify/PIPE (act_uram share).
### 2 BUG BẮT TRƯỚC SILICON
1. **c3_tapoff không init khi skip ACTL**: bình thường LS_ACTLW set `c3_tapoff<=c3_tap0`;
   skip → kick đầu dùng tapoff RÁC của layer trước. Fix: decode set thẳng từ c3_cfg[8:0]
   (l_c3w latch cùng cycle nên không dùng được wire c3_tap0). layer_sequencer.v decode block.
   Build đầu (bt13i03yp) đã kill + relaunch để lấy fix (bkhkyejyj).
2. **Scatter 48 kênh tràn act_uram**: engine luôn sinh OCK=48 oc/kick (16 junk khi Cout=32);
   deint_ock=l_outt[11:4]=48 → max addr 47*2704+2703=132,495 word > depth 98,304 (nguy cơ
   alias đè plane). Fix HOST: resident_out → REG_OUT_BYTES=Cout*16 (deint_ock=32, max
   86,527 = vừa khít plane, không junk). inference_unified_v4.py conv3x3_hw.
### Verify khi bit ra: A/B RESACT=0 (y hệt Rev167) → RESACT=1: 106 det + c3_pack/c3_deint
### giảm (L00 plane 1.38MB không còn qua host), forward 0.85 → ước ~0.82s.

## 2026-07-08 tối — BUILD168 lần 1: WNS -0.566 (1019 endpoint, TNS -85) — ROOT CAUSE + FIX
- 100% path fail từ u_bram_actin: Vivado map mem 32768×128 BRAM thành CASCADE DOUT 7 tầng
  RAMB36E2 (logic 3.344ns/5ns). act_uram (mux 2:1 mới thêm) KHÔNG dính fail nào.
- FIX: act_bram_in.v thêm (* cascade_height = 1 *) → mỗi bit = 1 RAMB36E2 chế độ 32K×1
  (DEPTH 32768 = 2^15 khớp nguyên khối, hết cascade). Build lại cùng build168.tcl.
- Bit WNS-fail backup: bd_wrapper_rev168_200_resact_WNSFAIL.bit (KHÔNG load board).
- Bài học watcher: background task chết theo phiên → vivado bị kill giữa open_run; bit vẫn
  ghi xong trước đó (write_bitstream nằm trong impl run). Check tasklist trước khi kết luận.

## 2026-07-08 tối — REV168 RESIDENT-ACT ✅ BOARD-VERIFIED (build lần 2, WNS +0.031 @200MHz)
- cascade_height=1 fix ăn: WNS -0.566 → **+0.031** (tốt hơn cả Rev167 +0.006).
- A/B board (parking106, C3FAST=1 DWASYNC HEADQ ns12c):
  - RESACT=0: y hệt Rev167 từng số (106 det, 0.85s warm, c3_kick 0.170, c3_deint 0.076).
  - RESACT=1: **106 det (bit-exact qua đường on-chip)**, c3_deint 0.076→**0.051** (-25ms =
    đúng dự đoán L00 de-tile 1.38MB), c3_hw 0.263→0.232, c3_kick 0.170→0.168 (scatter ≈ rẻ
    hơn DMA-out per-pos), forward warm **0.85→0.82s (FPS 1.22)**.
- LATENCY-MODE BEST CMD từ giờ: RESACT=1 C3FAST=1 DWASYNC=1 HEADQ=1 UNI_NAME=ns12c.
- Bit: bd_wrapper_rev168_200_resact.bit (board+local).

## 2026-07-08 tối — REV169 IN-PLACE LAG SCATTER (RTL xong, elab PASS, build đang chạy)
### Thiết kế (phase 2: L01 in-place → chuỗi L00→L01→L02 full on-chip)
- Nguyên lý: kick hàng r đọc input hàng r-1..r+1 → output pos p phải TRỄ D=2 hàng (2*ncolg
  pos-word) mới được đè plane. Fill: u_ocdeint ghi out kick vào lag_buf slot pos[4:0]
  (base={pos[4:0],6'b0}, stride=1). Drain: u_ocdrain (instance oc_deint thứ 2) đọc slot
  dpos[4:0], ghi act_uram base=dpos stride=npos. dpos đuổi 1:1 sau fill khi pos>=D.
- `lag_buf.v` MỚI: 2048×128b (32KB, 8 RAMB36; RD_LAT=1 khớp out_bram mà oc_deint kỳ vọng).
- `layer_sequencer.v`: reg mới 0x1F0 LAG_CFG ([7]=en [5:0]=D); LS_DRAINW (tái chế slot
  LS_DOBDMA chết) + LS_ENDDR/ENDDRW (xả D slot cuối sau kick chót); dpos reset per-ob.
- `unified_acc_top.v`: mux act_uram scatter = lag ? drain-engine : fill-engine (scat_win =
  scat_active|drain_active); fill base/stride mux theo lag_en_c.
- Reg map: AXI_OFF_LAG_CFG 9'h1F0 (slot cuối). Host ghi MỖI c3 call (0 nếu không lag — reg persist!).
- Host: RESACT=1 (phase1 L00→L01) / RESACT=2 (+L01 lag → L02 resident-in). L01 lag: bit
  0x2000 KHÔNG 0x8000 (input ở act_uram, không phải bank); assert same-shape s1 + D<=31.
### Verify plan: A/B RESACT=0 (y hệt 168 baseline) → RESACT=1 (y hệt 168: 0.82s) → RESACT=2
### (106 det + c3_deint 51→~26ms, forward ~0.79s kỳ vọng).

## 2026-07-08 đêm — BUILD169: WNS +0.005 OK NHƯNG bit bị xóa bởi instance vivado THỨ 2
- Lệnh launch có `&` bên trong background task → vivado chạy ĐÔI. Instance 1 build OK
  (BUILD169_WNS = 0.005, exit 21:06:58); instance 2 chờ project lock, mở lại project,
  reset_run impl_1 (XÓA bd_wrapper.bit vừa build) và đang build lại y hệt nguồn.
- Xử lý: để instance 2 chạy nốt (nguồn không đổi → bit giống hệt; kill = mất thêm 2h như nhau).
  Watcher đổi sang: đợi impl_1/bd_wrapper.bit xuất hiện (log instance 2 buffer, không tail được).
- RULE: KHÔNG bao giờ `&` trong lệnh vivado launch (run_in_background đã đủ); backup .bit
  RA NGOÀI impl_1 NGAY khi build xong (đã có rule, lần này bit chết trước khi kịp backup).

## 2026-07-08 khuya — REV169 BUG: ghi REG 0x1F0 → SIGBUS + FIX (bỏ reg, suy lag trong HW)
- TRIỆU CHỨNG: bit rev169 lần đầu → mọi config (kể cả RESACT=0) SIGBUS (exit 135) ngay đầu
  forward. Cô lập: sed tắt riêng `hw.wr(REG_LAG_CFG,...)` → RESACT=0 chạy lại OK (106 det).
- ROOT CAUSE: ghi offset 0x1F0 → IP trả SLVERR (case default) → /dev/mem store fault = SIGBUS
  trên ARM. (host mmap 64KB OK, addressBlock 4KB OK → không phải host/DECERR). Dù RTL có case
  AXI_OFF_LAG_CFG, bit thực thi không decode được 0x1F0 (nghi state incremental double-vivado;
  KHÔNG debug thêm vì fix dưới né hẳn).
- FIX (tốt hơn, ít coupling): BỎ reg 0x1F0. Suy lag TRONG HW:
  `lag_en = reg_feat_en[12] & ~reg_feat_en[14] & reg_layer_mode` (RESOUT mà input KHÔNG ở bank
  = ca in-place). D = 2*ncolg = {c3_cfg[12:9],1'b0} tính trong LS. Host chỉ set bit LAYER_MODE.
  Sửa: axi_lite_slave (gỡ reg_lag_cfg), unified_acc_top (lag_en_out + lag_en_c từ feat), LS
  (port lag_en thay lag_cfg, l_lagd tự tính), unified_cnn_top (lag_en_w 1-bit). Host bỏ REG_LAG_CFG.
- 0x1F0 đánh dấu RESERVED trong cnn_defines.vh (đừng dùng — SLVERR).
- Elab PASS. Build169b clean (1 cây vivado, KHÔNG &). Bit lỗi cũ: rev169_200_lag (0x1F0) BỎ.

## 2026-07-08 khuya — REV169B IN-PLACE LAG ✅ BOARD-VERIFIED (WNS +0.024 @200MHz)
- Chain L00→L01→L02 chạy TRỌN on-chip (L00 out→act_uram; L01 in-place lag scatter đè plane;
  L02 đọc plane resident). A/B:
  - RESACT=0: 106 det, forward 0.85s warm (y hệt Rev168, không SIGBUS — fix 0x1F0 ăn).
  - RESACT=2: **106 det BIT-EXACT** (15 top-conf trùng khít, diff=0), c3_deint 0.051→**0.024**
    (−27ms, khớp dự đoán de-tile L01 plane 1.38MB biến mất), c3_hw 0.232→0.201,
    **forward warm 0.82→0.79s (FPS 1.27)**.
- Bit: bd_wrapper_rev169b_200_lag.bit (board+local). Lag SUY TRONG HW (không reg 0x1F0).
- BEST CMD latency: RESACT=2 C3FAST=1 DWASYNC=1 HEADQ=1 UNI_NAME=ns12c → 0.79s.
- FPS timeline latency: 0.91(HEADQ)→0.85(Rev167 C3FAST)→0.82(Rev168 resident)→0.79(Rev169 in-place).

## 2026-07-08 khuya — INTERRUPT-WAIT (INTWAIT=1) ✅ BOARD-VERIFIED (host-only)
- Hạ tầng đã có sẵn: uio4="fabric" @0xA0000000; irq_out→pl_ps_irq0→GIC61 (bd.bd nối sẵn).
  IP HLS chuẩn: irq=gie&ier&isr, isr set @ap_done, W1C @0x030.
- Host: mở /dev/uio4, GIE/IER=1; kick() = clear ISR + os.write(1) arm + ap_start;
  wait_done() = select+os.read (NHẢ GIL) + clear ISR. Gate INTWAIT=1 (mặc định busy-poll).
  Loại trừ STPROF (cần sample state khi compute).
- ĐO board (parking106, RESACT=2 ns12c warm):
  - INTWAIT đơn (DW-HW): 0.83s — CHẬM hơn chút (2 syscall/kick × 62 kick) vì không có thread
    CPU tận dụng GIL rảnh.
  - INTWAIT + DWASYNC: **0.77s** (vs DWASYNC busy-poll 0.79s), 106 det. Lợi ích khiêm tốn vì
    DWASYNC busy-poll ĐÃ overlap phần lớn DW (numpy nhả GIL). INTWAIT nhặt thêm phần Python-level.
- KẾT LUẬN: interrupt-wait là ENABLER (nhả GIL trong HW wait) — đã chạy đúng, bit-exact. Muốn
  ăn to (→~0.6s) phải THREAD thêm deint/pack/graph để chồng lên kick layer kế (follow-on).
- BEST CMD latency: INTWAIT=1 DWASYNC=1 RESACT=2 C3FAST=1 HEADQ=1 ns12c → 0.77s (FPS 1.30).

## 2026-07-08 khuya — DTASYNC (thread hóa de-tile) = NET ÂM, BỎ (giữ code gated-off)
- Cài _DetileFuture (lazy de-tile, dma_read copy an toàn khỏi kick kế, __getattr__/__getitem__
  auto-join nên thay ndarray trong suốt: upsample.repeat/dequant.astype/HEADQ.astype tự join).
- ĐO board (full stack DTASYNC+INTWAIT+DWASYNC): **0.79s** vs INTWAIT+DWASYNC **0.77s** → CHẬM HƠN.
- LÝ DO (xác nhận phân tích dependency): mạng tuyến tính → de-tile(N) nằm TRÊN critical path
  (consumer N+1 join ngay), hiếm khi chồng được kick nhánh độc lập. Thread-overhead (46 layer ×
  tạo thread) + GIL-contention (de-tile thread giành GIL với pack/graph main thread) > overlap thu.
- KẾT LUẬN: DWASYNC (thread ĐÚNG 1 nhánh độc lập/block = DW) là mức threading tối ưu; mở rộng
  sang de-tile là quá đà. Follow-on "thread host stages" KHÔNG đáng — dependency chặn.
- BEST latency GIỮ NGUYÊN: INTWAIT=1 DWASYNC=1 RESACT=2 C3FAST=1 HEADQ=1 ns12c → 0.77s (FPS 1.30).
- DTASYNC=1 để lại (mặc định tắt) như đường-chết có tài liệu; đừng bật.

## 2026-07-09 — 2 đòn host "rẻ" đều FIZZLE (đo board, thành thật ghi)
- **Đa nhân numpy = ĐƯỜNG CHẾT**: OPENBLAS_NUM_THREADS 1/4/unset đều forward 0.77s. Lý do:
  host bottleneck là op ELEMENT-WISE (quantize/rescale/cat/maxpool = numpy ufunc, single-thread
  bất kể BLAS), KHÔNG phải matmul BLAS. OpenBLAS threading chỉ giúp matmul (chỉ L62 FP32 3ms).
  numpy dùng openblas64 NEON từ ~ubuntu/.local (script tự insert sys.path dòng 15-17).
- **HEADQ float-roundtrip opt = BIT-EXACT nhưng lợi ÍT (trong sai số)**: _QInt8 subclass mang
  qscale, quantize_to_int8 rescale int→int (round((int8×qscale)/scale) đúng thứ tự = bit-exact).
  106 det giữ, boxes khớp. Nhưng forward 0.77→0.77 (mảng float dequant không đắt như tưởng).
  GIỮ code (sạch hơn, đọc int8 1/4 băng thông) nhưng không có FPS gain. Deploy board.
- KẾT LUẬN: host đã cạn thật. Resident-act mở rộng chỉ còn 4 cặp HW→HW 1-consumer (L05→L09,
  L26→L30, L31→L32, L35→L39) — tất cả Cout>48 (cần multi-ocb scatter), ước ~20-30ms tổng.
  Đa số c1 layer feed graph op (cat/resid) nên deint BẮT BUỘC ở CPU → 197ms phần lớn bất khả giảm.

## 2026-07-09 — XÓA D:\yolov5-fpga\weights (dữ liệu N_config CŨ, gây nhầm lẫn)
- weights/L*_N_config.txt (127 file, tạo 2026-06-03) là snapshot DominoSearch SỚM/THÔ hơn —
  56/63 lớp = N=3 (trông "đồng đều"), KHÔNG khớp model đang deploy: so với layer_params.json
  thật (weights_unified_c3_ns12c) lệch 35/46 lớp. Nguồn N THẬT = yolov5/runs/sens/layerwise_nm.json
  (2026-07-03, DominoSearch pool [16,8,6,4,3,2,1], non-uniform thật, dùng để pack model board).
- Đã kiểm tra không script hiện hành nào phụ thuộc weights/ (chỉ 3 debug tool cùng ngày 06-03:
  sanity_check.py/verify_sparse_golden.py/inspect_coo.py — cũng thuộc thế hệ cũ, không dùng
  trong pipeline hiện tại domino_search.py→gpu_sparse_train_c3.py→make_unified_weights.py).
- User xác nhận xóa. Nguồn tham chiếu N-per-layer đúng từ nay: runs/sens/layerwise_nm.json.

## 2026-07-09 — Dọn board + yolov5/: xóa bitstream cũ, script chết, phát hiện N-config lệch
### Board (~/rtl_bd): xóa 23 bitstream cũ (chỉ giữ rev169b), c3dbg*/nsdbg*/bench_*.py (debug xong),
  2 ảnh trùng md5 (parking_ref.jpg, test_input.jpg == parking106.jpg), .bak_hostopt, __pycache__.
### D:\yolov5-fpga\weights XÓA (N_config cũ 06-03, lệch 35/46 lớp so với model đang deploy — xem
  entry trước). Nguồn N thật = yolov5/runs/sens/layerwise_nm.json (07-03).
### D:\yolov5-fpga\yolov5: xóa __pycache__, README.zh-CN.md, 4 log train cũ (~8.3MB),
  gen_spconv1x1_l26.py (PoC module spconv1x1_top.v KHÔNG có trong RTL, chưa từng build),
  gpu_sparse_train.py (thay bởi gpu_sparse_train_c3.py — chỉ bị nhắc trong comment/docstring
  của densify_train.py + gpu_sparse_train_lw.py, KHÔNG phải import Python thật, xóa an toàn).
  User chọn GIỮ eval_*/sens_scan/densify/gate_nm/... (giá trị phân tích/ablation cho luận văn).
### ⚠️ PHÁT HIỆN CHƯA GIẢI QUYẾT: verify N_CFG deploy (ns12c) vs layerwise_nm.json target —
  38/47 khớp; 3 lệch giải thích được (L42/49/56 HEADQ qua PTQ riêng, đúng thiết kế); **5 lệch
  CHƯA rõ nguyên nhân**: L20,L22,L27,L29,L36 (model.7.m.0.*, model.9.m.0.*, model.14.m.0.cv1) —
  target N=6 nhưng weight thật trong ns12c chỉ đạt N=3 (packer đo ground-truth từ weight, xem
  make_unified_weights.py:182 — không phải bug đóng gói, mà weight/mask thật sự sparse hơn dự kiến).
  Nghi: gpu_sparse_train_c3.py áp mask sai cho 5 lớp này, hoặc checkpoint pack bị cũ hơn scheme
  mới nhất. CHƯA điều tra sâu/sửa — cần quyết định có ảnh hưởng mAP đáng kể không trước khi retrain.

## 2026-07-09 — BUG FIX: HEADQ + maxlayers<46 gây sai detection (106→129), ĐÃ SỬA + VERIFY
- ROOT CAUSE: inference_unified_v4.py nhánh CPU fallback (dòng ~933, khi use_c1x1_hw=False vì
  n_hw[0]>=maxlayers) gọi quant_conv() trả int8 THÔ, KHÔNG bọc _QInt8(out, qscale) như nhánh HW.
  Khi lớp đó là L42/49/56 (HEADQ=1) và rơi vào CPU fallback (do maxlayers giới hạn + call-order
  trong run_graph khiến n_hw[0] vượt maxlayers trước khi tới các lớp này), quantize_to_int8()
  downstream đọc getattr(x,'qscale',None)=None → hiểu nhầm int8 thô là float → chia trực tiếp
  cho output_scale (số nhỏ ~0.01-0.05) → giá trị RÁC tràn ra ngoài [-128,127] trước clip →
  feed sai vào L60/L61 (detect head) → sai lệch số detection (106→129, phát hiện qua sweep
  maxlayers thử nghiệm giả thuyết "offload it hon co nhanh hon khong" — 129 chinh la trieu chung).
- KHÔNG xảy ra ở cấu hình BEST/mặc định (maxlayers=999 hoặc =46=tổng số lớp c1 → n_hw[0] không
  bao giờ vượt maxlayers → L42/49/56 luôn đi HW → không chạm nhánh lỗi). Chỉ lộ khi ép maxlayers
  thấp hơn tổng — nhưng vẫn là bug thật cần vá vì --maxlayers là API công khai của script.
- FIX: thêm `if _headq_deq: out = _QInt8(out, np.float32(m['output_scale']))` vào SAU nhánh CPU
  fallback (else quant_conv), giống hệt nhánh HW. VERIFY board: sweep maxlayers 0,5,10,20,30,46
  với HEADQ=1 → TOÀN BỘ 106 phát hiện (trước fix: 129 ở mọi maxlayers<46). Deploy board xong.
