// layer_sequencer.v -- B0 on-chip layer loop (removes the Python per-kick host loop).
// On 1 ap_start (LAYER_MODE=1) it sweeps an ENTIRE conv layer: for every out-ch-block (oc),
// position (pos = Hout*ceil(Wout/16)), and ic-chunk (ic) it DMAs the WT blob + ACT tile from
// DDR into the kernel BRAMs, pulses the kernel's per-tile interface (ap_start/keep/no_flush/
// dma_in_done), waits the kernel done, and (after the last ic-chunk) DMAs the OUT tile back.
// It OWNS the dma_engine while LAYER_MODE=1 (muxed against dma_sequencer in cnn_acc_top).
//
// DDR tile-stream layout (host bakes 1x/layer, see conv_layer_B0):
//   ACT : act_base + (pos*N_ICB + ic)*CHUNK_BYTES   (1x1 packed tiles) | raw CHW plane (c3)
//   WT  : wt_base  + (oc*N_ICB + ic)*WT_BLOCK_BYTES (blob incl requant tail)
//   OUT : out_base + (oc*N_POS + pos)*OUT_TILE      (host de-interleaves 1x/layer)
//
// B-c3 9-pass (Rev139): conv3x3 = 9 tap-shifted conv1x1 passes. The ic loop runs 9*NICC kicks
// (tap-major, icc-minor — host packs WT chunks in that order); per kick this sequencer supplies
// the tap-shifted PIXEL BASE via act_uram_off (act_uram holds the raw CHW plane, loaded by
// LS_ACTL) plus pad masks. All incremental adders — no divides/multiplies in the loop.
// c3_cfg (0x170): [7:0]=W_in [11:8]=ncolg [16:12]=nicc(5b: Cin/16 per tap, max 16 @Cin=256)
//                 [24:17]=nrow(H_out) [25]=stride2 [26]=mask_bot [27]=mask_right [31:28]=last_pos
// LS_ACTL (Rev139): single-counter linear copy — act_total (0x180) chunks of act_chunk_bytes.
// Host MUST write ACT_TOTAL whenever B-i or c3 is enabled (1x1: npos*nicb; c3: ceil(plane/actc)).
//
// Kernel handshake mirrors dma_sequencer per tile: assert k_dma_in_done (held) after ACT+WT
// land, the kernel runs, asserts k_ap_done; we then drop it.
`timescale 1ns/1ps

module layer_sequencer (
    input  wire        aclk,
    input  wire        aresetn,

    input  wire        layer_mode,     // 1 = this sequencer drives the kernel + dma_engine
    input  wire        dw_mode,        // depthwise: ACT differs per oc-block (4 chans) → idx by (oc,pos)
    input  wire        feat_en_bi,     // B-i: ACT-stationary — load whole-layer ACT once into act_uram
    input  wire        feat_en_c3,     // B-c3: conv3x3 9-pass (implies ACT plane resident in act_uram)
    input  wire        feat_en_ot,     // Đòn 4b OT: OUT DDR pos-major (khớp byte-for-byte ACT chunk layer kế)
    input  wire        ap_start,       // SW ap_start (shared); acted on only when layer_mode=1
    input  wire        k_ap_done,      // kernel ST_DONE pulse (per tile)

    // config (latched at ap_start)
    input  wire [7:0]  n_ocb,          // ceil(Cout/OCK)
    input  wire [15:0] n_pos,          // Hout*ceil(Wout/16)
    input  wire [7:0]  n_icb,          // ic-kicks per (oc,pos): 1x1 = ceil(Cin/16); c3 = 9*NICC
    input  wire [23:0] wt_block_bytes, // per (oc,ic) WT blob bytes (uniform, padded)
    input  wire [23:0] act_chunk_bytes,// per-tile ACT bytes/stride (256=conv1x1; c3: LS_ACTL chunk)
    input  wire [23:0] out_tile_bytes, // B-w: per-kick OUT bytes = OC_PER_KICK*16
    input  wire [31:0] c3_cfg,         // Rev139: c3 9-pass config (see header)
    input  wire [15:0] act_total,      // Rev139: LS_ACTL chunk count (l_nact)
    input  wire [20:0] c3_slice,       // Rev139 fix: 16*H*W bytes — icc-offset trong plane (REG_STRIDE)
    input  wire [63:0] act_base,
    input  wire [63:0] wt_base,
    input  wire [63:0] out_base,

    // dma_engine handshake (muxed owner in cnn_acc_top)
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,    // 0=ACT 1=WT 2=OUT
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    // kernel per-tile interface (muxed in unified_acc_top)
    output reg         k_ap_start,
    output reg         k_dma_in_done,
    output reg         k_keep_psum,
    output reg         k_no_flush,

    // Rev139 c3 per-kick pad masks + per-layer stride (→ unified_acc_top ACT capture)
    output reg         c3_zero,        // this kick's tap row is out of range → capture zeros
    output reg         c3_maskl,       // pos 0 out of range (left pad)
    output reg         c3_maskr,       // last valid col's read out of range (right pad)
    output wire [3:0]  c3_lastp,       // (W_out-1) & 15
    output wire        c3_stride2,

    output reg         irq,
    output wire        ls_busy,        // 1 while looping (→ host ap_idle = ~ls_busy)
    output reg         ls_done_latch,  // set at LS_DONE, cleared on ap_start (→ host ap_done)
    output wire [17:0] wt_uram_off,    // B2: current chunk's WT-URAM offset (Rev139: 18b, 256KB)
    output wire [20:0] act_uram_off,   // B-i/c3: ACT-URAM byte offset (Rev139: 21b, 1.5MB)
    output wire [3:0]  dbg_ls_state,
    // ── DEINT-ON-HW (feat bit12): scatter per-kick + DMA per-ob CHW ────────────
    input  wire        deint_hw,
    input  wire        deint_scat_done,
    output wire [15:0] deint_pos,
    output wire [15:0] deint_npos,
    output wire [7:0]  deint_ock,
    output reg         deint_scat_start,
    output reg         deint_dma_active,
    // ── Rev169 IN-PLACE LAG SCATTER: fill → lag_buf, drain (trễ D pos-word) → act_uram ──
    input  wire        lag_en,         // Rev169: RESOUT & ~INBANK & layer_mode (D=2*ncolg tự tính)
    output reg         drain_start,    // pulse: drain slot dpos[4:0] → act_uram @ base=dpos
    input  wire        drain_done,
    output wire [15:0] drain_pos       // dpos hiện hành (base scatter cho engine drain)
);
    localparam TGT_ACT = 2'd0, TGT_WT = 2'd1, TGT_OUT = 2'd2;
    // CHUNK_BYTES rationale (1024 not 864): every 256B burst 256-aligned, never crosses 4KB.
    localparam CHUNK_BYTES = 24'd1024, OUT_TILE = 24'd64;

    // B2 WT-stationary: LS_WTU/WTUW load the whole oc-block's WT into URAM (loop load_ic),
    // then the pos×ic kick loop reads WT from URAM (no per-kick WT DMA).
    localparam LS_IDLE=4'd0, LS_WTU=4'd1, LS_WTUW=4'd2, LS_ACT=4'd3, LS_ACTW=4'd4,
               LS_KICK=4'd5, LS_RUN=4'd6, LS_OUT=4'd7, LS_OUTW=4'd8, LS_DONE=4'd9,
               LS_ACTL=4'd10, LS_ACTLW=4'd11,   // B-i/c3: load whole-layer ACT into act_uram once
               LS_DSCATW=4'd12,                  // DEINT/RESIDENT: chờ fill scatter xong
               // Rev169: LS_DOBDMA (4'd13, deint legacy chết Rev165) tái chế cho lag drain
               LS_DRAINW=4'd13, LS_ENDDR=4'd14, LS_ENDDRW=4'd15;
    reg [3:0] ls;
    reg [7:0] load_ic;   // WT-URAM load chunk counter (0..N_ICB-1)
    assign dbg_ls_state = ls;
    assign ls_busy = (ls != LS_IDLE);

    // loop counters + latched config
    reg [7:0]  oc;   reg [15:0] pos;  reg [7:0] ic;
    reg [15:0] la_pos;                   // Rev139: single LS_ACTL chunk counter (0..l_nact-1)
    reg [7:0]  l_nocb; reg [15:0] l_npos; reg [7:0] l_nicb;
    reg [23:0] l_wbb; reg [23:0] l_actc; reg [23:0] l_outt; reg [63:0] l_actb, l_wtb, l_outb;
    reg [15:0] l_nact;
    reg        ap_seen;
    // ── DEINT-ON-HW ──
    reg [63:0] ob_base_r;      // ob CHW block base = out_base + oc*npos*OUT_TILE (latch @pos==0)
    reg [23:0] l_obbytes;      // ob CHW block bytes = OUT_TILE * n_pos
    assign deint_pos  = pos;
    assign deint_npos = l_npos;
    assign deint_ock  = l_outt[11:4];   // OCK = out_tile_bytes / 16
    // ── Rev169 lag ──
    reg        l_lag;                   // in-place lag mode (latch lag_cfg[7])
    reg [5:0]  l_lagd;                  // lag D pos-word (latch lag_cfg[5:0])
    reg [15:0] dpos;                    // pos kế tiếp cần drain (fill pos - D, đuổi 1:1)
    assign drain_pos = dpos;

    // ── Rev139 c3 latched config + incremental counters ──────────────────────
    reg [8:0]  l_c3w;      // W_in 9b (Rev147: L00 W=416)
    reg [3:0]  l_c3ncolg;  // ceil(W_out/16)
    reg [4:0]  l_c3nicc;   // Cin/16 per tap (5b: max 16 @Cin=256; icc rollover → next tap)
    reg [7:0]  l_c3nrow;   // H_out
    reg        l_c3s2;     // stride-2
    reg        l_c3mb;     // mask_bot: last row's kh=2 read out of range
    reg        l_c3mr;     // mask_right: last valid col's kw=2 read out of range
    reg [3:0]  l_c3lastp;  // (W_out-1) & 15
    assign c3_stride2 = l_c3s2;
    assign c3_lastp   = l_c3lastp;

    reg [7:0]  c3_r;       // output row       (tracks pos)
    reg [3:0]  c3_colg;    // output col group (tracks pos)
    reg [1:0]  c3_kw, c3_kh; // kernel tap      (tracks ic)
    reg [4:0]  c3_icc;     // ic-chunk within tap (tracks ic)
    reg [20:0] c3_rowpix;  // r*s*W_in
    reg [20:0] c3_colpix;  // colg*16*s
    reg [20:0] c3_tapoff;  // (kh-1)*W_in + (kw-1)  (wraps negative — masked kicks only)
    reg [20:0] c3_iccpix;  // icc*16*HW — kênh-slice offset trong plane (BUG FIX board-caught: thiếu → mọi icc đọc 16 kênh đầu)
    wire [20:0] c3_sw   = l_c3s2 ? {11'd0, l_c3w, 1'b0} : {12'd0, l_c3w};   // s*W_in
    wire [20:0] c3_s16  = l_c3s2 ? 21'd32 : 21'd16;                          // s*16
    wire [20:0] c3_tap0 = 21'd0 - {12'd0, l_c3w} - 21'd1;                    // -(W_in+1)

    // c3 active only for conv3x3 9-pass (non-dw). ACT resident in act_uram for B-i OR c3.
    // Rev139d: B-i mở rộng cho DW (bỏ ~dw_mode — act_idx dw formula hoạt động y hệt).
    wire c3_active  = feat_en_c3 & ~dw_mode;
    wire bi_active  = feat_en_bi;
    wire res_active = bi_active | c3_active;

    // WT-URAM offset: load phase uses load_ic, kick phase uses ic. off = chunk * WT_BLK.
    wire load_phase = (ls==LS_WTU) || (ls==LS_WTUW);

    // address index products — Rev139c: PIPELINED 2 tầng (path fail @150: l_nicb→DSP mult
    // →64b add→dma_req_addr = 7.6ns combinational). Các issue-state giờ chờ prep==2 rồi mới
    // phát dma_req nên product luôn kịp settle (idx_r edge+1, prod_r edge+2, issue edge+3).
    wire [31:0] wtl_idx = ({16'd0, oc} * {24'd0, l_nicb}) + {24'd0, load_ic}; // oc*N_ICB+load_ic
    // conv: ACT shared across oc-blocks → idx (pos,ic). dwconv: ACT = the oc-block's 4 channels
    // (depthwise, N_ICB=1) → idx (oc,pos), same layout as OUT (oc*N_POS+pos).
    wire [31:0] act_idx = dw_mode ? (({16'd0, oc} * {16'd0, l_npos}) + {16'd0, pos})
                                  : (({16'd0, pos} * {24'd0, l_nicb}) + {24'd0, ic});
    // Đòn 4b OT (feat bit12): out_idx pos-major → OUT DDR = đúng format ACT chunks layer kế
    // (768B tile (48oc,pos) = 3 chunk (16ic,16pos) liên tiếp; byte trong tile = oc*16+pos
    //  ≡ ic*16+pos của chunk — khớp tuyệt đối, không cần host pack/deint giữa 2 conv1x1).
    wire [31:0] out_idx = (feat_en_ot && !deint_hw) ? (({16'd0, pos} * {24'd0, l_nocb}) + {24'd0, oc})
                                     : (({16'd0, oc} * {16'd0, l_npos}) + {16'd0, pos});
    reg  [31:0] wtd_idx_r;  reg [55:0] wtd_b_r;    // WT DDR product pipe
    reg  [31:0] outd_idx_r; reg [55:0] outd_b_r;   // OUT DDR product pipe
    reg  [1:0]  prep;   // issue-state settle counter (bất biến: =0 khi vào issue-state —
                        // chỉ tăng trong issue-state, luôn về 0 khi phát req)

    // ── ACT/WT-URAM byte offsets — PIPELINED 2 tầng (timing fix @150MHz) ─────
    // Path fail Rev139 lần 1: chuỗi DSP (idx mult × l_actc) đổ COMBINATIONAL vào 7 tầng
    // cascade addr của 48 URAM (8.7ns). Offset luôn ổn định ≥3 cycle trước khi dùng
    // (dma_req registered → arbitration → beat đầu; k_ap_start → ST_WAIT_DMA → read đầu)
    // → pipeline 2 tầng: stage1 = index, stage2 = product (DSP hấp thụ reg nội bộ).
    // load phase (LS_ACTL): linear chunk copy la_pos*l_actc.
    // kick phase: 1x1 B-i = act_idx*l_actc (packed tiles); c3 = tap-shifted pixel base.
    wire        act_load_phase = (ls==LS_ACTL) || (ls==LS_ACTLW);
    wire [31:0] act_uoff_idx = act_load_phase ? {16'd0, la_pos} : act_idx;
    reg  [31:0] uoff_idx_r;                       // stage 1: index
    reg  [51:0] uoff_b_r;                         // stage 2: index * l_actc (DSP piped)
    reg  [31:0] wt_idx_r;                         // stage 1: WT chunk index
    reg  [55:0] wt_off_r;                         // stage 2: index * l_wbb (DSP piped)
    reg  [20:0] c3_off_r;                         // c3 pixel base (registered adder)
    always @(posedge aclk) begin
        uoff_idx_r <= act_uoff_idx;
        uoff_b_r   <= {20'd0, uoff_idx_r} * {28'd0, l_actc};
        wt_idx_r   <= load_phase ? {24'd0, load_ic} : {24'd0, ic};
        wt_off_r   <= wt_idx_r * l_wbb;
        c3_off_r   <= c3_rowpix + c3_colpix + c3_tapoff + c3_iccpix;
        // Rev139c: DDR address products (WT/OUT); ACT DDR tái dùng uoff_b_r (cùng idx mux —
        // load phase = la_pos, kick phase = act_idx — đúng nghĩa cho cả LS_ACTL lẫn LS_ACT).
        wtd_idx_r  <= wtl_idx;
        wtd_b_r    <= {24'd0, wtd_idx_r} * {32'd0, l_wbb};
        outd_idx_r <= out_idx;
        outd_b_r   <= {24'd0, outd_idx_r} * {32'd0, l_outt};
    end
    assign wt_uram_off  = wt_off_r[17:0];
    assign act_uram_off = (c3_active && !act_load_phase) ? c3_off_r : uoff_b_r[20:0];

    always @(posedge aclk) begin
        if (!aresetn) begin
            ls<=LS_IDLE; dma_req<=0; dma_req_tgt<=0; dma_req_addr<=0; dma_req_bytes<=0;
            k_ap_start<=0; k_dma_in_done<=0; k_keep_psum<=0; k_no_flush<=0; irq<=0;
            ls_done_latch<=0; oc<=0; pos<=0; ic<=0; load_ic<=0; ap_seen<=0;
            la_pos<=0; l_nact<=0; prep<=2'd0;
            deint_scat_start<=1'b0; deint_dma_active<=1'b0; ob_base_r<=64'd0; l_obbytes<=24'd0;
            drain_start<=1'b0; l_lag<=1'b0; l_lagd<=6'd0; dpos<=16'd0;
            l_nocb<=0; l_npos<=0; l_nicb<=0; l_wbb<=0; l_actc<=0; l_outt<=0; l_actb<=0; l_wtb<=0; l_outb<=0;
            l_c3w<=0; l_c3ncolg<=0; l_c3nicc<=0; l_c3nrow<=0; l_c3s2<=0; l_c3mb<=0; l_c3mr<=0; l_c3lastp<=0;
            c3_r<=0; c3_colg<=0; c3_kw<=0; c3_kh<=0; c3_icc<=0;
            c3_rowpix<=0; c3_colpix<=0; c3_tapoff<=0; c3_iccpix<=0;
            c3_zero<=0; c3_maskl<=0; c3_maskr<=0;
        end else begin
            dma_req<=1'b0; k_ap_start<=1'b0; irq<=1'b0; deint_scat_start<=1'b0;  // 1-cyc pulses default low
            drain_start<=1'b0;
            case (ls)
                LS_IDLE: begin
                    k_dma_in_done<=0;
                    if (ap_start && layer_mode && !ap_seen) begin
                        ap_seen<=1'b1; ls_done_latch<=1'b0;   // clear done for this run
                        l_nocb<=n_ocb; l_npos<=n_pos; l_nicb<=n_icb; l_wbb<=wt_block_bytes;
                        l_actc<=act_chunk_bytes; l_outt<=out_tile_bytes;
                        l_obbytes<=(out_tile_bytes[15:0] * n_pos);   // DEINT: ob CHW block bytes
                        l_actb<=act_base; l_wtb<=wt_base; l_outb<=out_base;
                        l_nact<=act_total;
                        // Rev147: W_in 9b (L00 W=416 tràn 8b — board-caught within1 36.5%);
                        // ncolg dời [12:9]; nicc mã hóa nicc-1 4b [16:13] (tiết kiệm 1 bit).
                        l_c3w<=c3_cfg[8:0]; l_c3ncolg<=c3_cfg[12:9];
                        l_c3nicc<={1'b0,c3_cfg[16:13]} + 5'd1;
                        l_c3nrow<=c3_cfg[24:17]; l_c3s2<=c3_cfg[25];
                        l_c3mb<=c3_cfg[26]; l_c3mr<=c3_cfg[27]; l_c3lastp<=c3_cfg[31:28];
                        oc<=0; pos<=0; ic<=0; load_ic<=0; la_pos<=0;
                        dpos<=16'd0; l_lag<=lag_en; l_lagd<={c3_cfg[12:9],1'b0};  // Rev169 D=2*ncolg
                        c3_r<=0; c3_colg<=0; c3_kw<=0; c3_kh<=0; c3_icc<=0;
                        c3_rowpix<=0; c3_colpix<=0; c3_iccpix<=0;
                        // B-i/c3: load whole-layer ACT first (LS_ACTL) then the oc-loop reads it
                        // from act_uram. Non-resident / dw: straight to WT-stationary load (LS_WTU).
                        // Rev168: act_total==0 = plane ĐÃ resident trong act_uram (scatter từ layer
                        // trước) → skip ACTL, oc-loop đọc thẳng. Dùng act_total (input) — l_nact
                        // latch cùng cycle nên chưa kịp cập nhật. ⚠ c3_tapoff bình thường init ở
                        // LS_ACTLW (c3_tap0) — skip thì init thẳng từ c3_cfg[8:0] (= l_c3w đang latch).
                        if (res_active && act_total == 16'd0)
                            c3_tapoff <= 21'd0 - {12'd0, c3_cfg[8:0]} - 21'd1;
                        ls<=(res_active && act_total != 16'd0) ? LS_ACTL : LS_WTU;
                    end
                    if (!ap_start) ap_seen<=1'b0;
                end

                // ── B-i/c3: DMA the layer ACT into act_uram ONCE (linear, l_nact chunks) ──
                // Rev139c: chờ prep==2 cho product pipe settle rồi mới phát req (addr = base + prod_r).
                LS_ACTL: begin
                    if (prep == 2'd2) begin
                        prep<=2'd0;
                        dma_req<=1'b1; dma_req_tgt<=TGT_ACT;
                        dma_req_addr<=l_actb + {12'd0, uoff_b_r}; dma_req_bytes<=l_actc;
                        ls<=LS_ACTLW;
                    end else prep<=prep+2'd1;
                end
                LS_ACTLW: if (dma_done) begin
                    if (la_pos + 16'd1 < l_nact) begin la_pos<=la_pos+16'd1; ls<=LS_ACTL; end
                    else begin la_pos<=16'd0; c3_tapoff<=c3_tap0; ls<=LS_WTU; end   // ACT resident → oc-loop
                end

                // ── WT-STATIONARY load: DMA chunk load_ic of this oc-block into URAM at
                //    offset load_ic*WT_BLK (wt_uram_off); loop all N_ICB chunks (once/oc-block) ──
                LS_WTU: begin
                    if (prep == 2'd2) begin
                        prep<=2'd0;
                        dma_req<=1'b1; dma_req_tgt<=TGT_WT;
                        dma_req_addr<=l_wtb + {8'd0, wtd_b_r}; dma_req_bytes<=l_wbb;
                        ls<=LS_WTUW;
                    end else prep<=prep+2'd1;
                end
                LS_WTUW: if (dma_done) begin
                    if (load_ic != l_nicb-8'd1) begin load_ic<=load_ic+8'd1; ls<=LS_WTU; end
                    // B-i/c3: ACT already in act_uram → kick directly (skip per-tile LS_ACT DMA)
                    else begin load_ic<=8'd0; pos<=16'd0; ic<=8'd0; ls<=res_active?LS_KICK:LS_ACT; end
                end

                // ── DMA ACT tile for (pos,ic) into IN BRAM (issue once → wait done) ──
                LS_ACT: begin
                    if (prep == 2'd2) begin
                        prep<=2'd0;
                        dma_req<=1'b1; dma_req_tgt<=TGT_ACT;
                        dma_req_addr<=l_actb + {12'd0, uoff_b_r}; dma_req_bytes<=l_actc;
                        ls<=LS_ACTW;
                    end else prep<=prep+2'd1;
                end
                LS_ACTW: if (dma_done) ls<=LS_KICK;

                // ── kick kernel: pulse ap_start; kernel reads WT from URAM[ic*WT_BLK + addr] ──
                LS_KICK: begin
                    k_keep_psum<=(ic!=8'd0);
                    k_no_flush <=(ic!=l_nicb-8'd1);
                    // c3 masks for THIS kick (r, colg, tap) — stable during the kick
                    c3_zero <= c3_active & ((c3_kh==2'd0 && c3_r==8'd0)
                                          | (c3_kh==2'd2 && l_c3mb && c3_r==l_c3nrow-8'd1));
                    c3_maskl<= c3_active & (c3_colg==4'd0 && c3_kw==2'd0);
                    c3_maskr<= c3_active & (l_c3mr && c3_colg==l_c3ncolg-4'd1 && c3_kw==2'd2);
                    k_ap_start <=1'b1;
                    ls<=LS_RUN;
                end

                // ── hold dma_in_done; kernel drains+computes+stores; wait k_ap_done ──
                LS_RUN: begin
                    k_dma_in_done<=1'b1;          // held; kernel waits it in ST_WAIT_DMA
                    if (k_ap_done) begin
                        k_dma_in_done<=1'b0;
                        // next ic-chunk: resident modes read act_uram (skip LS_ACT); else DMA the tile
                        if (ic != l_nicb-8'd1) begin
                            ic<=ic+8'd1;
                            // c3: advance (icc → tap) counters alongside ic
                            if (c3_active) begin
                                if (c3_icc == l_c3nicc-5'd1) begin
                                    c3_icc <= 5'd0; c3_iccpix <= 21'd0;   // tap mới: về slice 0
                                    if (c3_kw == 2'd2) begin
                                        c3_kw <= 2'd0; c3_kh <= c3_kh + 2'd1;
                                        c3_tapoff <= c3_tapoff + {12'd0, l_c3w} - 21'd2;  // next kernel row
                                    end else begin
                                        c3_kw <= c3_kw + 2'd1;
                                        c3_tapoff <= c3_tapoff + 21'd1;
                                    end
                                end else begin
                                    c3_icc <= c3_icc + 5'd1;
                                    c3_iccpix <= c3_iccpix + c3_slice;    // slice kênh kế: +16*HW
                                end
                            end
                            ls<=res_active?LS_KICK:LS_ACT;
                        end else begin
                            ic<=8'd0;
                            // c3: reset tap counters for the next (pos) kick loop
                            c3_icc<=5'd0; c3_iccpix<=21'd0; c3_kw<=2'd0; c3_kh<=2'd0; c3_tapoff<=c3_tap0;
                            ls<=LS_OUT;
                        end
                    end
                end

                // ── DMA OUT tile for (oc,pos) → DDR (issue once → wait done) ──
                LS_OUT: begin
                    if (deint_hw) begin
                        // DEINT: scatter kick này vào deint_bram (on-chip), KHÔNG DMA per-kick.
                        if (pos == 16'd0) ob_base_r <= l_outb + {8'd0, outd_b_r};  // ob CHW base @pos0
                        deint_scat_start <= 1'b1;
                        ls <= LS_DSCATW;
                    end else if (prep == 2'd2) begin
                        prep<=2'd0;
                        dma_req<=1'b1; dma_req_tgt<=TGT_OUT;
                        dma_req_addr<=l_outb + {8'd0, outd_b_r}; dma_req_bytes<=l_outt;
                        ls<=LS_OUTW;
                    end else prep<=prep+2'd1;
                end

                // ── Rev168/169: chờ fill scatter xong.
                //    lag=0 (Rev168): fill đã ghi thẳng act_uram → advance/hoàn tất.
                //    lag=1 (Rev169): fill mới vào lag_buf; nếu pos đủ xa (>= D) thì drain
                //    dpos (= pos - D) từ lag_buf → act_uram (in-place, đè hàng đã tiêu thụ).
                LS_DSCATW: if (deint_scat_done) begin
                    if (l_lag && pos >= {10'd0, l_lagd}) begin
                        drain_start <= 1'b1;
                        ls <= LS_DRAINW;
                    end else if (pos != l_npos-16'd1) begin
                        pos<=pos+16'd1;
                        // c3: advance (colg → row) counters GIỐNG LS_OUTW (deint hỗ trợ cả c3)
                        if (c3_active) begin
                            if (c3_colg == l_c3ncolg-4'd1) begin
                                c3_colg <= 4'd0; c3_colpix <= 21'd0;
                                c3_r <= c3_r + 8'd1; c3_rowpix <= c3_rowpix + c3_sw;
                            end else begin
                                c3_colg <= c3_colg + 4'd1; c3_colpix <= c3_colpix + c3_s16;
                            end
                        end
                        ls<=res_active?LS_KICK:LS_ACT;
                    end else if (l_lag && dpos != l_npos) begin
                        ls <= LS_ENDDR;               // hết kick nhưng lag_buf còn D slot
                    end else begin
                        // Rev168 RESIDENT hoàn tất (như nhánh else LS_OUTW). nocb=1 only.
                        pos<=16'd0; dpos<=16'd0;
                        c3_r<=8'd0; c3_colg<=4'd0; c3_rowpix<=21'd0; c3_colpix<=21'd0;
                        if (oc != l_nocb-8'd1) begin oc<=oc+8'd1; load_ic<=8'd0; ls<=LS_WTU; end
                        else ls<=LS_DONE;
                    end
                end
                // ── Rev169: chờ drain steady-state xong rồi advance pos (logic y hệt DSCATW) ──
                LS_DRAINW: if (drain_done) begin
                    dpos <= dpos + 16'd1;
                    if (pos != l_npos-16'd1) begin
                        pos<=pos+16'd1;
                        if (c3_active) begin
                            if (c3_colg == l_c3ncolg-4'd1) begin
                                c3_colg <= 4'd0; c3_colpix <= 21'd0;
                                c3_r <= c3_r + 8'd1; c3_rowpix <= c3_rowpix + c3_sw;
                            end else begin
                                c3_colg <= c3_colg + 4'd1; c3_colpix <= c3_colpix + c3_s16;
                            end
                        end
                        ls<=res_active?LS_KICK:LS_ACT;
                    end else if (dpos + 16'd1 != l_npos) begin
                        ls <= LS_ENDDR;
                    end else begin
                        pos<=16'd0; dpos<=16'd0;
                        c3_r<=8'd0; c3_colg<=4'd0; c3_rowpix<=21'd0; c3_colpix<=21'd0;
                        if (oc != l_nocb-8'd1) begin oc<=oc+8'd1; load_ic<=8'd0; ls<=LS_WTU; end
                        else ls<=LS_DONE;
                    end
                end
                // ── Rev169: xả nốt lag_buf sau kick cuối (D slot còn lại) ──
                LS_ENDDR:  begin drain_start<=1'b1; ls<=LS_ENDDRW; end
                LS_ENDDRW: if (drain_done) begin
                    dpos <= dpos + 16'd1;
                    if (dpos + 16'd1 == l_npos) begin
                        pos<=16'd0; dpos<=16'd0;
                        c3_r<=8'd0; c3_colg<=4'd0; c3_rowpix<=21'd0; c3_colpix<=21'd0;
                        if (oc != l_nocb-8'd1) begin oc<=oc+8'd1; load_ic<=8'd0; ls<=LS_WTU; end
                        else ls<=LS_DONE;
                    end else ls<=LS_ENDDR;
                end
                LS_OUTW: if (dma_done) begin
                    deint_dma_active<=1'b0;   // DEINT ob DMA xong (harmless khi non-deint)
                    // next pos: resident modes read act_uram (skip LS_ACT); else DMA the tile
                    if (pos != l_npos-16'd1) begin
                        pos<=pos+16'd1;
                        // c3: advance (colg → row) counters alongside pos
                        if (c3_active) begin
                            if (c3_colg == l_c3ncolg-4'd1) begin
                                c3_colg <= 4'd0; c3_colpix <= 21'd0;
                                c3_r <= c3_r + 8'd1; c3_rowpix <= c3_rowpix + c3_sw;
                            end else begin
                                c3_colg <= c3_colg + 4'd1; c3_colpix <= c3_colpix + c3_s16;
                            end
                        end
                        ls<=res_active?LS_KICK:LS_ACT;  // same WT in URAM
                    end else begin
                        pos<=16'd0;
                        c3_r<=8'd0; c3_colg<=4'd0; c3_rowpix<=21'd0; c3_colpix<=21'd0;
                        if (oc != l_nocb-8'd1) begin oc<=oc+8'd1; load_ic<=8'd0; ls<=LS_WTU; end
                        else ls<=LS_DONE;
                    end
                end

                LS_DONE: begin irq<=1'b1; ls_done_latch<=1'b1; ls<=LS_IDLE; end
                default: ls<=LS_IDLE;
            endcase
        end
    end
endmodule
