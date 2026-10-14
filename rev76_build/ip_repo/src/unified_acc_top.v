// unified_acc_top.v -- Unified accelerator: 1x1 + 3x3 + dwconv on shared 4 SPEC
// Replaces conv1x1_top + conv3x3_top + dwconv3x3_top (3 kernels → 1 module).
// Conv type selected by AXI_OFF_LAYER_TYPE register (set before ap_start).
//
// FSM branches on swc_en (from config_decoder):
//   swc_en=0 (1x1): ST_WAIT_DMA → ST_LOAD_WT → ST_LOAD_ACT → ST_SHUF_WAIT
//                 → ST_COMPUTE → ST_FLUSH_MAC → ST_WRITEBACK → ST_DONE
//   swc_en=1 (3x3/dw): ST_WAIT_DMA → ST_PIX_CLR → ST_ACT_LOAD → ST_WT_LOAD
//                    → ST_KW_SHUF → ST_COMPUTE → ST_FLUSH_MAC
//                    → [loop 9 kh/kw positions] → ST_FLUSH_RR → ST_DONE
//
// Resource: 256 DSP, 1 AXI slave, 1 BRAM set (vs 3x for 3-kernel design)
// AXI base address: 0xa0000000 (FROZEN)

`timescale 1ns/1ps
`include "cnn_defines.vh"

(* keep_hierarchy = "yes" *)
module unified_acc_top (
    input  wire        aclk,
    input  wire        aresetn,

    // ── AXI4-Lite config slave ────────────────────────────────────────────────
    input  wire [8:0]  s_awaddr,  input  wire        s_awvalid, output wire       s_awready,
    input  wire [31:0] s_wdata,   input  wire [3:0]  s_wstrb,
    input  wire        s_wvalid,  output wire        s_wready,
    output wire [1:0]  s_bresp,   output wire        s_bvalid,  input  wire       s_bready,
    input  wire [8:0]  s_araddr,  input  wire        s_arvalid, output wire       s_arready,
    output wire [31:0] s_rdata,   output wire [1:0]  s_rresp,
    output wire        s_rvalid,  input  wire        s_rready,

    // ── DMA interface (driven by parent cnn_acc_top) ──────────────────────────
    input  wire [9:0]  dma_in_addr,
    input  wire [7:0]  dma_in_data,
    input  wire        dma_in_wr_en,
    input  wire [5:0]  dma_in_w128_addr,   // wide ACT write (128-bit beat/cycle)
    input  wire [127:0] dma_in_w128_data,
    input  wire        dma_in_w128_en,
    input  wire        dma_in_done,
    input  wire        dma_in_active,  // dma_engine busy (R1 read-overtake-write guard)
    input  wire [11:0] dma_wt_addr,
    input  wire [7:0]  dma_wt_data,
    input  wire        dma_wt_wr_en,
    input  wire [12:0] dma_wt_w128_addr,   // wide WT write (128-bit beat → URAM word, within-chunk)
    input  wire [127:0] dma_wt_w128_data,
    input  wire        dma_wt_w128_en,
    input  wire [18:0] dma_out_addr,
    output wire [7:0]  dma_out_data,
    // Đòn 4a' OW (feat bit8): out DMA 128-bit/beat (đường dma_engine out_wide Rev130 có sẵn)
    output wire [127:0] dma_out_data128,
    output wire         dma_out_wide,

    output wire        irq_out,

    // ── Config register outputs (to parent cnn_acc_top for DMA calculation) ───
    output wire [31:0] reg_cin_out,
    output wire [31:0] reg_hin_out,
    output wire [31:0] reg_win_out,
    output wire [3:0]  reg_ncfg_out,
    output wire [23:0] reg_out_bytes_out,
    output wire [23:0] reg_wt_bytes_out,
    output wire [63:0] reg_in_addr_out,
    output wire [63:0] reg_wt_addr_out,
    output wire [63:0] reg_out_addr_out,
    output wire [31:0] reg_cout_out,      // systolic (LAYER_TYPE=3): expose Cout
    output wire [1:0]  reg_conv_type_out, // systolic mode = (==3)
    input  wire        sys_busy_in,       // systolic_seq busy   (drives ap_idle when systolic)
    input  wire        sys_done_latch_in, // systolic done latch (drives ap_done when systolic)
    output wire        ap_start_out,
    output wire        ap_done_out,   // kernel done — direct to dma_sequencer

    // ── Layer-level pipeline (prefetch next tile DMA) ─────────────────────────
    output wire        dma_nxt_req_o,

    // ── Debug: sequencer state readable via AXI at 0x170 ─────────────────────
    input  wire [3:0]  dbg_seq_state_in,
    // ── Debug: kernel FSM state readable via AXI at 0x180 ────────────────────
    output wire [3:0]  dbg_kernel_state_out,

    // ── B0 layer_sequencer interface (active when reg_layer_mode=1) ──────────
    output wire        reg_layer_mode_out,       // config → cnn_acc_top layer_sequencer
    output wire        dw_mode_out,              // depthwise → layer_sequencer ACT idx by (oc,pos)
    output wire [7:0]  reg_n_ocb_out,
    output wire [15:0] reg_n_pos_out,
    output wire [7:0]  reg_n_icb_out,
    output wire [23:0] reg_wt_blk_bytes_out,
    output wire [23:0] reg_act_chunk_out,
    output wire        k_done_pulse_out,         // kernel ST_DONE (1-cyc) → layer_sequencer
    output wire        lt_graphop_out,           // Rev172: conv_type==4 → LS passthrough (l_gop)
    input  wire        ls_k_ap_start,            // per-tile kick from layer_sequencer
    input  wire        ls_k_dma_in_done,
    input  wire        ls_k_keep_psum,
    input  wire        ls_k_no_flush,
    input  wire        ls_busy,                  // layer loop running (→ host ap_idle)
    input  wire        ls_done_latch,            // layer complete (→ host ap_done)
    input  wire [17:0] ls_wt_uram_off,           // WT-URAM chunk offset (ic*WT_BLK; load+read) — Rev139: 18b (256KB)
    input  wire [20:0] ls_act_uram_off,          // B-i/c3: ACT-URAM byte offset — Rev139: 21b (1.5MB); c3 kick = pixel base
    // ── Đòn 4b netseq_v2 ─────────────────────────────────────────────────────
    input  wire        ns_wr_en_in,              // cổng ghi reg-file từ netseq (→ axi ns_wr)
    input  wire [8:0]  ns_wr_addr_in,
    input  wire [31:0] ns_wr_data_in,
    input  wire        ns_busy_in,               // trạng thái segment → ap_idle/done mux
    input  wire        ns_done_in,
    input  wire        ns_err_in,                // Rev145: debug đọc được tại 0x190[31]
    input  wire [3:0]  ns_dbg_in,                // Rev145: netseq FSM state → 0x190[30:27]
    output wire        reg_feat_en_ns_out,       // NS enable (feat bit10) → unified_cnn_top routing
    output wire        reg_feat_en_ot_out,       // OT enable (feat bit11) → layer_sequencer
    output wire        reg_feat_en_bi_out,       // B-i enable (REG_FEAT_EN bit0) → layer_sequencer feat_en_bi
    // ── B-c3 9-pass (Rev139) ─────────────────────────────────────────────────
    output wire        reg_feat_en_c3_out,       // c3 enable (REG_FEAT_EN bit1) → layer_sequencer feat_en_c3
    output wire [31:0] reg_c3_cfg_out,           // 0x170 → layer_sequencer c3 counters cfg
    output wire [15:0] reg_act_total_out,        // 0x180 → layer_sequencer LS_ACTL chunk count
    output wire        lag_en_out,               // Rev169: RESOUT & ~INBANK & layer_mode (in-place lag)
    output wire [20:0] reg_c3_slice_out,         // REG_STRIDE tái dụng cho c3: 16*H*W bytes (icc-slice)
    input  wire        ls_c3_zero,               // per-kick: tap row out of range → all ACT bytes = 0
    input  wire        ls_c3_maskl,              // per-kick: pos 0 out of range (left pad col)
    input  wire        ls_c3_maskr,              // per-kick: last valid col's read out of range (right pad)
    input  wire [3:0]  ls_c3_lastp,              // per-layer: last valid position = (W_out-1) & 15
    input  wire        ls_c3_stride2,            // per-layer: stride-2 (ACT addr step = 2)
    // ── DEINT-ON-HW (feat bit12=0x2000) — scatter out_bram→deint_bram, DMA đọc CHW ──
    output wire        reg_feat_en_deint_out,    // bit12 → layer_sequencer
    input  wire        deint_scat_start,         // LS: pulse mỗi kick (scatter 1 kick vào deint_bram)
    input  wire [15:0] deint_pos,                // LS: pos-tile base (offset trong deint_bram)
    input  wire [15:0] deint_npos,               // LS: N_POS (stride oc_local)
    input  wire [7:0]  deint_ock,                // LS: OC per kick
    input  wire [16:0] deint_tbase,              // Rev200: word base TILED scatter (pos*cpad)
    input  wire        deint_dma_active,         // LS: DMA đang đọc deint_bram (MUX dma_out_data128)
    output wire        deint_scat_done,          // → LS: scatter (fill) xong
    // ── Rev169 IN-PLACE LAG: drain lag_buf slot dpos[4:0] → act_uram @ base=dpos ──
    input  wire        drain_start,              // LS: pulse per drain
    input  wire [15:0] drain_pos,                // LS: dpos
    output wire        drain_done                // → LS: drain xong
);

    // ── AXI slave (IP_TYPE=1: stride+pad enabled for all conv types) ──────────
    wire        ap_start_w;
    wire [31:0] reg_cin, reg_cout, reg_hin, reg_win;
    wire [4:0]  reg_n_config;   // Rev139: 5-bit (c3 dense = 16 entries/M-block)
    wire [1:0]  reg_poc_raw, reg_accu_mode_raw;
    wire        reg_apply_relu;
    wire [63:0] reg_wt_addr, reg_in_addr, reg_out_addr;
    wire [31:0] reg_stride, reg_pad;
    wire [3:0]  reg_conv_type;   // Rev170: 4-bit đầy đủ (trước chỉ lấy [1:0] từ axi_lite_slave
                                 // 4-bit output -> LAYER_TYPE 4-15 bị cắt, không phân biệt được)
    wire [23:0] reg_out_bytes;
    wire [23:0] reg_wt_bytes;
    wire        reg_layer_mode;
    wire [7:0]  reg_n_ocb;
    wire [15:0] reg_n_pos;
    wire [7:0]  reg_n_icb;
    wire [23:0] reg_wt_blk_bytes;
    wire [23:0] reg_act_chunk;
    wire [16:0] reg_feat_en;              // feature enable (REG_FEAT_EN = 0x190 bits[17:1])
                                          // [0]=B-i [1]=c3 [2]=fastkick [3]=C3W [4]=W11 wide-1x1
                                          // [5]=RR4 quad-writeback [6]=RQD rq-dirty-skip [7]=OW wide-out
                                          // [8]=WDW wide-feed DW [9]=FR4 quad-flush swc/DW
                                          // ZCU102 Rev200: [15]=SCATOUT [16]=RESINBASE (multi-region)
    wire [31:0] reg_c3_cfg;               // Rev139: c3 9-pass cfg (0x170)
    wire [15:0] reg_act_total;            // Rev139: LS_ACTL chunk count (0x180)
    assign reg_feat_en_bi_out = reg_feat_en[0];   // bit0 = B-i ACT-stationary → layer_sequencer
    assign reg_feat_en_ns_out = reg_feat_en[10] & reg_layer_mode;  // Đòn 4b NS mode
    assign reg_feat_en_ot_out = reg_feat_en[11];  // Đòn 4b OT: OUT pos-major
    assign reg_feat_en_deint_out = reg_feat_en[12]; // DEINT-ON-HW (0x2000) → layer_sequencer
    wire seqfast = reg_feat_en[13];  // Rev167 C3FAST (0x4000): 1x1-seq KHÔNG drain giữa entries
    wire act_in_bank = reg_feat_en[14]; // Rev168 RESIDENT (0x8000): input plane ở act_bram_in (BRAM)
    // ZCU102 Rev200 multi-region act_uram (graphop end-to-end). Cả hai GATED — bit tắt = y nguyên:
    //   SCATOUT  (0x10000): scatter đích += reg_out_addr[20:4] (word). Layer RESOUT không DMA-out
    //                       nên reg_out_addr rảnh → host ghi OFFSET act_uram (byte) vào đó.
    //   RESINBASE(0x20000): mọi đường đọc act_uram (byte/W11/C3W/DW) += reg_in_addr[20:0]. Layer
    //                       resident-in không ACT-DMA nên reg_in_addr rảnh → host ghi OFFSET byte.
    wire        scat_out_en   = reg_feat_en[15];
    wire        resin_base_en = reg_feat_en[16];
    wire [16:0] scat_base_w   = scat_out_en   ? reg_out_addr[20:4] : 17'd0;
    wire [20:0] resin_off     = resin_base_en ? reg_in_addr[20:0]  : 21'd0;
    // Rev174 TIMING: register 2 tổng base scatter (xem chú thích tại u_ocdeint)
    reg  [16:0] scat_tbase_r, scat_chw_r;
    always @(posedge aclk) begin
        scat_tbase_r <= deint_tbase + scat_base_w;
        scat_chw_r   <= {1'b0, deint_pos} + scat_base_w;
    end
    // Base hiệu dụng cho MỌI đường đọc act_uram tile-based (byte 1x1 / W11 / DW / C3W).
    // resin_off PHẢI chia hết 256 (host đảm bảo) → [20:4] của tổng vẫn đúng cho word path.
    // (c3_rd_addr KHÔNG offset — chuỗi c3 resident luôn ở vùng 0, giữ nguyên hành vi cũ.)
    // Rev174 TIMING: REGISTER tổng (ls_act_uram_off + resin_off) — gỡ adder 21b khỏi đường
    // ls_reg(state)→mux→adder→adder→URAM CAS_ADDR (WNS −0.157/−0.266 ở rev172/173).
    // AN TOÀN: đây là BASE per-tile, ổn định nhiều cycle trước lần đọc đầu của tile
    // (uoff pipe 2 tầng trong LS settle trước kick; engine đọc sau kick ≥2 cyc) →
    // độ trễ 1 nhịp của thanh ghi không bao giờ bị quan sát.
    reg  [20:0] ls_act_off_eff;
    always @(posedge aclk) ls_act_off_eff <= ls_act_uram_off + resin_off;
    assign reg_feat_en_c3_out = reg_feat_en[1];   // bit1 = B-c3 9-pass → layer_sequencer
    assign reg_c3_cfg_out     = reg_c3_cfg;
    assign reg_act_total_out  = reg_act_total;
    // Rev169: lag = layer resident-out (bit12) mà input KHÔNG ở bank (bit14=0 → input nằm
    // act_uram → phải in-place lag). D=2*ncolg do LS tự tính. Bỏ reg 0x1F0 (SLVERR trên IP cũ).
    // Rev174 ROOT-CAUSE Gate D: fix "lag = c3-only" (Rev173) chỉ vá lag_en_c NỘI BỘ mà SÓT
    // chỗ này — LS nhận lag_en=1 cho conv1x1 RESOUT → l_lag=1 (nhánh DRAINW không ob-reset)
    // + l_tiled=0 (cpad rơi về 48) = ĐÚNG layout H10 đo được trên board (fill ghi tiled base
    // vì lag_en_c nội bộ =0 chặn drain-writes → x6 nguyên vẹn, khớp probe). feat[1] = C3 flag.
    assign lag_en_out         = reg_feat_en[12] & ~reg_feat_en[14] & reg_layer_mode & reg_feat_en[1];
    assign reg_c3_slice_out   = reg_stride[20:0]; // c3: host ghi 16*H*W vào REG_STRIDE (0x120)

    assign reg_act_chunk_out    = reg_act_chunk;
    assign reg_layer_mode_out   = reg_layer_mode;
    assign reg_n_ocb_out        = reg_n_ocb;
    assign reg_n_pos_out        = reg_n_pos;
    assign reg_n_icb_out        = reg_n_icb;
    assign reg_wt_blk_bytes_out = reg_wt_blk_bytes;
    assign reg_cin_out        = reg_cin;
    assign reg_hin_out        = reg_hin;
    assign reg_win_out        = reg_win;
    assign reg_ncfg_out       = reg_n_config[3:0];  // mode-0 legacy DMA sizing only (ncfg≤15 there)
    assign reg_out_bytes_out  = reg_out_bytes;
    assign reg_wt_bytes_out   = reg_wt_bytes;
    assign reg_in_addr_out    = reg_in_addr;
    assign reg_wt_addr_out    = reg_wt_addr;
    assign reg_out_addr_out   = reg_out_addr;
    assign reg_cout_out       = reg_cout;
    assign reg_conv_type_out  = reg_conv_type;
    wire   systolic_mode_u    = (reg_conv_type == 2'd3);   // LAYER_TYPE=3 = systolic conv1x1
    reg ap_done_r, ap_idle_r;
    reg ap_done_latch;  // latched ap_done — SW-readable, cleared on next ap_start
    reg [3:0]  state;   // FSM state (hoisted up: used below for dbg_kernel_state assign)

    assign ap_start_out         = ap_start_w;
    assign ap_done_out          = ap_done_latch;  // level: HIGH until next ap_start
    assign dbg_kernel_state_out = state;
    // B0: host polls LAYER status (ls_busy/ls_done_latch) instead of per-tile kernel status
    // Đòn 4b: khi NS mode (feat bit10) → host thấy trạng thái SEGMENT (netseq) thay vì layer.
    wire ns_mode_u = reg_feat_en[10] & reg_layer_mode;
    // Rev170 GRAPHOP: forward-declare (instantiation + mux ở dưới, sau act_uram)
    wire graphop_mode = (reg_conv_type == 4'd4);
    reg  graphop_active, graphop_done_latch, graphop_start_r;
    wire graphop_done;
    wire ap_idle_axil = graphop_mode ? ~graphop_active
                      : ns_mode_u ? ~ns_busy_in
                      : systolic_mode_u ? ~sys_busy_in
                      : reg_layer_mode  ? ~ls_busy      : ap_idle_r;
    wire ap_done_axil = graphop_mode ? graphop_done_latch
                      : ns_mode_u ? ns_done_in
                      : systolic_mode_u ? sys_done_latch_in
                      : reg_layer_mode  ? ls_done_latch : ap_done_latch;

    axi_lite_slave #(.IP_TYPE(1)) u_axil (
        .aclk(aclk), .aresetn(aresetn),
        .awaddr(s_awaddr), .awvalid(s_awvalid), .awready(s_awready),
        .wdata(s_wdata),   .wstrb(s_wstrb),     .wvalid(s_wvalid),  .wready(s_wready),
        .bresp(s_bresp),   .bvalid(s_bvalid),   .bready(s_bready),
        .araddr(s_araddr), .arvalid(s_arvalid), .arready(s_arready),
        .rdata(s_rdata),   .rresp(s_rresp),     .rvalid(s_rvalid),  .rready(s_rready),
        .ap_done_in(ap_done_axil), .ap_idle_in(ap_idle_axil),
        .ns_wr_en(ns_wr_en_in), .ns_wr_addr(ns_wr_addr_in), .ns_wr_data(ns_wr_data_in),
        .ns_err_in(ns_err_in), .ns_dbg_in(ns_dbg_in),
        .dbg_seq_state_in(dbg_seq_state_in),
        .dbg_kernel_state_in(state),
        .reg_cin(reg_cin), .reg_cout(reg_cout), .reg_hin(reg_hin), .reg_win(reg_win),
        .reg_n_config(reg_n_config), .reg_poc(reg_poc_raw),
        .reg_accu_mode(reg_accu_mode_raw), .reg_apply_relu(reg_apply_relu),
        .reg_wt_ddr_addr(reg_wt_addr), .reg_in_ddr_addr(reg_in_addr),
        .reg_out_ddr_addr(reg_out_addr),
        .reg_stride(reg_stride), .reg_pad(reg_pad),
        .reg_conv_type(reg_conv_type),
        .reg_out_bytes(reg_out_bytes),
        .reg_wt_bytes(reg_wt_bytes),
        .reg_layer_mode(reg_layer_mode), .reg_feat_en(reg_feat_en),
        .reg_n_ocb(reg_n_ocb), .reg_n_pos(reg_n_pos),
        .reg_n_icb(reg_n_icb), .reg_wt_blk_bytes(reg_wt_blk_bytes), .reg_act_chunk(reg_act_chunk),
        .reg_c3_cfg(reg_c3_cfg), .reg_act_total(reg_act_total),
        .ap_start(ap_start_w), .irq_out(irq_out)
    );

    // ── Config decoder ────────────────────────────────────────────────────────
    wire        swc_en, use_halo, dw_mode;
    wire [1:0]  eff_poc, eff_accu_mode;
    assign dw_mode_out = dw_mode;       // → layer_sequencer (B0 dw ACT addressing by oc)

    config_decoder u_cfg (
        .reg_conv_type  (reg_conv_type),
        .reg_poc        (reg_poc_raw),
        .reg_accu_mode  (reg_accu_mode_raw),
        .reg_stride     (reg_stride[0]),
        .swc_en         (swc_en),
        .eff_poc        (eff_poc),
        .eff_accu_mode  (eff_accu_mode),
        .use_halo       (use_halo),
        .dw_mode        (dw_mode)
    );

    // ── Input BRAM (ping-pong): depth=1024 covers max tile (30x30) with halo ──
    wire [7:0]  in_bram_rd8_data;   // baseline: in_bram_wide rd8 output
    wire [7:0]  act_uram_rd_data;   // B-i/c3: act_uram byte-read output
    // B-i ACT-stationary: Rev139d mở rộng cho CẢ DW (act layout (oc,pos) tiles — LS_ACTL nạp
    // tuyến tính, kick đọc qua act_idx dw-formula; host chỉ cần ghi ACT_TOTAL = nocb*npos).
    wire        bi_act = reg_feat_en[0];
    // B-c3 9-pass (Rev139): conv3x3 as 9 tap-shifted conv1x1 passes over the raw CHW plane
    // resident in act_uram. Rides the whole verified 1x1 path; only the ACT address changes.
    wire        c3_act  = reg_feat_en[1] & ~dw_mode;
    wire        act_res = bi_act | c3_act;   // ACT resident in act_uram (either mode)
    // Kernel ACT read: from act_uram (B-i/c3) or in_bram_wide (baseline). All RD_LAT=3 (Rev139).
    wire [7:0]  in_bram_rd_data = act_res ? act_uram_rd_data : in_bram_rd8_data;
    reg  [9:0]  in_bram_rd_addr;
    reg         pp_in_sel;
    // ── Rev139d wide-ACT (C3W, feat bit3): đọc ACT c3 theo WORD (2-3 word/entry thay 16 byte)
    //    + barrel 16 lane từ cửa sổ 48B. Gate riêng để A/B với đường byte (feat bit3 off). ──
    wire         c3w = c3_act & reg_feat_en[3];
    // ── Đòn 4a W11 (feat bit4): 1x1 ACT load theo WORD — mỗi NZ entry là đúng 1 word 16B-aligned
    //    (addr = ls_act_uram_off + {icc,idx,4'b0}, idx CHUNG mọi SPEC) → 4 cycle thay 20.
    //    Chỉ khi B-i (ACT trong act_uram); addr REGISTER trước 1 cycle (w11_addr_r) để nhẹ URAM path.
    wire         w11 = reg_feat_en[4] & bi_act & ~c3_act & ~swc_en;
    reg  [16:0]  w11_addr_r;              // word addr, chốt tại cnt0, issue từ cnt1
    // ── Rev142 WDW (feat bit9): DW ACT load theo WORD — window 16B unaligned trong chunk
    //    (offset g*54+kh*18+kw) → 2 word + barrel 32B (tái dùng aw_base/aw_w0/w1 của C3W;
    //    DW không bao giờ ở ST_LOAD_ACT_W nên không tranh chấp). Cần DWBI (ACT trong act_uram).
    wire         wdw = reg_feat_en[8] & dw_mode & bi_act;
    reg  [2:0]   aw_cnt;
    reg  [20:0]  aw_base;                 // byte base của entry (ls_act_uram_off + c3_idxhw)
    reg  [127:0] aw_w0, aw_w1, aw_w2;     // cửa sổ 48B (3 word; w2 chỉ dùng stride2)
    wire [383:0] aw_win = {aw_w2, aw_w1, aw_w0};
    wire [127:0] act_w_data;              // act_uram word-read out (RD_LAT=2)
    wire         w11_ld;                  // W11 load active (assign dưới FSM — cần state)
    wire         awsel_active;            // act_uram word-port select (C3W | W11)
    // (Rev139: R1 fast path PRUNED — conv3x3 goes 9-pass on the SPEC array; frees 256 DSP for NS=12)

    // ── c3 ACT address: byte addr = pixel_base(kick) + idx*HW + pos*stride ──────
    // ls_act_uram_off = tap-shifted pixel base from layer_sequencer (plane at act_uram offset 0).
    // idx*HW: c3_idxhw = wt_rd_idx[slot] * c3_hw (Rev150 sparse-c3). Dense pack idx=slot
    // → slot*HW (bit-exact với accumulator cũ); sparse pack idx=kept-channel → đúng CHW base.
    // 1 multiply 4×18b, tính 1×/slot ở ST_FLUSH_MAC (không trong read path per-pixel).
    // B-w params (khai báo SỚM — Rev139c: ncfg_esz_r bên dưới cần ESZ):
    localparam NS  = `N_SPEC_ACT;          // active SPEC count (4 baseline / 12 Option C)
    localparam ESZ = 1 + NS * `G;          // WT entry bytes = idx + NS*G weights (49 @ NS=12)
    reg  [17:0] c3_hw;        // input plane H*W (max 208*208=43264); registered product, settles early
    reg  [20:0] c3_idxhw;     // idx * HW accumulator (reset per M-block load)
    reg  [20:0] c3_rd_addr;   // current ACT byte address (base + pos*stride)
    reg  [11:0] ncfg_esz_r;   // Rev139c: n_config*ESZ registered (bỏ chuỗi 2 mult khỏi addr paths)
    always @(posedge aclk) begin
        c3_hw      <= reg_hin[8:0] * reg_win[8:0];
        ncfg_esz_r <= reg_n_config * ESZ[7:0];
    end

    // IN BRAM: byte-write (DMA) + 8-bit read (mode-0). (R1 rd128 port removed Rev139.)
    in_bram_wide u_bram_in (
        .clk(aclk), .rst_n(aresetn),
        .pp_bank_sel(pp_in_sel),
        .dma_wr_addr(dma_in_addr), .dma_wr_data(dma_in_data), .dma_wr_en(dma_in_wr_en),
        // B-i/c3: when ACT resident, the ACT DMA writes act_uram instead → gate in_bram write off.
        .dma_w128_addr(dma_in_w128_addr), .dma_w128_data(dma_in_w128_data), .dma_w128_en(dma_in_w128_en & ~act_res),
        .rd8_addr(in_bram_rd_addr), .rd8_data(in_bram_rd8_data)
    );

    // ── B-i/c3: ACT-stationary URAM (1.5MB Rev139). LS_ACTL DMAs the whole layer's ACT here ONCE;
    //    1x1 kicks read tile (pos,ic) via ls_act_uram_off + in_bram_rd_addr; c3 kicks read the raw
    //    CHW plane via the engine-computed c3_rd_addr (pixel base + idx*HW + pos*stride). ──
    // Rev168: khai báo scatter (dời lên từ khối DEINT phía dưới — dùng trong mux act_uram)
    wire [7:0]   ocd_rd_oc;                 // oc_local oc_deint muốn đọc
    wire [16:0]  ocd_wr_addr; wire [127:0] ocd_wr_data; wire ocd_wr_en; // 17b (đích act_uram, plane tới 1.38MB = 86528 word)
    reg          deint_scat_active;         // high start..done
    always @(posedge aclk) begin
        if (!aresetn)                 deint_scat_active <= 1'b0;
        else if (deint_scat_start)    deint_scat_active <= 1'b1;
        else if (deint_scat_done)     deint_scat_active <= 1'b0;
    end
    // Rev168 RESIDENT-ACT: (1) oc_deint scatter → act_uram write port (out plane CHW staging @0
    //     = plane input layer kế; ACTL/scatter tách bạch thời gian nên mux theo deint_scat_active);
    //     (2) act_in_bank (feat[14]): input plane nằm ở act_bram_in (BRAM 512KB) — ACTL ghi bank,
    //     engine đọc bank → act_uram trống trọn 1.5MB nhận output plane (L00 out 1.38MB).
    // Rev169 LAG: lag_en_c=1 → fill (u_ocdeint) ghi lag_buf (KHÔNG act_uram); drain (u_ocdrain,
    // trễ D pos-word) mới ghi act_uram — in-place an toàn (kick hàng r đọc r-1..r+1, chỉ đè ≤ r-2).
    // Rev173 FIX: lag CHỈ dành cho c3 (in-place L01). Bản cũ (bit12 & ~bit14) khiến conv1x1
    // RESOUT-không-INBANK rơi nhầm vào LAG thay vì tiled scatter (gap bắt khi thiết kế Gate D).
    // c3 RESACT cũ giữ nguyên: L00 (12+14→CHW scatter), L01 (12,~14,c3→lag). conv1x1 cũ không dùng bit12.
    wire         lag_en_c = reg_feat_en[12] & ~reg_feat_en[14] & reg_layer_mode & c3_act; // Rev169 lag (Rev173: &c3)
    // Rev200: RESOUT trên conv1x1 (~c3, ~lag) scatter theo TILED position-major (chuẩn mới).
    // KV260 chưa từng dùng conv1x1+RESOUT → không đổi hành vi flow cũ nào.
    wire         scat_tiled = reg_feat_en[12] & ~c3_act & ~lag_en_c & reg_layer_mode;
    wire [7:0]   ocd2_rd_oc;
    wire [16:0]  ocd2_wr_addr; wire [127:0] ocd2_wr_data; wire ocd2_wr_en;
    reg          drain_active;               // high drain_start..drain_done
    always @(posedge aclk) begin
        if (!aresetn)         drain_active <= 1'b0;
        else if (drain_start) drain_active <= 1'b1;
        else if (drain_done)  drain_active <= 1'b0;
    end
    wire         scat_win      = deint_scat_active | drain_active;
    wire [16:0]  scat_wr_waddr = lag_en_c ? ocd2_wr_addr : ocd_wr_addr;
    wire [127:0] scat_wr_wdata = lag_en_c ? ocd2_wr_data : ocd_wr_data;
    wire         scat_wr_wen   = lag_en_c ? ocd2_wr_en   : ocd_wr_en;
    wire [16:0]  actl_wr_waddr = ls_act_uram_off[20:4] + {11'd0, dma_in_w128_addr}; // ACTL: tile base + word
    // ── Rev170 GRAPHOP: engine cat/resid_cat trên PL (LAYER_TYPE=4, chạy độc lập FSM chính) ──
    // Gate conv_type==4; FSM chính đứng yên ở ST_IDLE khi graphop_mode → không đụng đường cũ.
    // Register-map TÁI DÙNG (host ghi ý nghĩa khác khi conv_type=4, KHÔNG thêm reg mới):
    //   addr_a=in_addr addr_b=wt_addr addr_c=out_addr addr_dst=stride  (act_uram byte-off, 21b)
    //   n_pos=win[15:0] ch_ab=n_ocb ch_c=n_icb resid_mode=pad[0]
    //   {sh,m0} A=cin B=cout C=hin  ([20:16]=shift, [15:0]=M0)
    // (graphop_mode/active/done_latch/start_r/done: forward-declared ở trên, gần ap_done_axil)
    wire [20:0]  go_rd_addr;
    wire [16:0]  go_wr_waddr; wire [127:0] go_wr_wdata; wire go_wr_wen;
    always @(posedge aclk) begin
        if (!aresetn)             graphop_active <= 1'b0;
        else if (graphop_start_r) graphop_active <= 1'b1;
        else if (graphop_done)    graphop_active <= 1'b0;
    end
    graphop_engine u_graphop (
        .clk(aclk), .rst_n(aresetn),
        .start(graphop_start_r), .done(graphop_done),
        .resid_mode(reg_pad[0]),
        .addr_a(reg_in_addr[20:0]), .addr_b(reg_wt_addr[20:0]),
        .addr_c(reg_out_addr[20:0]), .addr_dst(reg_stride[20:0]),
        .n_pos(reg_win[15:0]), .ch_ab(reg_n_ocb), .ch_c(reg_n_icb),
        .m0_a(reg_cin[15:0]),  .sh_a(reg_cin[20:16]),
        .m0_b(reg_cout[15:0]), .sh_b(reg_cout[20:16]),
        .m0_c(reg_hin[15:0]),  .sh_c(reg_hin[20:16]),
        .act_rd_addr(go_rd_addr), .act_rd_data(act_uram_rd_data),
        .act_wr_waddr(go_wr_waddr), .act_wr_wdata(go_wr_wdata), .act_wr_wen(go_wr_wen)
    );

    wire [16:0]  actu_wr_waddr = graphop_active ? go_wr_waddr
                               : scat_win ? scat_wr_waddr : actl_wr_waddr;
    wire [127:0] actu_wr_wdata = graphop_active ? go_wr_wdata
                               : scat_win ? scat_wr_wdata : dma_in_w128_data;
    wire         actu_wr_wen   = graphop_active ? go_wr_wen
                               : scat_win ? scat_wr_wen
                               : (dma_in_w128_en & act_res & ~act_in_bank);
    wire [20:0]  act_rd_baddr  = graphop_active ? go_rd_addr
                               : c3_act ? c3_rd_addr : (ls_act_off_eff + {11'd0, in_bram_rd_addr});
    // Rev139d C3W + Đòn4a W11 word port (time-mux; active trong ST_LOAD_ACT_W / W11-load)
    wire [16:0]  act_rd_waddr  = w11_ld ? w11_addr_r : (aw_base[20:4] + {14'd0, aw_cnt} - 17'd1);
    wire [7:0]   act_rd_data_u,  act_rd_data_bk;
    wire [127:0] act_w_data_u,   act_w_data_bk;
    act_uram u_bram_act (
        .clk(aclk),
        .wr_waddr(actu_wr_waddr),
        .wr_wdata(actu_wr_wdata),
        .wr_wen(actu_wr_wen),
        .rd_baddr(act_rd_baddr),
        .rd_data(act_rd_data_u),
        .rd_wsel(awsel_active),
        .rd_waddr(act_rd_waddr),
        .rd_wdata(act_w_data_u)
    );
    act_bram_in u_bram_actin (
        .clk(aclk),
        .wr_waddr(actl_wr_waddr[14:0]),        // plane @0, <=512KB → 15b đủ
        .wr_wdata(dma_in_w128_data),
        .wr_wen(dma_in_w128_en & act_res & act_in_bank),
        .rd_baddr(act_rd_baddr[18:0]),
        .rd_data(act_rd_data_bk),
        .rd_wsel(awsel_active),
        .rd_waddr(act_rd_waddr[14:0]),
        .rd_wdata(act_w_data_bk)
    );
    // mux sau output-reg 2 tầng (select per-layer tĩnh → route thoải mái)
    assign act_uram_rd_data = act_in_bank ? act_rd_data_bk : act_rd_data_u;
    assign act_w_data       = act_in_bank ? act_w_data_bk  : act_w_data_u;

    // ── Weight URAM (B2: WT-stationary, 256KB Rev139) ─────────────────────────
    // Replaces the 4096-byte WT bram_ping_pong. Holds an entire oc-block's weights (all
    // N_ICB chunks) so the layer_sequencer DMAs WT ONCE per oc-block (LS_WTU) instead of
    // every kick. ls_wt_uram_off = current chunk's URAM offset (ic*WT_BLK), added to BOTH
    // the DMA write addr (during load) and the kernel read addr (during kick) — load and
    // read are different phases (drain guard), never concurrent. Mode-0/per-tile: off=0
    // (layer_sequencer idle), behaves like the old single-chunk WT BRAM at addr 0.
    // Rev139: + word-read mode (ST_WT_LOAD_W, c3 wide WT: 16B/cycle, 64B-aligned entries).
    reg  [11:0] wt_bram_rd_addr;
    wire [7:0]  wt_bram_rd_data;
    reg         pp_wt_sel;
    reg  [6:0]  wtw_cnt;        // c3 wide WT load word counter (0..n_cfg*4+2)
    reg         wtw_active;     // 1 = word-read mode drives the shared URAM read port
    wire [127:0] wt_w_data;
    // wide write: word addr = within-chunk word (dma_wt_w128_addr) + chunk base (off>>4).
    // WT_BLK is 16-aligned so off[3:0]=0 → off>>4 exact. read stays byte-addressed.
    wire [13:0] wt_uram_wr_waddr = {1'b0, dma_wt_w128_addr} + ls_wt_uram_off[17:4];
    wire [17:0] wt_uram_rd_baddr = {6'd0, wt_bram_rd_addr} + ls_wt_uram_off;
    wire [13:0] wt_uram_rd_waddr = ls_wt_uram_off[17:4] + {7'd0, wtw_cnt};
    wt_uram u_bram_wt (
        .clk(aclk),
        .wr_waddr(wt_uram_wr_waddr), .wr_wdata(dma_wt_w128_data), .wr_wen(dma_wt_w128_en),
        .rd_baddr(wt_uram_rd_baddr), .rd_data(wt_bram_rd_data),
        .rd_wsel(wtw_active), .rd_waddr(wt_uram_rd_waddr), .rd_wdata(wt_w_data)
    );

    // ── Output BRAM (ping-pong, depth=784) ────────────────────────────────────
    // wb_done_cnt/out_bram_cnt declared here (before use in assign below) to
    // avoid Vivado "used before declaration" warning which can cause wrong width.
    reg [9:0]  wb_done_cnt;   // B-w: widened [8:0]→[9:0] (NS=12 → NS*64=768 > 511)
    reg [7:0]  out_bram_cnt;

    // Rev36: registered OUT BRAM write pipeline (skew-free, see assign below)
    reg [9:0]  owr_ptr;     // write address = count of outputs written so far
    reg [7:0]  owr_data_r;  // registered rr_out (locked with owr_ptr timing)
    reg        owr_en_r;    // registered rr_valid (1-cycle delayed write enable)
    reg [23:0] owr_q_r;     // Đòn 4a' RR4: lane 1-3 (cùng tầng FF với owr_data_r)
    reg        owr_quad_r;  // 1 = quad write (4 lane), 0 = byte (legacy/swc)

    // Đòn 4a': OUT buffer word-hóa — ghi 32-bit/byte-lane (RR4 quad = 4 lane/cycle,
    // legacy byte = 1 lane), đọc byte (dma byte path, RD_LAT=1 giữ Rev37) + 128-bit
    // (dma out_wide path, cùng dma_out_addr — wide mode addr nhảy +16/beat).
    // LUT-RAM 49×128 (784B) — né hẳn họ bug RAM-generator (Rev58/HARTGXRamGen).
    wire [7:0]  out_bram_wr_waddr;   // 32-bit word address
    wire [31:0] out_bram_wr_data;
    wire [3:0]  out_bram_wr_we;      // byte lanes
    reg         pp_out_sel;          // giữ cho FSM flip logic cũ (không còn dùng — PINGPONG=0)

    // ── DEINT-ON-HW: deint_bram (staging CHW) + oc_deint (scatter) + rd MUX ────────
    // Scatter mỗi kick: out_bram word oc_local (128b) → deint_bram[oc_local*npos+pos].
    // Sau N_POS kick (hết ob) → deint_bram = CHW ob-block; DMA đọc linear qua dma_out_addr[16:4].
    // deint_scat_active mux out_bram rd sang oc_deint; deint_dma_active mux dma_out_data128.
    // (ocd_* + deint_scat_active: khai báo DỜI LÊN trước act_uram — Rev168 dùng ở đó)
    wire [127:0] out_bram_rd128; wire [127:0] deint_bram_rd128;
    // rd_addr: scatter → oc_local*16 (byte, word=[9:4]); thường → dma_out_addr[9:0]
    wire [9:0] ob_rd_addr_mux = deint_scat_active ? {ocd_rd_oc[5:0], 4'd0} : dma_out_addr[9:0];
    out_bram_wide u_bram_out (
        .clk(aclk),
        .wr_waddr(out_bram_wr_waddr), .wr_wdata(out_bram_wr_data), .wr_we(out_bram_wr_we),
        .rd_addr(ob_rd_addr_mux), .rd_bdata(dma_out_data), .rd_wdata128(out_bram_rd128)
    );
    oc_deint #(.AW(17)) u_ocdeint (
        .clk(aclk), .rst_n(aresetn), .start(deint_scat_start),
        .ock(deint_ock),
        // Rev168: đích act_uram CHW (base=pos, stride=npos). Rev169 lag: đích lag_buf
        // slot pos[4:0] (base={pos[4:0],6'b0}, stride=1 — word oc liền kề trong slot).
        // Rev200 SCATOUT: + scat_base_w (word) — scatter về vùng offset (multi-region). Gated=0 cũ.
        // Rev200 TILED (conv1x1 RESOUT, ~c3): word = tbase(pos*cpad) + oc → position-major,
        //   khớp chuẩn TILED consumer conv1x1/graphop. c3 GIỮ CHW (hành vi cũ y nguyên).
        // Rev174 TIMING: 2 tổng base REGISTER hoá (scat_tbase_r/scat_chw_r — khai báo trên,
        //   cạnh scat_base_w). An toàn: deint_pos/tbase đổi ở DSCATW-done, scatter start kế
        //   cách cả một kick engine → giá trị _r luôn settle trước khi oc_deint latch @start.
        .base  (lag_en_c ? {6'd0, deint_pos[4:0], 6'b0}
               : scat_tiled ? scat_tbase_r
               : scat_chw_r),
        .stride(lag_en_c ? 17'd1 : scat_tiled ? 17'd1 : {1'b0, deint_npos}),
        .ob_rd_addr(ocd_rd_oc), .ob_rd_data(out_bram_rd128),
        .db_wr_addr(ocd_wr_addr), .db_wr_data(ocd_wr_data), .db_wr_en(ocd_wr_en),
        .done(deint_scat_done)
    );
    // ── Rev169: lag_buf (32 slot × 64 word) + engine drain (lag_buf → act_uram) ──
    wire [127:0] lag_rd_data;
    lag_buf u_lagbuf (
        .clk(aclk),
        .wr_addr(ocd_wr_addr[10:0]), .wr_data(ocd_wr_data),
        .wr_en(ocd_wr_en & lag_en_c),
        .rd_addr({drain_pos[4:0], 6'b0} | {5'd0, ocd2_rd_oc[5:0]}),
        .rd_data(lag_rd_data)
    );
    oc_deint #(.AW(17)) u_ocdrain (
        .clk(aclk), .rst_n(aresetn), .start(drain_start),
        .ock(deint_ock), .base({1'b0, drain_pos}), .stride({1'b0, deint_npos}),
        .ob_rd_addr(ocd2_rd_oc), .ob_rd_data(lag_rd_data),
        .db_wr_addr(ocd2_wr_addr), .db_wr_data(ocd2_wr_data), .db_wr_en(ocd2_wr_en),
        .done(drain_done)
    );
    // Rev165: deint_bram (512KB, 128 BRAM) XÓA — deint-on-HW đo wash (Rev164).
    // Rev168: oc_deint TÁI SINH = resident-act scatter → act_uram (bit12 nghĩa mới: RESOUT;
    // LS bỏ LS_DOBDMA — xem layer_sequencer). deint_dma_active không bao giờ bật nữa.
    assign deint_bram_rd128 = 128'd0;
    assign dma_out_data128  = out_bram_rd128;

    // ── Weight registers: 16 NZ slots × (1 idx + NS SPEC × G groups) ────────
    // B-w: parameterized by NS=`N_SPEC_ACT (=4 → BIT-IDENTICAL Rev135; =12 → Option C wider array).
    // wt_rd_arr[spec][grp][slot] replaces the flat wt_rd_s{s}g{g}[slot]. FSM loads it serially.
    // (NS/ESZ localparams: khai báo sớm phía trên — Rev139c)
    reg [3:0]        wt_rd_idx  [0:15];
    reg signed [7:0] wt_rd_arr  [0:NS-1][0:`G-1][0:15];   // [spec][grp][slot]

    wire [3:0]        wt_idx   [0:15];
    wire signed [7:0] spec_wg0 [0:NS-1][0:15];
    wire signed [7:0] spec_wg1 [0:NS-1][0:15];
    wire signed [7:0] spec_wg2 [0:NS-1][0:15];
    wire signed [7:0] spec_wg3 [0:NS-1][0:15];
    genvar wi, ssi;
    generate
        for (wi = 0; wi < 16; wi = wi + 1) begin : g_wt_idx
            assign wt_idx[wi] = wt_rd_idx[wi];
        end
        for (ssi = 0; ssi < NS; ssi = ssi + 1) begin : g_spec_wt
            for (wi = 0; wi < 16; wi = wi + 1) begin : g_slot
                assign spec_wg0[ssi][wi] = wt_rd_arr[ssi][0][wi];
                assign spec_wg1[ssi][wi] = wt_rd_arr[ssi][1][wi];
                assign spec_wg2[ssi][wi] = wt_rd_arr[ssi][2][wi];
                assign spec_wg3[ssi][wi] = wt_rd_arr[ssi][3][wi];
            end
        end
    endgenerate

    // ── Activation window registers ───────────────────────────────────────────
    // Rev163 200MHz: act_gN broadcast tới 12 spec (fo=12) → net-bound 5.1ns khi spec đặt xa
    //   (deint_bram BRAM đẩy layout). max_fanout ép replicate → local copy mỗi cụm spec → route ngắn.
    (* max_fanout = 4 *) reg signed [7:0] act_g0 [0:15];
    (* max_fanout = 4 *) reg signed [7:0] act_g1 [0:15];
    (* max_fanout = 4 *) reg signed [7:0] act_g2 [0:15];
    (* max_fanout = 4 *) reg signed [7:0] act_g3 [0:15];
    reg signed [7:0] new_col [0:3];   // rightmost column for 3x3 SHIFT operation

    // ── i_spec control signals ────────────────────────────────────────────────
    localparam SHUF_LOAD  = 2'b00;
    localparam SHUF_SHIFT = 2'b01;
    localparam SHUF_HOLD  = 2'b10;

    reg        wt_load_r, acc_clr_r;
    reg [1:0]  shuf_mode_r;

    // ── sliding_win_ctrl (1 instance — bypassed via MUX when swc_en=0) ────────
    reg        swc_start;
    wire [1:0] swc_kh, swc_kw;
    wire [1:0] swc_shuf_mode_raw;
    wire       swc_kernel_done;

    sliding_win_ctrl u_swc (
        .clk          (aclk),
        .rst_n        (aresetn),
        .start_pixel  (swc_start),
        .kh_out       (swc_kh),
        .kw_out       (swc_kw),
        .shuf_mode_out(swc_shuf_mode_raw),
        .kernel_done  (swc_kernel_done)
    );

    // MUX: when swc_en=0 (1x1), hardwire SHUF_LOAD (no sliding needed)
    wire [1:0] swc_shuf_mode_mux = swc_en ? swc_shuf_mode_raw : SHUF_LOAD;

    // ── NS × i_spec (shared for all conv types) — B-w: NS=`N_SPEC_ACT (4 baseline / 12 Option C) ──
    wire signed [31:0] ps [0:NS-1][0:3][0:15];  // [SPEC][group][spatial]
    wire mac_valid [0:NS-1];

    genvar si;
    generate
        for (si = 0; si < NS; si = si + 1) begin : g_spec
            (* keep_hierarchy = "yes" *) i_spec u_spec (
                .clk(aclk), .rst_n(aresetn),
                .n_cfg(swc_en ? reg_n_config[3:0] : 4'd1), .poc(eff_poc), .apply_relu(reg_apply_relu),
                .wt_idx_0(wt_idx[0]),   .wt_idx_1(wt_idx[1]),
                .wt_idx_2(wt_idx[2]),   .wt_idx_3(wt_idx[3]),
                .wt_idx_4(wt_idx[4]),   .wt_idx_5(wt_idx[5]),
                .wt_idx_6(wt_idx[6]),   .wt_idx_7(wt_idx[7]),
                .wt_idx_8(wt_idx[8]),   .wt_idx_9(wt_idx[9]),
                .wt_idx_10(wt_idx[10]), .wt_idx_11(wt_idx[11]),
                .wt_idx_12(wt_idx[12]), .wt_idx_13(wt_idx[13]),
                .wt_idx_14(wt_idx[14]), .wt_idx_15(wt_idx[15]),
                .wt_val_g0_0(spec_wg0[si][0]),   .wt_val_g0_1(spec_wg0[si][1]),
                .wt_val_g0_2(spec_wg0[si][2]),   .wt_val_g0_3(spec_wg0[si][3]),
                .wt_val_g0_4(spec_wg0[si][4]),   .wt_val_g0_5(spec_wg0[si][5]),
                .wt_val_g0_6(spec_wg0[si][6]),   .wt_val_g0_7(spec_wg0[si][7]),
                .wt_val_g0_8(spec_wg0[si][8]),   .wt_val_g0_9(spec_wg0[si][9]),
                .wt_val_g0_10(spec_wg0[si][10]), .wt_val_g0_11(spec_wg0[si][11]),
                .wt_val_g0_12(spec_wg0[si][12]), .wt_val_g0_13(spec_wg0[si][13]),
                .wt_val_g0_14(spec_wg0[si][14]), .wt_val_g0_15(spec_wg0[si][15]),
                .wt_val_g1_0(spec_wg1[si][0]),   .wt_val_g1_1(spec_wg1[si][1]),
                .wt_val_g1_2(spec_wg1[si][2]),   .wt_val_g1_3(spec_wg1[si][3]),
                .wt_val_g1_4(spec_wg1[si][4]),   .wt_val_g1_5(spec_wg1[si][5]),
                .wt_val_g1_6(spec_wg1[si][6]),   .wt_val_g1_7(spec_wg1[si][7]),
                .wt_val_g1_8(spec_wg1[si][8]),   .wt_val_g1_9(spec_wg1[si][9]),
                .wt_val_g1_10(spec_wg1[si][10]), .wt_val_g1_11(spec_wg1[si][11]),
                .wt_val_g1_12(spec_wg1[si][12]), .wt_val_g1_13(spec_wg1[si][13]),
                .wt_val_g1_14(spec_wg1[si][14]), .wt_val_g1_15(spec_wg1[si][15]),
                .wt_val_g2_0(spec_wg2[si][0]),   .wt_val_g2_1(spec_wg2[si][1]),
                .wt_val_g2_2(spec_wg2[si][2]),   .wt_val_g2_3(spec_wg2[si][3]),
                .wt_val_g2_4(spec_wg2[si][4]),   .wt_val_g2_5(spec_wg2[si][5]),
                .wt_val_g2_6(spec_wg2[si][6]),   .wt_val_g2_7(spec_wg2[si][7]),
                .wt_val_g2_8(spec_wg2[si][8]),   .wt_val_g2_9(spec_wg2[si][9]),
                .wt_val_g2_10(spec_wg2[si][10]), .wt_val_g2_11(spec_wg2[si][11]),
                .wt_val_g2_12(spec_wg2[si][12]), .wt_val_g2_13(spec_wg2[si][13]),
                .wt_val_g2_14(spec_wg2[si][14]), .wt_val_g2_15(spec_wg2[si][15]),
                .wt_val_g3_0(spec_wg3[si][0]),   .wt_val_g3_1(spec_wg3[si][1]),
                .wt_val_g3_2(spec_wg3[si][2]),   .wt_val_g3_3(spec_wg3[si][3]),
                .wt_val_g3_4(spec_wg3[si][4]),   .wt_val_g3_5(spec_wg3[si][5]),
                .wt_val_g3_6(spec_wg3[si][6]),   .wt_val_g3_7(spec_wg3[si][7]),
                .wt_val_g3_8(spec_wg3[si][8]),   .wt_val_g3_9(spec_wg3[si][9]),
                .wt_val_g3_10(spec_wg3[si][10]), .wt_val_g3_11(spec_wg3[si][11]),
                .wt_val_g3_12(spec_wg3[si][12]), .wt_val_g3_13(spec_wg3[si][13]),
                .wt_val_g3_14(spec_wg3[si][14]), .wt_val_g3_15(spec_wg3[si][15]),
                .bm_word(16'd0),
                .bm_val_g0_0(8'sd0),.bm_val_g0_1(8'sd0),.bm_val_g0_2(8'sd0),.bm_val_g0_3(8'sd0),
                .bm_val_g0_4(8'sd0),.bm_val_g0_5(8'sd0),.bm_val_g0_6(8'sd0),.bm_val_g0_7(8'sd0),
                .bm_val_g0_8(8'sd0),.bm_val_g0_9(8'sd0),.bm_val_g0_10(8'sd0),.bm_val_g0_11(8'sd0),
                .bm_val_g0_12(8'sd0),.bm_val_g0_13(8'sd0),.bm_val_g0_14(8'sd0),.bm_val_g0_15(8'sd0),
                .bm_val_g1_0(8'sd0),.bm_val_g1_1(8'sd0),.bm_val_g1_2(8'sd0),.bm_val_g1_3(8'sd0),
                .bm_val_g1_4(8'sd0),.bm_val_g1_5(8'sd0),.bm_val_g1_6(8'sd0),.bm_val_g1_7(8'sd0),
                .bm_val_g1_8(8'sd0),.bm_val_g1_9(8'sd0),.bm_val_g1_10(8'sd0),.bm_val_g1_11(8'sd0),
                .bm_val_g1_12(8'sd0),.bm_val_g1_13(8'sd0),.bm_val_g1_14(8'sd0),.bm_val_g1_15(8'sd0),
                .bm_val_g2_0(8'sd0),.bm_val_g2_1(8'sd0),.bm_val_g2_2(8'sd0),.bm_val_g2_3(8'sd0),
                .bm_val_g2_4(8'sd0),.bm_val_g2_5(8'sd0),.bm_val_g2_6(8'sd0),.bm_val_g2_7(8'sd0),
                .bm_val_g2_8(8'sd0),.bm_val_g2_9(8'sd0),.bm_val_g2_10(8'sd0),.bm_val_g2_11(8'sd0),
                .bm_val_g2_12(8'sd0),.bm_val_g2_13(8'sd0),.bm_val_g2_14(8'sd0),.bm_val_g2_15(8'sd0),
                .bm_val_g3_0(8'sd0),.bm_val_g3_1(8'sd0),.bm_val_g3_2(8'sd0),.bm_val_g3_3(8'sd0),
                .bm_val_g3_4(8'sd0),.bm_val_g3_5(8'sd0),.bm_val_g3_6(8'sd0),.bm_val_g3_7(8'sd0),
                .bm_val_g3_8(8'sd0),.bm_val_g3_9(8'sd0),.bm_val_g3_10(8'sd0),.bm_val_g3_11(8'sd0),
                .bm_val_g3_12(8'sd0),.bm_val_g3_13(8'sd0),.bm_val_g3_14(8'sd0),.bm_val_g3_15(8'sd0),
                .wt_load(wt_load_r), .acc_clr(acc_clr_r),
                .shuf_mode(shuf_mode_r),
                .new_col_0(new_col[0]), .new_col_1(new_col[1]),
                .new_col_2(new_col[2]), .new_col_3(new_col[3]),
                .act_l_g0_p0(act_g0[0]),  .act_l_g0_p1(act_g0[1]),  .act_l_g0_p2(act_g0[2]),  .act_l_g0_p3(act_g0[3]),
                .act_l_g0_p4(act_g0[4]),  .act_l_g0_p5(act_g0[5]),  .act_l_g0_p6(act_g0[6]),  .act_l_g0_p7(act_g0[7]),
                .act_l_g0_p8(act_g0[8]),  .act_l_g0_p9(act_g0[9]),  .act_l_g0_p10(act_g0[10]),.act_l_g0_p11(act_g0[11]),
                .act_l_g0_p12(act_g0[12]),.act_l_g0_p13(act_g0[13]),.act_l_g0_p14(act_g0[14]),.act_l_g0_p15(act_g0[15]),
                .act_l_g1_p0(act_g1[0]),  .act_l_g1_p1(act_g1[1]),  .act_l_g1_p2(act_g1[2]),  .act_l_g1_p3(act_g1[3]),
                .act_l_g1_p4(act_g1[4]),  .act_l_g1_p5(act_g1[5]),  .act_l_g1_p6(act_g1[6]),  .act_l_g1_p7(act_g1[7]),
                .act_l_g1_p8(act_g1[8]),  .act_l_g1_p9(act_g1[9]),  .act_l_g1_p10(act_g1[10]),.act_l_g1_p11(act_g1[11]),
                .act_l_g1_p12(act_g1[12]),.act_l_g1_p13(act_g1[13]),.act_l_g1_p14(act_g1[14]),.act_l_g1_p15(act_g1[15]),
                .act_l_g2_p0(act_g2[0]),  .act_l_g2_p1(act_g2[1]),  .act_l_g2_p2(act_g2[2]),  .act_l_g2_p3(act_g2[3]),
                .act_l_g2_p4(act_g2[4]),  .act_l_g2_p5(act_g2[5]),  .act_l_g2_p6(act_g2[6]),  .act_l_g2_p7(act_g2[7]),
                .act_l_g2_p8(act_g2[8]),  .act_l_g2_p9(act_g2[9]),  .act_l_g2_p10(act_g2[10]),.act_l_g2_p11(act_g2[11]),
                .act_l_g2_p12(act_g2[12]),.act_l_g2_p13(act_g2[13]),.act_l_g2_p14(act_g2[14]),.act_l_g2_p15(act_g2[15]),
                .act_l_g3_p0(act_g3[0]),  .act_l_g3_p1(act_g3[1]),  .act_l_g3_p2(act_g3[2]),  .act_l_g3_p3(act_g3[3]),
                .act_l_g3_p4(act_g3[4]),  .act_l_g3_p5(act_g3[5]),  .act_l_g3_p6(act_g3[6]),  .act_l_g3_p7(act_g3[7]),
                .act_l_g3_p8(act_g3[8]),  .act_l_g3_p9(act_g3[9]),  .act_l_g3_p10(act_g3[10]),.act_l_g3_p11(act_g3[11]),
                .act_l_g3_p12(act_g3[12]),.act_l_g3_p13(act_g3[13]),.act_l_g3_p14(act_g3[14]),.act_l_g3_p15(act_g3[15]),
                .psum_g0_p0(ps[si][0][0]),  .psum_g0_p1(ps[si][0][1]),  .psum_g0_p2(ps[si][0][2]),  .psum_g0_p3(ps[si][0][3]),
                .psum_g0_p4(ps[si][0][4]),  .psum_g0_p5(ps[si][0][5]),  .psum_g0_p6(ps[si][0][6]),  .psum_g0_p7(ps[si][0][7]),
                .psum_g0_p8(ps[si][0][8]),  .psum_g0_p9(ps[si][0][9]),  .psum_g0_p10(ps[si][0][10]),.psum_g0_p11(ps[si][0][11]),
                .psum_g0_p12(ps[si][0][12]),.psum_g0_p13(ps[si][0][13]),.psum_g0_p14(ps[si][0][14]),.psum_g0_p15(ps[si][0][15]),
                .psum_g1_p0(ps[si][1][0]),  .psum_g1_p1(ps[si][1][1]),  .psum_g1_p2(ps[si][1][2]),  .psum_g1_p3(ps[si][1][3]),
                .psum_g1_p4(ps[si][1][4]),  .psum_g1_p5(ps[si][1][5]),  .psum_g1_p6(ps[si][1][6]),  .psum_g1_p7(ps[si][1][7]),
                .psum_g1_p8(ps[si][1][8]),  .psum_g1_p9(ps[si][1][9]),  .psum_g1_p10(ps[si][1][10]),.psum_g1_p11(ps[si][1][11]),
                .psum_g1_p12(ps[si][1][12]),.psum_g1_p13(ps[si][1][13]),.psum_g1_p14(ps[si][1][14]),.psum_g1_p15(ps[si][1][15]),
                .psum_g2_p0(ps[si][2][0]),  .psum_g2_p1(ps[si][2][1]),  .psum_g2_p2(ps[si][2][2]),  .psum_g2_p3(ps[si][2][3]),
                .psum_g2_p4(ps[si][2][4]),  .psum_g2_p5(ps[si][2][5]),  .psum_g2_p6(ps[si][2][6]),  .psum_g2_p7(ps[si][2][7]),
                .psum_g2_p8(ps[si][2][8]),  .psum_g2_p9(ps[si][2][9]),  .psum_g2_p10(ps[si][2][10]),.psum_g2_p11(ps[si][2][11]),
                .psum_g2_p12(ps[si][2][12]),.psum_g2_p13(ps[si][2][13]),.psum_g2_p14(ps[si][2][14]),.psum_g2_p15(ps[si][2][15]),
                .psum_g3_p0(ps[si][3][0]),  .psum_g3_p1(ps[si][3][1]),  .psum_g3_p2(ps[si][3][2]),  .psum_g3_p3(ps[si][3][3]),
                .psum_g3_p4(ps[si][3][4]),  .psum_g3_p5(ps[si][3][5]),  .psum_g3_p6(ps[si][3][6]),  .psum_g3_p7(ps[si][3][7]),
                .psum_g3_p8(ps[si][3][8]),  .psum_g3_p9(ps[si][3][9]),  .psum_g3_p10(ps[si][3][10]),.psum_g3_p11(ps[si][3][11]),
                .psum_g3_p12(ps[si][3][12]),.psum_g3_p13(ps[si][3][13]),.psum_g3_p14(ps[si][3][14]),.psum_g3_p15(ps[si][3][15]),
                .mac_valid(mac_valid[si])
            );
        end
    endgenerate

    // ── accu_top (combinational) ──────────────────────────────────────────────
    wire signed [31:0] accu [0:3][0:15];

    accu_top u_accu (
        .accu_mode(eff_accu_mode),
        .s0_g0p0(ps[0][0][0]),  .s0_g0p1(ps[0][0][1]),  .s0_g0p2(ps[0][0][2]),  .s0_g0p3(ps[0][0][3]),
        .s0_g0p4(ps[0][0][4]),  .s0_g0p5(ps[0][0][5]),  .s0_g0p6(ps[0][0][6]),  .s0_g0p7(ps[0][0][7]),
        .s0_g0p8(ps[0][0][8]),  .s0_g0p9(ps[0][0][9]),  .s0_g0p10(ps[0][0][10]),.s0_g0p11(ps[0][0][11]),
        .s0_g0p12(ps[0][0][12]),.s0_g0p13(ps[0][0][13]),.s0_g0p14(ps[0][0][14]),.s0_g0p15(ps[0][0][15]),
        .s0_g1p0(ps[0][1][0]),  .s0_g1p1(ps[0][1][1]),  .s0_g1p2(ps[0][1][2]),  .s0_g1p3(ps[0][1][3]),
        .s0_g1p4(ps[0][1][4]),  .s0_g1p5(ps[0][1][5]),  .s0_g1p6(ps[0][1][6]),  .s0_g1p7(ps[0][1][7]),
        .s0_g1p8(ps[0][1][8]),  .s0_g1p9(ps[0][1][9]),  .s0_g1p10(ps[0][1][10]),.s0_g1p11(ps[0][1][11]),
        .s0_g1p12(ps[0][1][12]),.s0_g1p13(ps[0][1][13]),.s0_g1p14(ps[0][1][14]),.s0_g1p15(ps[0][1][15]),
        .s0_g2p0(ps[0][2][0]),  .s0_g2p1(ps[0][2][1]),  .s0_g2p2(ps[0][2][2]),  .s0_g2p3(ps[0][2][3]),
        .s0_g2p4(ps[0][2][4]),  .s0_g2p5(ps[0][2][5]),  .s0_g2p6(ps[0][2][6]),  .s0_g2p7(ps[0][2][7]),
        .s0_g2p8(ps[0][2][8]),  .s0_g2p9(ps[0][2][9]),  .s0_g2p10(ps[0][2][10]),.s0_g2p11(ps[0][2][11]),
        .s0_g2p12(ps[0][2][12]),.s0_g2p13(ps[0][2][13]),.s0_g2p14(ps[0][2][14]),.s0_g2p15(ps[0][2][15]),
        .s0_g3p0(ps[0][3][0]),  .s0_g3p1(ps[0][3][1]),  .s0_g3p2(ps[0][3][2]),  .s0_g3p3(ps[0][3][3]),
        .s0_g3p4(ps[0][3][4]),  .s0_g3p5(ps[0][3][5]),  .s0_g3p6(ps[0][3][6]),  .s0_g3p7(ps[0][3][7]),
        .s0_g3p8(ps[0][3][8]),  .s0_g3p9(ps[0][3][9]),  .s0_g3p10(ps[0][3][10]),.s0_g3p11(ps[0][3][11]),
        .s0_g3p12(ps[0][3][12]),.s0_g3p13(ps[0][3][13]),.s0_g3p14(ps[0][3][14]),.s0_g3p15(ps[0][3][15]),
        .s1_g0p0(ps[1][0][0]),  .s1_g0p1(ps[1][0][1]),  .s1_g0p2(ps[1][0][2]),  .s1_g0p3(ps[1][0][3]),
        .s1_g0p4(ps[1][0][4]),  .s1_g0p5(ps[1][0][5]),  .s1_g0p6(ps[1][0][6]),  .s1_g0p7(ps[1][0][7]),
        .s1_g0p8(ps[1][0][8]),  .s1_g0p9(ps[1][0][9]),  .s1_g0p10(ps[1][0][10]),.s1_g0p11(ps[1][0][11]),
        .s1_g0p12(ps[1][0][12]),.s1_g0p13(ps[1][0][13]),.s1_g0p14(ps[1][0][14]),.s1_g0p15(ps[1][0][15]),
        .s1_g1p0(ps[1][1][0]),  .s1_g1p1(ps[1][1][1]),  .s1_g1p2(ps[1][1][2]),  .s1_g1p3(ps[1][1][3]),
        .s1_g1p4(ps[1][1][4]),  .s1_g1p5(ps[1][1][5]),  .s1_g1p6(ps[1][1][6]),  .s1_g1p7(ps[1][1][7]),
        .s1_g1p8(ps[1][1][8]),  .s1_g1p9(ps[1][1][9]),  .s1_g1p10(ps[1][1][10]),.s1_g1p11(ps[1][1][11]),
        .s1_g1p12(ps[1][1][12]),.s1_g1p13(ps[1][1][13]),.s1_g1p14(ps[1][1][14]),.s1_g1p15(ps[1][1][15]),
        .s1_g2p0(ps[1][2][0]),  .s1_g2p1(ps[1][2][1]),  .s1_g2p2(ps[1][2][2]),  .s1_g2p3(ps[1][2][3]),
        .s1_g2p4(ps[1][2][4]),  .s1_g2p5(ps[1][2][5]),  .s1_g2p6(ps[1][2][6]),  .s1_g2p7(ps[1][2][7]),
        .s1_g2p8(ps[1][2][8]),  .s1_g2p9(ps[1][2][9]),  .s1_g2p10(ps[1][2][10]),.s1_g2p11(ps[1][2][11]),
        .s1_g2p12(ps[1][2][12]),.s1_g2p13(ps[1][2][13]),.s1_g2p14(ps[1][2][14]),.s1_g2p15(ps[1][2][15]),
        .s1_g3p0(ps[1][3][0]),  .s1_g3p1(ps[1][3][1]),  .s1_g3p2(ps[1][3][2]),  .s1_g3p3(ps[1][3][3]),
        .s1_g3p4(ps[1][3][4]),  .s1_g3p5(ps[1][3][5]),  .s1_g3p6(ps[1][3][6]),  .s1_g3p7(ps[1][3][7]),
        .s1_g3p8(ps[1][3][8]),  .s1_g3p9(ps[1][3][9]),  .s1_g3p10(ps[1][3][10]),.s1_g3p11(ps[1][3][11]),
        .s1_g3p12(ps[1][3][12]),.s1_g3p13(ps[1][3][13]),.s1_g3p14(ps[1][3][14]),.s1_g3p15(ps[1][3][15]),
        .s2_g0p0(ps[2][0][0]),  .s2_g0p1(ps[2][0][1]),  .s2_g0p2(ps[2][0][2]),  .s2_g0p3(ps[2][0][3]),
        .s2_g0p4(ps[2][0][4]),  .s2_g0p5(ps[2][0][5]),  .s2_g0p6(ps[2][0][6]),  .s2_g0p7(ps[2][0][7]),
        .s2_g0p8(ps[2][0][8]),  .s2_g0p9(ps[2][0][9]),  .s2_g0p10(ps[2][0][10]),.s2_g0p11(ps[2][0][11]),
        .s2_g0p12(ps[2][0][12]),.s2_g0p13(ps[2][0][13]),.s2_g0p14(ps[2][0][14]),.s2_g0p15(ps[2][0][15]),
        .s2_g1p0(ps[2][1][0]),  .s2_g1p1(ps[2][1][1]),  .s2_g1p2(ps[2][1][2]),  .s2_g1p3(ps[2][1][3]),
        .s2_g1p4(ps[2][1][4]),  .s2_g1p5(ps[2][1][5]),  .s2_g1p6(ps[2][1][6]),  .s2_g1p7(ps[2][1][7]),
        .s2_g1p8(ps[2][1][8]),  .s2_g1p9(ps[2][1][9]),  .s2_g1p10(ps[2][1][10]),.s2_g1p11(ps[2][1][11]),
        .s2_g1p12(ps[2][1][12]),.s2_g1p13(ps[2][1][13]),.s2_g1p14(ps[2][1][14]),.s2_g1p15(ps[2][1][15]),
        .s2_g2p0(ps[2][2][0]),  .s2_g2p1(ps[2][2][1]),  .s2_g2p2(ps[2][2][2]),  .s2_g2p3(ps[2][2][3]),
        .s2_g2p4(ps[2][2][4]),  .s2_g2p5(ps[2][2][5]),  .s2_g2p6(ps[2][2][6]),  .s2_g2p7(ps[2][2][7]),
        .s2_g2p8(ps[2][2][8]),  .s2_g2p9(ps[2][2][9]),  .s2_g2p10(ps[2][2][10]),.s2_g2p11(ps[2][2][11]),
        .s2_g2p12(ps[2][2][12]),.s2_g2p13(ps[2][2][13]),.s2_g2p14(ps[2][2][14]),.s2_g2p15(ps[2][2][15]),
        .s2_g3p0(ps[2][3][0]),  .s2_g3p1(ps[2][3][1]),  .s2_g3p2(ps[2][3][2]),  .s2_g3p3(ps[2][3][3]),
        .s2_g3p4(ps[2][3][4]),  .s2_g3p5(ps[2][3][5]),  .s2_g3p6(ps[2][3][6]),  .s2_g3p7(ps[2][3][7]),
        .s2_g3p8(ps[2][3][8]),  .s2_g3p9(ps[2][3][9]),  .s2_g3p10(ps[2][3][10]),.s2_g3p11(ps[2][3][11]),
        .s2_g3p12(ps[2][3][12]),.s2_g3p13(ps[2][3][13]),.s2_g3p14(ps[2][3][14]),.s2_g3p15(ps[2][3][15]),
        .s3_g0p0(ps[3][0][0]),  .s3_g0p1(ps[3][0][1]),  .s3_g0p2(ps[3][0][2]),  .s3_g0p3(ps[3][0][3]),
        .s3_g0p4(ps[3][0][4]),  .s3_g0p5(ps[3][0][5]),  .s3_g0p6(ps[3][0][6]),  .s3_g0p7(ps[3][0][7]),
        .s3_g0p8(ps[3][0][8]),  .s3_g0p9(ps[3][0][9]),  .s3_g0p10(ps[3][0][10]),.s3_g0p11(ps[3][0][11]),
        .s3_g0p12(ps[3][0][12]),.s3_g0p13(ps[3][0][13]),.s3_g0p14(ps[3][0][14]),.s3_g0p15(ps[3][0][15]),
        .s3_g1p0(ps[3][1][0]),  .s3_g1p1(ps[3][1][1]),  .s3_g1p2(ps[3][1][2]),  .s3_g1p3(ps[3][1][3]),
        .s3_g1p4(ps[3][1][4]),  .s3_g1p5(ps[3][1][5]),  .s3_g1p6(ps[3][1][6]),  .s3_g1p7(ps[3][1][7]),
        .s3_g1p8(ps[3][1][8]),  .s3_g1p9(ps[3][1][9]),  .s3_g1p10(ps[3][1][10]),.s3_g1p11(ps[3][1][11]),
        .s3_g1p12(ps[3][1][12]),.s3_g1p13(ps[3][1][13]),.s3_g1p14(ps[3][1][14]),.s3_g1p15(ps[3][1][15]),
        .s3_g2p0(ps[3][2][0]),  .s3_g2p1(ps[3][2][1]),  .s3_g2p2(ps[3][2][2]),  .s3_g2p3(ps[3][2][3]),
        .s3_g2p4(ps[3][2][4]),  .s3_g2p5(ps[3][2][5]),  .s3_g2p6(ps[3][2][6]),  .s3_g2p7(ps[3][2][7]),
        .s3_g2p8(ps[3][2][8]),  .s3_g2p9(ps[3][2][9]),  .s3_g2p10(ps[3][2][10]),.s3_g2p11(ps[3][2][11]),
        .s3_g2p12(ps[3][2][12]),.s3_g2p13(ps[3][2][13]),.s3_g2p14(ps[3][2][14]),.s3_g2p15(ps[3][2][15]),
        .s3_g3p0(ps[3][3][0]),  .s3_g3p1(ps[3][3][1]),  .s3_g3p2(ps[3][3][2]),  .s3_g3p3(ps[3][3][3]),
        .s3_g3p4(ps[3][3][4]),  .s3_g3p5(ps[3][3][5]),  .s3_g3p6(ps[3][3][6]),  .s3_g3p7(ps[3][3][7]),
        .s3_g3p8(ps[3][3][8]),  .s3_g3p9(ps[3][3][9]),  .s3_g3p10(ps[3][3][10]),.s3_g3p11(ps[3][3][11]),
        .s3_g3p12(ps[3][3][12]),.s3_g3p13(ps[3][3][13]),.s3_g3p14(ps[3][3][14]),.s3_g3p15(ps[3][3][15]),
        .out_g0p0(accu[0][0]),  .out_g0p1(accu[0][1]),  .out_g0p2(accu[0][2]),  .out_g0p3(accu[0][3]),
        .out_g0p4(accu[0][4]),  .out_g0p5(accu[0][5]),  .out_g0p6(accu[0][6]),  .out_g0p7(accu[0][7]),
        .out_g0p8(accu[0][8]),  .out_g0p9(accu[0][9]),  .out_g0p10(accu[0][10]),.out_g0p11(accu[0][11]),
        .out_g0p12(accu[0][12]),.out_g0p13(accu[0][13]),.out_g0p14(accu[0][14]),.out_g0p15(accu[0][15]),
        .out_g1p0(accu[1][0]),  .out_g1p1(accu[1][1]),  .out_g1p2(accu[1][2]),  .out_g1p3(accu[1][3]),
        .out_g1p4(accu[1][4]),  .out_g1p5(accu[1][5]),  .out_g1p6(accu[1][6]),  .out_g1p7(accu[1][7]),
        .out_g1p8(accu[1][8]),  .out_g1p9(accu[1][9]),  .out_g1p10(accu[1][10]),.out_g1p11(accu[1][11]),
        .out_g1p12(accu[1][12]),.out_g1p13(accu[1][13]),.out_g1p14(accu[1][14]),.out_g1p15(accu[1][15]),
        .out_g2p0(accu[2][0]),  .out_g2p1(accu[2][1]),  .out_g2p2(accu[2][2]),  .out_g2p3(accu[2][3]),
        .out_g2p4(accu[2][4]),  .out_g2p5(accu[2][5]),  .out_g2p6(accu[2][6]),  .out_g2p7(accu[2][7]),
        .out_g2p8(accu[2][8]),  .out_g2p9(accu[2][9]),  .out_g2p10(accu[2][10]),.out_g2p11(accu[2][11]),
        .out_g2p12(accu[2][12]),.out_g2p13(accu[2][13]),.out_g2p14(accu[2][14]),.out_g2p15(accu[2][15]),
        .out_g3p0(accu[3][0]),  .out_g3p1(accu[3][1]),  .out_g3p2(accu[3][2]),  .out_g3p3(accu[3][3]),
        .out_g3p4(accu[3][4]),  .out_g3p5(accu[3][5]),  .out_g3p6(accu[3][6]),  .out_g3p7(accu[3][7]),
        .out_g3p8(accu[3][8]),  .out_g3p9(accu[3][9]),  .out_g3p10(accu[3][10]),.out_g3p11(accu[3][11]),
        .out_g3p12(accu[3][12]),.out_g3p13(accu[3][13]),.out_g3p14(accu[3][14]),.out_g3p15(accu[3][15])
    );

    // ── tile_ctrl: halo driven by use_halo from config_decoder ────────────────
    wire [9:0] tc_h_start, tc_w_start;
    wire [5:0] tc_h_act,   tc_w_act;
    wire [1:0] tc_pad_top, tc_pad_bot, tc_pad_left, tc_pad_right;
    wire       tc_valid, tc_last;
    reg        tc_start, tc_next;

    tile_ctrl u_tile (
        .clk(aclk), .rst_n(aresetn),
        .Hin(reg_hin[9:0]), .Win(reg_win[9:0]),
        // 1x1: single tile = full image; 3x3/dw: 28x28 tiles
        .tile_h(swc_en ? 6'd28        : reg_hin[5:0]),
        .tile_w(swc_en ? 6'd28        : reg_win[5:0]),
        .halo  (use_halo ? 2'd1 : 2'd0),
        .start(tc_start), .tile_next(tc_next),
        .tile_h_start(tc_h_start), .tile_w_start(tc_w_start),
        .tile_h_act(tc_h_act),     .tile_w_act(tc_w_act),
        .pad_top(tc_pad_top),  .pad_bot(tc_pad_bot),
        .pad_left(tc_pad_left),.pad_right(tc_pad_right),
        .tile_valid(tc_valid), .tile_last(tc_last)
    );

    // ── router ────────────────────────────────────────────────────────────────
    wire rtr_valid_s0;
    router u_router (
        .router_mode(2'b10), .sel(4'b1111),
        .data_in(8'd0), .valid_in(wt_load_r),
        .data_out_0(), .data_out_1(), .data_out_2(), .data_out_3(),
        .valid_out_0(rtr_valid_s0), .valid_out_1(), .valid_out_2(), .valid_out_3()
    );

    // ── Per-channel requantization parameters (loaded from WT BRAM tail) ────────
    // Layout after NZ weight entries: interleaved [m0_lo, m0_hi, sh, bias×4] × Cout
    // Up to 256 output channels. Indexed by out_ch = flush_spec*4+flush_g (spec×G + g)
    reg [15:0] rq_m0   [0:255];  // M0 multiplier per output channel (uint16)
    reg [4:0]  rq_sh   [0:255];  // right-shift amount per output channel (5-bit)
    reg signed [31:0] rq_bias [0:255];  // INT32 bias per output channel

    // ── rr_unit with pipeline register (accu_top comb → rr_unit) ─────────────
    wire signed [31:0] accu_flat [0:63];
    genvar gfi, pfi;
    generate
        for (gfi = 0; gfi < 4; gfi = gfi + 1) begin : gf_g
            for (pfi = 0; pfi < 16; pfi = pfi + 1) begin : gf_p
                assign accu_flat[gfi*16 + pfi] = accu[gfi][pfi];
            end
        end
    endgenerate

    reg [1:0] flush_g;
    reg [3:0] flush_p;
    reg [3:0] flush_spec;  // B-w: up to NS SPECs (was [1:0]); writeback iterates 0..NS-1
    (* keep = "true" *) reg flush_valid;
    // 3x3/dw: main-FSM-owned kernel position (lock-step). Replaces free-running
    // sliding_win_ctrl for timing: u_swc raced ahead and fired kernel_done before
    // the main FSM finished position 0 → psum never accumulated. Now the main FSM
    // owns (mf_kh,mf_kw), advancing one tap per compute pass; each pass re-loads the
    // full shifted 16-wide window from BRAM and SHUF_LOADs it (no SHIFT needed).
    reg [1:0] mf_kh, mf_kw;
    // 3x3/dw multi-input-channel: each input channel = one ic iteration. The 9 taps
    // of channel c accumulate into psum, then advance to channel c+1 (no acc_clr),
    // mirroring the 1x1 ic_blk loop (validated). IN BRAM layout = CHW: channel c at
    // c*tile_chw (tile_chw = Hin*Win). Weights for (c,tap) at (c*9+tap)*n_cfg*17.
    reg  [7:0]  swc_ic;                 // current input channel (0..Cin-1)
    // dwconv: the 4 GROUPS = 4 output(=input) channels processed in parallel (no
    // cross-channel sum). ST_ACT_LOAD loads 4 DIFFERENT channels (base+g at g*tile_chw)
    // into act_g0..g3 via this group counter (vs conv3x3 which copies act_g0→g1/g2/g3).
    reg  [1:0]  swc_grp;                // dwconv: which group/channel being loaded (0..3)
    // Channel-tiling (Cin > what fits IN/WT BRAM in one kick): the host splits Cin into
    // chunks and the psum_array PERSISTS between kicks (cleared only on acc_clr). Two
    // control bits packed into the otherwise-unused PAD register (0x130, datapath
    // ignores pad — host supplies halo): keep_psum = skip acc_clr (accumulate onto the
    // running psum from the previous chunk); no_flush = skip ST_FLUSH_RR (just
    // accumulate, no requant/output this kick). first chunk: keep=0,flush=1→no_flush=0;
    // middle: keep=1,no_flush=1; last: keep=1,no_flush=0; single: keep=0,no_flush=0.
    // B0: layer_sequencer drives keep/no_flush/ap_start/dma_in_done per tile when layer_mode=1
    wire        mode_layer    = reg_layer_mode;
    wire        keep_psum     = mode_layer ? ls_k_keep_psum  : reg_pad[8];
    wire        no_flush      = mode_layer ? ls_k_no_flush   : reg_pad[9];
    wire        ap_start_eff  = mode_layer ? ls_k_ap_start   : ap_start_w;
    wire        dma_in_done_eff = mode_layer ? ls_k_dma_in_done : dma_in_done;
    // Rev139 fastkick (feat bit2): rq regs persist across LS ic-kicks → load RQ only on the
    // first kick (keep_psum=0); writeback only on the flush kick (no_flush=0). Middle kicks
    // of the LS ic-loop skip both (their rr outputs were discarded anyway — LS_OUT DMAs the
    // out_bram once, AFTER the last kick). A/B-testable per layer via REG_FEAT_EN bit2.
    wire        fastkick      = reg_feat_en[2] & mode_layer & ~swc_en;
    wire [15:0] tile_chw = reg_hin[7:0] * reg_win[7:0];   // per-channel IN BRAM stride
    // conv3x3: accumulate over Cin input channels. dwconv: depthwise (NO cross-channel
    // sum — each out_ch = its own in_ch) → only 1 ic iteration (the 4 groups handle 4
    // channels). TODO dwconv: ST_ACT_LOAD must load 4 DIFFERENT channels into
    // act_g0..g3 (currently copies act_g0); until then dw is single-channel only.
    wire [7:0]  swc_n_ic = dw_mode ? 8'd1 : reg_cin[7:0]; // input channels per tile

    wire [5:0] accu_idx = {flush_g, flush_p};

    // 1x1: flush_spec selects which i_spec to read (accu_top only passes SPEC0).
    // Use ps[flush_spec][g][p] directly for 1x1; accu_top adder-tree for 3x3/dw.
    wire signed [31:0] psum_spec [0:NS-1][0:3][0:15];
    genvar si2, gi2, pi2;
    generate
        for (si2 = 0; si2 < NS; si2 = si2 + 1) begin : gps_s
            for (gi2 = 0; gi2 < 4; gi2 = gi2 + 1) begin : gps_g
                for (pi2 = 0; pi2 < 16; pi2 = pi2 + 1) begin : gps_p
                    assign psum_spec[si2][gi2][pi2] = ps[si2][gi2][pi2];
                end
            end
        end
    endgenerate

    wire signed [31:0] psum_1x1 = psum_spec[flush_spec][flush_g][flush_p];
    wire signed [31:0] psum_raw = swc_en ? accu_flat[accu_idx] : psum_1x1;

    // Output channel index during flush: spec*G + g (not including spatial position p)
    wire [7:0] out_ch = {2'b0, flush_spec, flush_g};   // B-w: flush_spec now [3:0] → {2'b0,4b,2b}=8b

    // ── Rev152 200MHz: MUX-TREE per-spec cho psum select (768:1 = NET-BOUND, gom psum trải
    //    khắp systolic array → routing 3.9ns @200). Tầng -1: MỖI spec chọn (flush_g,flush_p)
    //    = 64:1 LOCAL (psum 1 spec ở cùng cột array → routing ngắn), register NS×4 lane.
    //    Tầng 0 (pipe0 dưới): gom 12 spec-value (12:1, ít net). Cũng register swc/params/valid
    //    để CĂN (+1 cycle). Bit-exact (giá trị y hệt, chỉ trễ 1 — valid-driven writeback trong suốt).
    // ── Rev159 200MHz: THÊM 1 STAGE cho nhánh SWC (c3/dw). Path cũ = psum_spec →
    //    accu_top (cộng 12 spec, CARRY8×4) → mux 64:1 (accu_idx) → swc_pm0 = -0.146ns
    //    (37 endpoint, worst post-route). TÁCH: register accu_flat (adder-tree out) rồi
    //    mux đọc BẢN REGISTERED. accu_flat ỔN ĐỊNH suốt flush (psum_spec đóng băng) nên
    //    register liên tục = đúng giá trị. pm1(1×1)+params+valid delay +1 khớp (mux cuối
    //    2:1 cùng latency). +1 cycle TỔNG, valid-driven (owr đếm owr_en_r, wb đếm rr_valid)
    //    → BIT-EXACT (swc_pm0(T+2)=accu_flat(T)[accu_idx(T)] = giá trị cũ trễ 1). ──────────
    // Stage -2 (s2): register accu_flat (split adder|mux) + pm1 local + idx/params/valid.
    reg signed [31:0] accu_flat_r [0:63];
    reg signed [31:0] pm1_0 [0:NS-1];
    reg signed [31:0] pm1_1 [0:NS-1];
    reg signed [31:0] pm1_2 [0:NS-1];
    reg signed [31:0] pm1_3 [0:NS-1];
    reg  [1:0]        flush_g_s2;
    reg  [3:0]        flush_p_s2;
    reg  [3:0]        flush_spec_s2;
    reg               swc_en_s2;
    reg signed [31:0] bias_s2;
    reg  [15:0]       m0_s2;
    reg  [4:0]        sh_s2;
    (* keep = "true" *) reg valid_s2;
    integer pmi, afi;
    always @(posedge aclk) begin
        if (!aresetn) begin
            swc_en_s2 <= 1'b0; valid_s2 <= 1'b0;
            flush_g_s2 <= 2'd0; flush_p_s2 <= 4'd0; flush_spec_s2 <= 4'd0;
            bias_s2 <= 32'sd0; m0_s2 <= 16'h4000; sh_s2 <= 5'd15;
        end else begin
            for (afi = 0; afi < 64; afi = afi + 1) accu_flat_r[afi] <= accu_flat[afi];
            for (pmi = 0; pmi < NS; pmi = pmi + 1) begin
                pm1_0[pmi] <= psum_spec[pmi][flush_g][flush_p];          // 64:1 local/spec
                pm1_1[pmi] <= psum_spec[pmi][flush_g][flush_p | 4'd1];
                pm1_2[pmi] <= psum_spec[pmi][flush_g][flush_p | 4'd2];
                pm1_3[pmi] <= psum_spec[pmi][flush_g][flush_p | 4'd3];
            end
            flush_g_s2    <= flush_g;
            flush_p_s2    <= flush_p;
            flush_spec_s2 <= flush_spec;
            swc_en_s2     <= swc_en;
            valid_s2      <= flush_valid;
            bias_s2       <= rq_bias[out_ch];
            m0_s2         <= rq_m0[out_ch];
            sh_s2         <= rq_sh[out_ch];
        end
    end
    // Stage -1 (d1): mux 64:1 ĐỌC BẢN REGISTERED (chỉ mux, không kèm adder → nhanh) +
    //    pm1 gather 12:1 (register để căn +1 với swc) + params/valid delay.
    reg signed [31:0] swc_pm0, swc_pm1, swc_pm2, swc_pm3;
    reg signed [31:0] pm1g_0, pm1g_1, pm1g_2, pm1g_3;
    reg               swc_en_d1;
    reg signed [31:0] pm1_bias;
    reg  [15:0]       pm1_m0;
    reg  [4:0]        pm1_sh;
    (* keep = "true" *) reg pm1_valid;
    always @(posedge aclk) begin
        if (!aresetn) begin
            swc_pm0 <= 32'sd0; swc_pm1 <= 32'sd0; swc_pm2 <= 32'sd0; swc_pm3 <= 32'sd0;
            pm1g_0 <= 32'sd0; pm1g_1 <= 32'sd0; pm1g_2 <= 32'sd0; pm1g_3 <= 32'sd0;
            swc_en_d1 <= 1'b0;
            pm1_bias <= 32'sd0; pm1_m0 <= 16'h4000; pm1_sh <= 5'd15; pm1_valid <= 1'b0;
        end else begin
            swc_pm0 <= accu_flat_r[{flush_g_s2, flush_p_s2}];            // registered mux (fast)
            swc_pm1 <= accu_flat_r[{flush_g_s2, flush_p_s2 | 4'd1}];
            swc_pm2 <= accu_flat_r[{flush_g_s2, flush_p_s2 | 4'd2}];
            swc_pm3 <= accu_flat_r[{flush_g_s2, flush_p_s2 | 4'd3}];
            pm1g_0 <= pm1_0[flush_spec_s2];                              // 12:1 gather
            pm1g_1 <= pm1_1[flush_spec_s2];
            pm1g_2 <= pm1_2[flush_spec_s2];
            pm1g_3 <= pm1_3[flush_spec_s2];
            swc_en_d1 <= swc_en_s2;
            pm1_bias  <= bias_s2;
            pm1_m0    <= m0_s2;
            pm1_sh    <= sh_s2;
            pm1_valid <= valid_s2;
        end
    end
    // Stage 0: mux cuối 2:1 (swc vs 1×1, cả 2 cùng latency +2). pipe0 dùng psum_raw_t + pm1_*.
    wire signed [31:0] psum_raw_t = swc_en_d1 ? swc_pm0 : pm1g_0;
    wire signed [31:0] psum_q1_t  = swc_en_d1 ? swc_pm1 : pm1g_1;
    wire signed [31:0] psum_q2_t  = swc_en_d1 ? swc_pm2 : pm1g_2;
    wire signed [31:0] psum_q3_t  = swc_en_d1 ? swc_pm3 : pm1g_3;

    // ── Rev151 200MHz: PIPELINE requant — tách mux psum (768:1, path dài @166/200)
    //    khỏi bias-add. Tầng0: register mux psum + rq params (theo flush state cycle này).
    //    Tầng1: bias-add + pass. +1 cycle latency; writeback VALID-DRIVEN (owr_ptr đếm
    //    owr_en_r, wb_done_cnt đếm rr_valid) nên dời 1 cycle TRONG SUỐT — bit-exact,
    //    không phá FSM. keep="true" giữ valid FF riêng (merge = lệch 1 pixel). ──────────
    reg signed [31:0] pipe0_psum_r, pipe0_bias_r;
    (* keep = "true" *) reg pipe0_valid_r;
    reg [15:0]        pipe0_m0_r;
    reg [4:0]         pipe0_sh_r;
    reg signed [31:0] pipe_accu_r;
    (* keep = "true" *) reg pipe_accu_valid_r;
    reg [15:0]        pipe_m0_r;
    reg [4:0]         pipe_sh_r;
    always @(posedge aclk) begin
        if (!aresetn) begin
            pipe0_psum_r      <= 32'sd0; pipe0_bias_r <= 32'sd0;
            pipe0_valid_r     <= 1'b0;
            pipe0_m0_r        <= 16'h4000; pipe0_sh_r <= 5'd15;
            pipe_accu_r       <= 32'sd0;
            pipe_accu_valid_r <= 1'b0;
            pipe_m0_r         <= 16'h4000;
            pipe_sh_r         <= 5'd15;
        end else begin
            // Stage 0: gom spec-value tầng -1 (pm1, đã register) — mux 12:1 thay 768:1.
            // params đã register ở pm1 (căn cùng cycle với psum_raw_t).
            pipe0_psum_r      <= psum_raw_t;
            pipe0_bias_r      <= pm1_bias;
            pipe0_valid_r     <= pm1_valid;
            pipe0_m0_r        <= pm1_m0;
            pipe0_sh_r        <= pm1_sh;
            // Stage 1: bias-add (INT32) + pass params — tất cả từ cùng cycle (aligned)
            pipe_accu_r       <= pipe0_psum_r + pipe0_bias_r;
            pipe_accu_valid_r <= pipe0_valid_r;
            pipe_m0_r         <= pipe0_m0_r;
            pipe_sh_r         <= pipe0_sh_r;
        end
    end

    wire signed [7:0] rr_out;
    wire              rr_valid;
    rr_unit u_rr (
        .clk(aclk), .rst_n(aresetn),
        .psum_in(pipe_accu_r), .rq_scale(pipe_m0_r), .rq_shift(pipe_sh_r),
        .apply_relu(reg_apply_relu), .valid_in(pipe_accu_valid_r),
        .data_out(rr_out), .valid_out(rr_valid)
    );

    // ── Đòn 4a' RR4 (feat bit6): 4 rr song song — writeback NS*64 → NS*16 cycle. ──
    // Thứ tự flush p-inner ⇒ 4 pos liên tiếp (cùng spec,g = cùng out_ch/rq params) =
    // 4 byte OUT liên tiếp, word-aligned. flush_p nhảy +4 (p[1:0]=0) → lane k = p|k;
    // lane 0 = chính u_rr (psum_1x1 sẵn có), lane 1-3 = 3 rr_unit clone chia sẻ
    // m0/sh/valid (cùng oc). ~3 DSP/rr ×3 = +9 DSP.
    wire rr4_en = reg_feat_en[5] & mode_layer & ~swc_en;
    wire rqd_en = reg_feat_en[6] & mode_layer & ~swc_en;   // RQ dirty-skip (bit7)
    // Rev142 FR4 (bit10): quad-flush cho đường swc/DW (ST_FLUSH_RR) — cùng 4 rr lanes.
    wire fr4_en = reg_feat_en[9] & mode_layer & swc_en;
    reg  rq_dirty;                                          // wt_uram đổi → RQ phải nạp lại
    // Lane 1-3: mux swc (accu_flat) / 1x1 (psum_spec) — mirror psum_raw của lane 0.
    wire signed [31:0] psum_q1 = swc_en ? accu_flat[{flush_g, flush_p | 4'd1}]
                                        : psum_spec[flush_spec][flush_g][flush_p | 4'd1];
    wire signed [31:0] psum_q2 = swc_en ? accu_flat[{flush_g, flush_p | 4'd2}]
                                        : psum_spec[flush_spec][flush_g][flush_p | 4'd2];
    wire signed [31:0] psum_q3 = swc_en ? accu_flat[{flush_g, flush_p | 4'd3}]
                                        : psum_spec[flush_spec][flush_g][flush_p | 4'd3];
    // Rev151: RR4 lanes cũng 2 tầng (căn với pipe chính). Tầng0 register psum, tầng1
    // bias-add dùng pipe0_bias_r (cùng out_ch — 4 lane chung spec/g).
    reg signed [31:0] pipe0_q1, pipe0_q2, pipe0_q3;
    reg signed [31:0] pipe_accu_q1, pipe_accu_q2, pipe_accu_q3;
    always @(posedge aclk) begin
        pipe0_q1 <= psum_q1_t; pipe0_q2 <= psum_q2_t; pipe0_q3 <= psum_q3_t;  // Rev152: mux-tree
        pipe_accu_q1 <= pipe0_q1 + pipe0_bias_r;
        pipe_accu_q2 <= pipe0_q2 + pipe0_bias_r;
        pipe_accu_q3 <= pipe0_q3 + pipe0_bias_r;
    end
    wire signed [7:0] rr_out_q1, rr_out_q2, rr_out_q3;
    rr_unit u_rr_q1 (.clk(aclk), .rst_n(aresetn), .psum_in(pipe_accu_q1),
        .rq_scale(pipe_m0_r), .rq_shift(pipe_sh_r), .apply_relu(reg_apply_relu),
        .valid_in(pipe_accu_valid_r), .data_out(rr_out_q1), .valid_out());
    rr_unit u_rr_q2 (.clk(aclk), .rst_n(aresetn), .psum_in(pipe_accu_q2),
        .rq_scale(pipe_m0_r), .rq_shift(pipe_sh_r), .apply_relu(reg_apply_relu),
        .valid_in(pipe_accu_valid_r), .data_out(rr_out_q2), .valid_out());
    rr_unit u_rr_q3 (.clk(aclk), .rst_n(aresetn), .psum_in(pipe_accu_q3),
        .rq_scale(pipe_m0_r), .rq_shift(pipe_sh_r), .apply_relu(reg_apply_relu),
        .valid_in(pipe_accu_valid_r), .data_out(rr_out_q3), .valid_out());

    // OW (feat bit8): out DMA wide — áp cho MỌI LS_OUT (cả DW, out 64B = 4 beat).
    assign dma_out_wide = (reg_feat_en[7] | reg_feat_en[12]) & mode_layer;  // OW | DEINT → wide out

    reg wr_addr_clr;  // pulsed by FSM at start of each inference (kept for FSM compat)

    // ── OUT BRAM write pipeline (Rev36: skew-free registered write) ───────────
    // ROOT CAUSE of lệch 1 (Rev21-35): the OUT BRAM write address (from
    // combinational wb_done_cnt) and the write data (rr_out, coming through a
    // LUT/DSP datapath) reach the inferred RAMB36E2 with DIFFERENT propagation
    // delays. Vivado retiming/phys_opt inserts a register on the slow rr_out
    // path but not on the address path → address arrives 1 cycle ahead of data
    // at the physical BRAM → BRAM[1] gets g0 (data lags addr by 1). This is a
    // SYNTHESIS artifact, invisible in behavioral sim (which is why every
    // RTL-level fix Rev22-33 failed). Rev32 (register all 3 equally) kept the
    // skew; Rev34 const-0xAB masked it; Rev35 guard broke it to zeros.
    //
    // FIX: register rr_out AND the write pointer into the SAME fabric FF stage
    // right before the BRAM. The BRAM now sees two clean FF outputs (owr_data_r,
    // owr_ptr) with identical timing — Vivado has no slow combinational path to
    // retime, so addr and data stay locked. owr_ptr counts owr_en_r pulses, so
    // it is in the same +1-delayed domain as owr_data_r → impossible to skew.
    // (owr_active wire declared after `state` below to avoid forward reference.)
    // Đòn 4a': word-hóa (giữ nguyên kỷ luật Rev36 — we/data/addr cùng 1 tầng FF).
    // Byte mode: 1 lane theo owr_ptr[1:0]; quad (RR4): 4 lane, ptr nhảy +4 (word-aligned).
    assign out_bram_wr_we    = owr_en_r ? (owr_quad_r ? 4'b1111
                                                      : (4'b0001 << owr_ptr[1:0])) : 4'b0000;
    assign out_bram_wr_data  = owr_quad_r ? {owr_q_r, owr_data_r} : {4{owr_data_r}};
    assign out_bram_wr_waddr = owr_ptr[9:2];

    // ── Top-level FSM ─────────────────────────────────────────────────────────
    // States shared by both paths:
    localparam ST_IDLE        = 4'd0;
    localparam ST_WAIT_DMA    = 4'd1;
    localparam ST_PIX_CLR     = 4'd2;   // 3x3/dw: acc_clr + start swc
    localparam ST_ACT_LOAD    = 4'd3;   // 3x3/dw: load activation row from BRAM
    localparam ST_WT_LOAD     = 4'd4;   // all: load 17-byte NZ weight entry
    localparam ST_KW_SHUF     = 4'd5;   // all: assert wt_load + shuf_mode
    localparam ST_COMPUTE     = 4'd6;   // all: wait MAC pipeline
    localparam ST_FLUSH_MAC   = 4'd7;   // all: drain MAC pipeline
    localparam ST_FLUSH_RR    = 4'd8;   // 3x3/dw: flush rr_unit
    localparam ST_DONE        = 4'd9;
    // 1x1-only states:
    localparam ST_LOAD_ACT_1X1= 4'd10;  // 1x1: load P=16 activations
    localparam ST_SHUF_WAIT   = 4'd11;  // 1x1: 1-cycle SHUF_LOAD → SHUF_HOLD
    localparam ST_WRITEBACK   = 4'd12;  // 1x1: flush all 256 rr outputs
    localparam ST_LOAD_RQ     = 4'd13;  // 1x1: load per-channel m0/sh/bias from WT BRAM tail
    localparam ST_WT_LOAD_W   = 4'd14;  // Rev139 c3: wide WT load (16B/cyc, 64B-aligned entries)
    localparam ST_LOAD_ACT_W  = 4'd15;  // Rev139d c3: wide ACT load (2-3 word/entry + barrel)
    // Rev172: graphop mode FSM chính đứng ST_IDLE (không tới ST_DONE) → LS chờ k_ap_done sẽ TREO.
    // graphop_done (pulse 1-cyc từ engine) thay thế — LS passthrough (l_gop) nhận và về LS_DONE.
    assign k_done_pulse_out = (state == ST_DONE) | (graphop_mode & graphop_done);
    assign lt_graphop_out   = graphop_mode;       // Rev172: → LS passthrough
    // act_uram word-port mux: C3W (ST_LOAD_ACT_W) / W11 (ST_LOAD_ACT_1X1) / WDW (ST_ACT_LOAD dw)
    assign w11_ld       = w11 && (state == ST_LOAD_ACT_1X1);
    assign awsel_active = (state == ST_LOAD_ACT_W) || w11_ld
                       || (wdw && (state == ST_ACT_LOAD));

    // Rev36: high during OUT BRAM writeback (1x1=ST_WRITEBACK, 3x3/dw=ST_FLUSH_RR)
    // (reg [3:0] state declared earlier, near ap_done_latch, for forward use at dbg_kernel_state)
    wire owr_active = (state == ST_WRITEBACK) || (state == ST_FLUSH_RR);
    reg [7:0]  flush_mac_cnt;
    reg [5:0]  act_ld_cnt;
    reg [6:0]  wt_ld_cnt;   // B-w: widened [4:0]→[6:0] (NS=12 entry byte count up to ~51)
    integer    ksh_s, ksh_g;   // B-w: ST_KW_SHUF slot-0 routing loop indices (NS×G)
    reg [3:0]  wt_slot_cnt;  // which NZ slot (0..n_cfg-1) is being loaded
    reg [7:0]  ic_blk_cnt;
    reg [3:0]  flush_k;      // 1x1 sequential: current NZ slot being processed (0..n_cfg-1)
    reg [9:0]  flush_cnt;   // B-w: widened [8:0]→[9:0] (writeback NS*64 outputs; NS=12→768)
    // flush_spec declared earlier (near flush_g) for forward use at psum_1x1/out_ch

    // n_ic_blks = ceil(Cin / M=16) for 1x1; reg_n_config still used for 3x3/dw
    wire [7:0] n_ic_blks_1x1 = (reg_cin[3:0] != 4'd0)
                                 ? reg_cin[11:4] + 8'd1
                                 : reg_cin[11:4];
    wire [7:0] n_ic_blks = swc_en ? {3'b0, reg_n_config} : n_ic_blks_1x1;

    // Total NZ weight bytes for 1x1: n_ic_blks × n_cfg × 17
    // Used to compute offset of m0/sh/bias in WT BRAM.
    // Rev139: ×ESZ (was hardcoded ×17 — only correct at NS=4!). c3: entries are 64B-aligned.
    // Rev139c: dùng ncfg_esz_r (registered) — bỏ chuỗi mult khỏi path vào wt_bram_rd_addr.
    wire [11:0] wt_nz_bytes_1x1 = c3_act ? {1'b0, reg_n_config, 6'b0}
                                         : (n_ic_blks_1x1 * ncfg_esz_r);
    // conv3x3 weight bytes = Cin × 9 taps × n_cfg × 17 (channel-major then tap).
    // dwconv weight bytes = 9 taps × n_cfg × 17 (4 channels live in the 4 groups, no
    // Cin dimension). Requant tail (7B/ch × Cout) follows, swc loads it from here.
    wire [11:0] wt_nz_bytes_swc = (dw_mode ? 12'd9 : {4'd0, reg_cin[7:0]} * 12'd9)
                                  * ncfg_esz_r;   // Rev139: ×ESZ; Rev139c: registered product

    // Requant load state registers
    reg [7:0]  rq_load_ch;   // which channel is being loaded (0..Cout-1)
    reg [2:0]  rq_load_byte; // byte offset within one channel's params (0..6)
    // Temp accumulator for 4-byte bias assembly
    reg [31:0] rq_bias_tmp;

    reg dma_nxt_req_r;
    assign dma_nxt_req_o = dma_nxt_req_r;

    // kernel_done latch: set on 1-cycle pulse from swc, cleared when FSM reads it
    reg kernel_done_latch;

    always @(posedge aclk) begin
        if (!aresetn) begin
            state           <= ST_IDLE;
            ap_done_r       <= 1'b0; ap_idle_r       <= 1'b1;
            wtw_cnt         <= 7'd0; wtw_active      <= 1'b0;
            c3_idxhw        <= 21'd0; c3_rd_addr     <= 21'd0;
            aw_cnt          <= 3'd0; aw_base         <= 21'd0;
            aw_w0 <= 128'd0; aw_w1 <= 128'd0; aw_w2 <= 128'd0;
            w11_addr_r      <= 17'd0;
            owr_q_r         <= 24'd0; owr_quad_r     <= 1'b0;
            rq_dirty        <= 1'b1;   // sau reset: bắt buộc nạp RQ lần đầu
            ap_done_latch   <= 1'b0;
            graphop_start_r <= 1'b0; graphop_done_latch <= 1'b0;   // Rev170 graphop
            wt_load_r       <= 1'b0; acc_clr_r       <= 1'b0;
            shuf_mode_r     <= SHUF_HOLD;
            swc_start       <= 1'b0;
            flush_mac_cnt   <= 8'd0;
            act_ld_cnt      <= 6'd0;
            wt_ld_cnt       <= 7'd0;
            wt_slot_cnt     <= 4'd0;
            ic_blk_cnt      <= 8'd0;
            flush_k         <= 4'd0;
            flush_cnt       <= 9'd0;
            wb_done_cnt     <= 9'd0;
            out_bram_cnt    <= 8'd0;
            owr_ptr         <= 10'd0;
            owr_data_r      <= 8'd0;
            owr_en_r        <= 1'b0;
            flush_g         <= 2'd0; flush_p         <= 4'd0;
            flush_spec      <= 2'd0;
            flush_valid     <= 1'b0;
            pp_in_sel       <= 1'b0; pp_out_sel      <= 1'b0;
            pp_wt_sel       <= 1'b0;
            in_bram_rd_addr <= 10'd0;
            wt_bram_rd_addr <= 12'd0;
            wr_addr_clr     <= 1'b0;
            tc_start        <= 1'b0; tc_next         <= 1'b0;
            dma_nxt_req_r   <= 1'b0;
            kernel_done_latch <= 1'b0;
            mf_kh           <= 2'd0; mf_kw           <= 2'd0;
            swc_ic          <= 8'd0;
            swc_grp         <= 2'd0;
            begin : wt_rst
                integer ri, rs, rg;
                for (ri = 0; ri < 16; ri = ri + 1) begin
                    wt_rd_idx[ri] <= 4'd0;
                    for (rs = 0; rs < NS; rs = rs + 1)
                        for (rg = 0; rg < `G; rg = rg + 1)
                            wt_rd_arr[rs][rg][ri] <= 8'sd0;
                end
            end
            begin : rq_rst
                integer rqi;
                for (rqi = 0; rqi < 256; rqi = rqi + 1) begin
                    rq_m0[rqi]   <= 16'h4000;
                    rq_sh[rqi]   <= 5'd15;
                    rq_bias[rqi] <= 32'sd0;
                end
            end
            rq_load_ch   <= 8'd0;
            rq_load_byte <= 3'd0;
            rq_bias_tmp  <= 32'd0;
            new_col[0] <= 8'sd0; new_col[1] <= 8'sd0;
            new_col[2] <= 8'sd0; new_col[3] <= 8'sd0;
            act_g0[ 0]<= 8'sd0; act_g0[ 1]<= 8'sd0; act_g0[ 2]<= 8'sd0; act_g0[ 3]<= 8'sd0;
            act_g0[ 4]<= 8'sd0; act_g0[ 5]<= 8'sd0; act_g0[ 6]<= 8'sd0; act_g0[ 7]<= 8'sd0;
            act_g0[ 8]<= 8'sd0; act_g0[ 9]<= 8'sd0; act_g0[10]<= 8'sd0; act_g0[11]<= 8'sd0;
            act_g0[12]<= 8'sd0; act_g0[13]<= 8'sd0; act_g0[14]<= 8'sd0; act_g0[15]<= 8'sd0;
            act_g1[ 0]<= 8'sd0; act_g1[ 1]<= 8'sd0; act_g1[ 2]<= 8'sd0; act_g1[ 3]<= 8'sd0;
            act_g1[ 4]<= 8'sd0; act_g1[ 5]<= 8'sd0; act_g1[ 6]<= 8'sd0; act_g1[ 7]<= 8'sd0;
            act_g1[ 8]<= 8'sd0; act_g1[ 9]<= 8'sd0; act_g1[10]<= 8'sd0; act_g1[11]<= 8'sd0;
            act_g1[12]<= 8'sd0; act_g1[13]<= 8'sd0; act_g1[14]<= 8'sd0; act_g1[15]<= 8'sd0;
            act_g2[ 0]<= 8'sd0; act_g2[ 1]<= 8'sd0; act_g2[ 2]<= 8'sd0; act_g2[ 3]<= 8'sd0;
            act_g2[ 4]<= 8'sd0; act_g2[ 5]<= 8'sd0; act_g2[ 6]<= 8'sd0; act_g2[ 7]<= 8'sd0;
            act_g2[ 8]<= 8'sd0; act_g2[ 9]<= 8'sd0; act_g2[10]<= 8'sd0; act_g2[11]<= 8'sd0;
            act_g2[12]<= 8'sd0; act_g2[13]<= 8'sd0; act_g2[14]<= 8'sd0; act_g2[15]<= 8'sd0;
            act_g3[ 0]<= 8'sd0; act_g3[ 1]<= 8'sd0; act_g3[ 2]<= 8'sd0; act_g3[ 3]<= 8'sd0;
            act_g3[ 4]<= 8'sd0; act_g3[ 5]<= 8'sd0; act_g3[ 6]<= 8'sd0; act_g3[ 7]<= 8'sd0;
            act_g3[ 8]<= 8'sd0; act_g3[ 9]<= 8'sd0; act_g3[10]<= 8'sd0; act_g3[11]<= 8'sd0;
            act_g3[12]<= 8'sd0; act_g3[13]<= 8'sd0; act_g3[14]<= 8'sd0; act_g3[15]<= 8'sd0;
        end else begin
            ap_done_r     <= 1'b0;
            wt_load_r     <= 1'b0;
            acc_clr_r     <= 1'b0;
            shuf_mode_r   <= SHUF_HOLD;
            swc_start     <= 1'b0;
            tc_start      <= 1'b0;
            tc_next       <= 1'b0;
            flush_valid   <= 1'b0;
            dma_nxt_req_r <= 1'b0;
            wr_addr_clr   <= 1'b0;
            graphop_start_r <= 1'b0;   // Rev170: default low (pulse 1 cyc ở ST_IDLE)
            if (graphop_done) graphop_done_latch <= 1'b1;  // latch graphop xong (host poll)
            // Latch kernel_done pulse so FSM can't miss it
            if (swc_kernel_done) kernel_done_latch <= 1'b1;

            case (state)
                ST_IDLE: begin
                    ap_idle_r <= 1'b1;
                    // Rev170 GRAPHOP: conv_type=4 → chạy graphop_engine ĐỘC LẬP, FSM chính đứng
                    // yên ở ST_IDLE (không vào ST_WAIT_DMA). ap_done qua graphop_done_latch.
                    if (ap_start_eff && graphop_mode) begin
                        ap_idle_r          <= 1'b0;
                        graphop_start_r    <= 1'b1;
                        graphop_done_latch <= 1'b0;
                    end else if (ap_start_eff) begin
                        ap_idle_r     <= 1'b0;
                        ap_done_latch <= 1'b0;  // clear done latch on new run
                        tc_start      <= 1'b1;
                        wr_addr_clr   <= 1'b1;
                        ic_blk_cnt    <= 8'd0;
                        state         <= ST_WAIT_DMA;
                    end
                end

                ST_WAIT_DMA: begin
                    if (dma_in_done_eff) begin
                        // 3x3/dw: flip on every tile (each tile is a real DMA).
                        // 1x1-old: flip only on ic_blk=0. (Rev139: R1 term removed — pruned.)
                        if (swc_en || (ic_blk_cnt == 8'd0)) begin
                            pp_in_sel <= ~pp_in_sel;
                            pp_wt_sel <= ~pp_wt_sel;
                        end
                        if (swc_en) begin
                            // 3x3/dw path: load per-channel requant from WT tail first
                            // (mirror 1x1). Tail starts after 9×n_cfg×ESZ weight bytes.
                            rq_load_ch      <= 8'd0;
                            rq_load_byte    <= 3'd0;
                            flush_cnt       <= 9'd0;
                            wt_bram_rd_addr <= wt_nz_bytes_swc;
                            state           <= ST_LOAD_RQ;
                        end else begin
                            // Rev139 fastkick (feat bit2): rq regs persist across the LS ic-loop —
                            // load them only on the FIRST kick of each (oc,pos) (keep_psum=0);
                            // subsequent kicks reuse them (same oc-block → same requant params).
                            // Đòn 4a' RQD (bit7): wt_uram không đổi từ lần RQ trước (cùng oc-block,
                            // WT-stationary) → skip nạp lại 336B. RQ chỉ còn 1×/oc-block thay vì mỗi pos.
                            if (ic_blk_cnt == 8'd0 && !(fastkick && keep_psum)
                                && !(rqd_en && !rq_dirty)) begin
                                // First DMA: load requant params from WT BRAM tail first
                                rq_load_ch      <= 8'd0;
                                rq_load_byte    <= 3'd0;
                                flush_cnt       <= 9'd0;
                                // BRAM addr for first m0 byte: wt_nz_bytes_1x1
                                wt_bram_rd_addr <= wt_nz_bytes_1x1;
                                state           <= ST_LOAD_RQ;
                            end else begin
                                // Subsequent ic_blks (or rq-skipped first): WT resident, no new DMA.
                                // Do NOT clear accumulator — psum must accumulate across ic_blks.
                                acc_clr_r       <= (ic_blk_cnt == 8'd0) ? ~keep_psum : 1'b0;
                                wt_ld_cnt       <= 7'd0;
                                wt_slot_cnt     <= 4'd0;
                                // Rev139c INCREMENTAL: entries/M-blocks LIÊN TIẾP trong chunk →
                                // ic_blk 0 = offset 0; ic_blk>0 = addr đang giữ (S+ESZ-1 của slot
                                // cuối M-block trước) + 1 = base M-block kế. Bỏ chuỗi mult DSP.
                                wt_bram_rd_addr <= (ic_blk_cnt == 8'd0) ? 12'd0
                                                                        : (wt_bram_rd_addr + 12'd1);
                                if (c3_act) begin
                                    wtw_cnt    <= 7'd0;
                                    wtw_active <= 1'b1;
                                    state      <= ST_WT_LOAD_W;
                                end else
                                    state      <= ST_WT_LOAD;
                            end
                        end
                    end
                end

                // ── 3x3/dw path states ─────────────────────────────────────────
                ST_PIX_CLR: begin
                    // clear psum only on the first chunk (keep_psum=0); channel-tiling
                    // chunks after the first accumulate onto the running psum.
                    acc_clr_r         <= ~keep_psum;
                    swc_start         <= 1'b1;
                    act_ld_cnt        <= 6'd0;
                    in_bram_rd_addr   <= 10'd0;  // channel 0, tap (0,0)
                    mf_kh             <= 2'd0;  // start at kernel tap (0,0)
                    mf_kw             <= 2'd0;
                    swc_ic            <= 8'd0;  // start at input channel 0
                    swc_grp           <= 2'd0;  // dwconv: start loading group/channel 0
                    aw_cnt            <= 3'd0;  // Rev142 WDW: hygiene (barrel counter)
                    dma_nxt_req_r     <= tc_valid & ~tc_last;
                    kernel_done_latch <= 1'b0;  // clear before new pixel scan
                    state             <= ST_ACT_LOAD;
                end

                // ── dwconv: load 4 DIFFERENT channels (base+g) into act_g0..g3 ──
                // The 4 groups are 4 output(=input) channels (depthwise, no cross-channel
                // sum). Read 16 positions of channel swc_grp at g*tile_chw + tap_base, then
                // advance swc_grp. conv3x3 path (below) instead copies act_g0→g1/g2/g3.
                ST_ACT_LOAD: if (dw_mode) begin
                    // ── Rev142 WDW: window 16B = 2 word + barrel (mirror C3W timing:
                    //    cnt0 chốt base, addr wire cnt1-2, w0@3, w1@4, barrel+advance@5). ──
                    if (wdw) begin : wdw_blk
                        integer dwp;
                        reg [5:0] dw_bo;
                        aw_cnt <= aw_cnt + 3'd1;
                        if (aw_cnt == 3'd0)
                            aw_base <= ls_act_off_eff + {11'd0, in_bram_rd_addr};   // Rev200: +resin_off (gated)
                        if (aw_cnt == 3'd3) aw_w0 <= act_w_data;
                        if (aw_cnt == 3'd4) aw_w1 <= act_w_data;
                        if (aw_cnt == 3'd5) begin
                            for (dwp = 0; dwp < 16; dwp = dwp + 1) begin : wdw_lane
                                dw_bo = aw_base[3:0] + dwp[5:0];
                                case (swc_grp)
                                    2'd0: act_g0[dwp] <= $signed(aw_win[dw_bo*8 +: 8]);
                                    2'd1: act_g1[dwp] <= $signed(aw_win[dw_bo*8 +: 8]);
                                    2'd2: act_g2[dwp] <= $signed(aw_win[dw_bo*8 +: 8]);
                                    2'd3: act_g3[dwp] <= $signed(aw_win[dw_bo*8 +: 8]);
                                endcase
                            end
                            aw_cnt <= 3'd0;
                            if (swc_grp < 2'd3) begin
                                swc_grp         <= swc_grp + 2'd1;
                                in_bram_rd_addr <= (swc_grp + 2'd1) * tile_chw
                                                   + mf_kh * reg_win[9:0] + {8'd0, mf_kw};
                            end else begin
                                swc_grp         <= 2'd0;
                                wt_bram_rd_addr <= (({10'd0, mf_kh} * 12'd3) + {10'd0, mf_kw})
                                                   * ({7'd0, reg_n_config} * ESZ[11:0]);
                                wt_ld_cnt       <= 7'd0;
                                wt_slot_cnt     <= 4'd0;
                                state           <= ST_WT_LOAD;
                            end
                        end
                    end else begin
                    in_bram_rd_addr <= in_bram_rd_addr + 10'd1;
                    // Rev139 RD_LAT=3: capture window 2..17 → 3..18 (+1)
                    if (act_ld_cnt >= 6'd3 && act_ld_cnt <= 6'd18) begin
                        case (swc_grp)
                            2'd0: act_g0[act_ld_cnt-6'd3] <= $signed(in_bram_rd_data);
                            2'd1: act_g1[act_ld_cnt-6'd3] <= $signed(in_bram_rd_data);
                            2'd2: act_g2[act_ld_cnt-6'd3] <= $signed(in_bram_rd_data);
                            2'd3: act_g3[act_ld_cnt-6'd3] <= $signed(in_bram_rd_data);
                        endcase
                    end
                    act_ld_cnt <= act_ld_cnt + 6'd1;
                    if (act_ld_cnt == 6'd18) begin
                        if (swc_grp < 2'd3) begin
                            swc_grp         <= swc_grp + 2'd1;
                            act_ld_cnt      <= 6'd0;
                            // next channel window at this tap: (g+1)*tile_chw + tap_base
                            in_bram_rd_addr <= (swc_grp + 2'd1) * tile_chw
                                               + mf_kh * reg_win[9:0] + {8'd0, mf_kw};
                        end else begin
                            swc_grp         <= 2'd0;
                            // dwconv weights: 9 taps, entry = idx + NS*G bytes (ESZ). tap base only.
                            wt_bram_rd_addr <= (({10'd0, mf_kh} * 12'd3) + {10'd0, mf_kw})
                                               * ({7'd0, reg_n_config} * ESZ[11:0]);
                            wt_ld_cnt       <= 7'd0;
                            wt_slot_cnt     <= 4'd0;
                            state           <= ST_WT_LOAD;
                        end
                    end
                    end
                end else begin
                    in_bram_rd_addr <= in_bram_rd_addr + 10'd1;
                    // Rev139 RD_LAT=3: capture window 2..17 → 3..18 (+1)
                    if (act_ld_cnt >= 6'd3 && act_ld_cnt <= 6'd18)
                        act_g0[act_ld_cnt - 6'd3] <= $signed(in_bram_rd_data);
                    act_ld_cnt <= act_ld_cnt + 6'd1;
                    if (act_ld_cnt == 6'd18) begin
                        act_g1[ 0]<=act_g0[ 0]; act_g1[ 1]<=act_g0[ 1]; act_g1[ 2]<=act_g0[ 2]; act_g1[ 3]<=act_g0[ 3];
                        act_g1[ 4]<=act_g0[ 4]; act_g1[ 5]<=act_g0[ 5]; act_g1[ 6]<=act_g0[ 6]; act_g1[ 7]<=act_g0[ 7];
                        act_g1[ 8]<=act_g0[ 8]; act_g1[ 9]<=act_g0[ 9]; act_g1[10]<=act_g0[10]; act_g1[11]<=act_g0[11];
                        // p15: act_g0[15] is assigned $signed(in_bram_rd_data) THIS same
                        // cycle (non-blocking → old value). Copy the NEW value directly so
                        // groups 1-3 don't get a 1-tap-stale position 15 (was: only group 0
                        // correct → oc1/2/3 p=15 wrong; masked until multi-out-ch testing).
                        act_g1[12]<=act_g0[12]; act_g1[13]<=act_g0[13]; act_g1[14]<=act_g0[14]; act_g1[15]<=$signed(in_bram_rd_data);
                        act_g2[ 0]<=act_g0[ 0]; act_g2[ 1]<=act_g0[ 1]; act_g2[ 2]<=act_g0[ 2]; act_g2[ 3]<=act_g0[ 3];
                        act_g2[ 4]<=act_g0[ 4]; act_g2[ 5]<=act_g0[ 5]; act_g2[ 6]<=act_g0[ 6]; act_g2[ 7]<=act_g0[ 7];
                        act_g2[ 8]<=act_g0[ 8]; act_g2[ 9]<=act_g0[ 9]; act_g2[10]<=act_g0[10]; act_g2[11]<=act_g0[11];
                        act_g2[12]<=act_g0[12]; act_g2[13]<=act_g0[13]; act_g2[14]<=act_g0[14]; act_g2[15]<=$signed(in_bram_rd_data);
                        act_g3[ 0]<=act_g0[ 0]; act_g3[ 1]<=act_g0[ 1]; act_g3[ 2]<=act_g0[ 2]; act_g3[ 3]<=act_g0[ 3];
                        act_g3[ 4]<=act_g0[ 4]; act_g3[ 5]<=act_g0[ 5]; act_g3[ 6]<=act_g0[ 6]; act_g3[ 7]<=act_g0[ 7];
                        act_g3[ 8]<=act_g0[ 8]; act_g3[ 9]<=act_g0[ 9]; act_g3[10]<=act_g0[10]; act_g3[11]<=act_g0[11];
                        act_g3[12]<=act_g0[12]; act_g3[13]<=act_g0[13]; act_g3[14]<=act_g0[14]; act_g3[15]<=$signed(in_bram_rd_data);
                        // Per-(channel,tap) weights: global tap index = ic*9 + (kh*3+kw).
                        // base = (ic*9 + tap_idx) * n_cfg * ESZ. Host lays out
                        // Cin × 9 taps × n_cfg × ESZ bytes (channel-major, then tap).
                        wt_bram_rd_addr <= (({4'd0, swc_ic} * 12'd9)
                                            + ({10'd0, mf_kh} * 12'd3) + {10'd0, mf_kw})
                                           * ({7'd0, reg_n_config} * ESZ[11:0]);
                        wt_ld_cnt       <= 7'd0;
                        wt_slot_cnt     <= 4'd0;
                        state           <= ST_WT_LOAD;
                    end
                end

                // ── Shared weight load (both paths) ────────────────────────────
                // Loads n_cfg NZ entries sequentially into slots wt_slot_cnt=0..n_cfg-1.
                // Each entry = 17 bytes: idx(1) + 4SPEC x 4groups x 1byte.
                // wt_ld_cnt counts bytes within one entry (0..18, with 2-cycle BRAM latency).
                // After all n_cfg slots filled, transitions to ST_KW_SHUF (3x3/dw) or
                // ST_LOAD_ACT_1X1 (1x1).
                // B-w: parameterized load into wt_rd_arr[spec][grp][slot]. Entry = idx + NS*G weights.
                //   3x3/dw: idx@cnt=2, weights@cnt 3..(NS*G+2). byte b=cnt-3 → spec=b>>2, grp=b[1:0].
                //   1x1   : idx@cnt=3, weights@cnt 4..(NS*G+3). byte b=cnt-4. addr uses ESZ (=17@NS4).
                ST_WT_LOAD: begin
                    if (swc_en) begin
                        wt_bram_rd_addr <= wt_bram_rd_addr + 12'd1;
                        // Rev139 RD_LAT=3: idx@2→3, weights 3..NS*G+2 → 4..NS*G+3
                        if (wt_ld_cnt == 7'd3)
                            wt_rd_idx[wt_slot_cnt] <= wt_bram_rd_data[3:0];
                        else if (wt_ld_cnt >= 7'd4 && wt_ld_cnt <= (NS*`G + 3))
                            wt_rd_arr[(wt_ld_cnt-7'd4) >> 2][(wt_ld_cnt-7'd4) & 7'd3][wt_slot_cnt]
                                <= $signed(wt_bram_rd_data);
                        if (wt_ld_cnt == (NS*`G + 3)) begin
                            if ({1'b0, wt_slot_cnt} < (reg_n_config - 5'd1)) begin
                                wt_slot_cnt <= wt_slot_cnt + 4'd1; wt_ld_cnt <= 7'd0;
                            end else begin
                                wt_slot_cnt <= 4'd0; state <= ST_KW_SHUF;
                            end
                        end else wt_ld_cnt <= wt_ld_cnt + 7'd1;
                    end else begin
                        // 1x1: Rev139c INCREMENTAL addr — entries liên tiếp (S+c-1 tại cycle c):
                        // entry-transition đã đặt addr = slot-base; +1 tại c=1..ESZ-1; giữ khi wait.
                        if (wt_ld_cnt != 7'd0 && wt_ld_cnt < ESZ)
                            wt_bram_rd_addr <= wt_bram_rd_addr + 12'd1;
                        // Rev139 RD_LAT=3: idx@3→4, weights 4..NS*G+3 → 5..NS*G+4
                        if (wt_ld_cnt == 7'd4)
                            wt_rd_idx[wt_slot_cnt] <= wt_bram_rd_data[3:0];
                        else if (wt_ld_cnt >= 7'd5 && wt_ld_cnt <= (NS*`G + 4))
                            wt_rd_arr[(wt_ld_cnt-7'd5) >> 2][(wt_ld_cnt-7'd5) & 7'd3][wt_slot_cnt]
                                <= $signed(wt_bram_rd_data);
                        if (wt_ld_cnt == (NS*`G + 4)) begin
                            if ({1'b0, wt_slot_cnt} < (reg_n_config - 5'd1)) begin
                                wt_slot_cnt <= wt_slot_cnt + 4'd1; wt_ld_cnt <= 7'd0;
                                wt_bram_rd_addr <= wt_bram_rd_addr + 12'd1;  // → base slot kế (S+ESZ)
                            end else begin
                                wt_slot_cnt     <= 4'd0;
                                act_ld_cnt      <= 6'd0;
                                in_bram_rd_addr <= 10'd0;
                                state           <= ST_LOAD_ACT_1X1;
                            end
                        end else wt_ld_cnt <= wt_ld_cnt + 7'd1;
                    end
                end

                // ── Rev139 c3: WIDE WT load — 64B-aligned entries, 16B/cycle from wt_uram
                //    word port (RD_LAT=2 via rd_wdata). Entry = 4 words: w0 = idx + wt[0..14],
                //    w1 = wt[15..30], w2 = wt[31..46], w3 = wt[47] + pad. Fixed byte→reg routing
                //    per word phase (no rotating mux). n_cfg entries → n_cfg*4 words + 2 drain. ──
                ST_WT_LOAD_W: begin
                    wtw_cnt <= wtw_cnt + 7'd1;
                    if (wtw_cnt >= 7'd2) begin : wtw_consume
                        integer wb;
                        reg [6:0] wk;   // consumed word index (function of counter, not state)
                        wk = wtw_cnt - 7'd2;
                        case (wk[1:0])
                            2'd0: begin
                                wt_rd_idx[wk[5:2]] <= wt_w_data[3:0];
                                for (wb = 1; wb < 16; wb = wb + 1)
                                    if (wb-1 < NS*`G)
                                        wt_rd_arr[(wb-1)>>2][(wb-1)&3][wk[5:2]]
                                            <= $signed(wt_w_data[wb*8 +: 8]);
                            end
                            2'd1: for (wb = 0; wb < 16; wb = wb + 1)
                                    if (wb+15 < NS*`G)
                                        wt_rd_arr[(wb+15)>>2][(wb+15)&3][wk[5:2]]
                                            <= $signed(wt_w_data[wb*8 +: 8]);
                            2'd2: for (wb = 0; wb < 16; wb = wb + 1)
                                    if (wb+31 < NS*`G)
                                        wt_rd_arr[(wb+31)>>2][(wb+31)&3][wk[5:2]]
                                            <= $signed(wt_w_data[wb*8 +: 8]);
                            2'd3: if (47 < NS*`G)
                                    wt_rd_arr[47>>2][47&3][wk[5:2]]
                                        <= $signed(wt_w_data[7:0]);
                        endcase
                        if (wk == {reg_n_config - 5'd1, 2'b11}) begin   // last word = n_cfg*4 - 1
                            wtw_active      <= 1'b0;
                            act_ld_cnt      <= 6'd0;
                            in_bram_rd_addr <= 10'd0;
                            // c3 ACT ch-offset = idx * HW. Sparse-c3: idx = wt_rd_idx[slot]
                            // (kept channel), NOT slot. Dense: wt_rd_idx[0]=0 → 0 (bit-exact
                            // với accumulator cũ). slot 0 dùng wt_rd_idx[0]. Rev150.
                            c3_idxhw        <= wt_rd_idx[0] * c3_hw;
                            aw_cnt          <= 3'd0;
                            state           <= c3w ? ST_LOAD_ACT_W : ST_LOAD_ACT_1X1;
                        end
                    end
                end

                // ── 1x1 only: load activations ─────────────────────────────────
                // ACT BRAM layout: CHW flat — BRAM[ic*P + p] = act[ic, pixel_p]
                // For ic_blk b, NZ slot 0 with position wt_rd_idx[0]:
                //   ic_abs = b*M + wt_rd_idx[0]
                //   base_addr = b*M*P + wt_rd_idx[0]*P = ic_abs * P
                //   read_addr = base_addr + act_ld_cnt (for P=16 consecutive pixels)
                ST_LOAD_ACT_1X1: begin
                    // ── Đòn 4a W11: 1 word-read thay 16 byte-read. cnt0 chốt addr (register
                    //    → URAM path nhẹ), cnt1 issue (addr từ w11_addr_r), rd_word@2, rd_wdata@3
                    //    → capture 16 lane tại cnt3 (block act_cap dưới). 4 cycle thay 20. ──
                    if (w11) begin
                        if (act_ld_cnt == 6'd0)
                            w11_addr_r <= ls_act_off_eff[20:4]   // Rev200: +resin_off (aligned 256)
                                          + {11'd0, ic_blk_cnt[1:0], wt_rd_idx[flush_k]};
                        if (act_ld_cnt == 6'd3) begin
                            shuf_mode_r <= SHUF_LOAD;
                            act_ld_cnt  <= 6'd0;
                            state       <= ST_SHUF_WAIT;
                        end else begin
                            act_ld_cnt <= act_ld_cnt + 6'd1;
                        end
                    end else begin
                    if (act_ld_cnt < 8'd16)
                        in_bram_rd_addr <= {ic_blk_cnt[1:0], wt_rd_idx[flush_k], act_ld_cnt[3:0]};
                    // Rev139 c3: raw-CHW plane addressing — base = pixel_base(kick, from LS)
                    // + idx*HW (c3_idxhw = wt_rd_idx[slot]*HW, sparse-capable); step = stride.
                    if (c3_act) begin
                        if (act_ld_cnt == 6'd0)
                            c3_rd_addr <= ls_act_uram_off + c3_idxhw;
                        else if (act_ld_cnt < 6'd16)
                            c3_rd_addr <= c3_rd_addr + (ls_c3_stride2 ? 21'd2 : 21'd1);
                    end
                    // Rev139 RD_LAT=3: last capture at 18→19
                    if (act_ld_cnt == 8'd19) begin
                        shuf_mode_r <= SHUF_LOAD;
                        act_ld_cnt  <= 6'd0;
                        state       <= ST_SHUF_WAIT;
                    end else begin
                        act_ld_cnt <= act_ld_cnt + 6'd1;
                    end
                    end
                end

                // ── Rev139d C3W: wide ACT load — cycle 0 chốt base; issue word cnt-1 tại
                //    cycle 1..N (addr = wire từ aw_base+aw_cnt-1); data RD_LAT=2 → w_k tại
                //    cycle k+3; assemble barrel 16 lane (+pad mask) 1 cycle sau word cuối. ──
                ST_LOAD_ACT_W: begin : aw_blk
                    integer awp;
                    reg [5:0] aw_bo;
                    aw_cnt <= aw_cnt + 3'd1;
                    if (aw_cnt == 3'd0)
                        aw_base <= ls_act_uram_off + c3_idxhw;
                    if (aw_cnt == 3'd3) aw_w0 <= act_w_data;
                    if (aw_cnt == 3'd4) aw_w1 <= act_w_data;
                    if (aw_cnt == 3'd5) aw_w2 <= act_w_data;   // stride1: ghi thừa, vô hại
                    if (aw_cnt == (ls_c3_stride2 ? 3'd6 : 3'd5)) begin
                        for (awp = 0; awp < 16; awp = awp + 1) begin : aw_lane
                            reg aw_msk;
                            aw_bo  = aw_base[3:0] + (ls_c3_stride2 ? (awp + awp) : awp);
                            aw_msk = ls_c3_zero
                                     || (ls_c3_maskl && (awp == 0))
                                     || (ls_c3_maskr && (awp == {28'd0, ls_c3_lastp}));
                            act_g0[awp] <= aw_msk ? 8'sd0 : $signed(aw_win[aw_bo*8 +: 8]);
                            act_g1[awp] <= aw_msk ? 8'sd0 : $signed(aw_win[aw_bo*8 +: 8]);
                            act_g2[awp] <= aw_msk ? 8'sd0 : $signed(aw_win[aw_bo*8 +: 8]);
                            act_g3[awp] <= aw_msk ? 8'sd0 : $signed(aw_win[aw_bo*8 +: 8]);
                        end
                        shuf_mode_r <= SHUF_LOAD;
                        aw_cnt      <= 3'd0;
                        state       <= ST_SHUF_WAIT;
                    end
                end

                ST_SHUF_WAIT: begin
                    shuf_mode_r <= SHUF_HOLD;
                    state       <= ST_KW_SHUF;
                end

                // ── Shared: assert wt_load + shuf_mode ────────────────────────
                ST_KW_SHUF: begin
                    // swc: full window re-loaded each tap (in_bram addr = kh*Win+kw) →
                    // always SHUF_LOAD (no SHIFT). 1x1: swc_shuf_mode_mux is already LOAD.
                    shuf_mode_r  <= swc_en ? SHUF_LOAD : swc_shuf_mode_mux;
                    new_col[0]   <= act_g0[0];
                    new_col[1]   <= act_g1[0];
                    new_col[2]   <= act_g2[0];
                    new_col[3]   <= act_g3[0];
                    // 1x1 sequential: route current NZ slot (flush_k) into slot 0 so
                    // weight_decoder (n_cfg=1) emits the correct channel index and weights.
                    if (!swc_en) begin
                        wt_rd_idx[0] <= wt_rd_idx[flush_k];
                        // B-w: route current NZ slot (flush_k) → slot 0 for all NS SPEC × G groups
                        for (ksh_s = 0; ksh_s < NS; ksh_s = ksh_s + 1)
                            for (ksh_g = 0; ksh_g < `G; ksh_g = ksh_g + 1)
                                wt_rd_arr[ksh_s][ksh_g][0] <= wt_rd_arr[ksh_s][ksh_g][flush_k];
                    end
                    wt_load_r    <= 1'b1;
                    flush_mac_cnt<= 8'd0;
                    // Rev167 C3FAST (đo STPROF: COMPUTE 5%, FLUSH 19%, LD_ACT 33%): entry k đã
                    // "phóng" (wt_load + act trong depth_shuffle) → pipeline tự chảy vào acc.
                    // Còn entry → nạp act k+1 NGAY (che pipeline, KHÔNG drain giữa entries;
                    // FLUSH_MAC 1 lần sau entry cuối). An toàn: act mới đè depth_shuffle sau
                    // ≥7 cyc (LOAD_ACT_W) > mul-pipeline 4; slot-0 route đè sau ≥9 cyc.
                    if (seqfast && !swc_en && ({1'b0, flush_k} < reg_n_config - 5'd1)) begin
                        flush_k    <= flush_k + 4'd1;
                        c3_idxhw   <= wt_rd_idx[flush_k + 4'd1] * c3_hw;
                        act_ld_cnt <= 6'd0;
                        aw_cnt     <= 3'd0;
                        state      <= c3w ? ST_LOAD_ACT_W : ST_LOAD_ACT_1X1;
                    end else
                        state    <= ST_COMPUTE;
                end

                ST_COMPUTE: begin
                    state <= ST_FLUSH_MAC;
                end

                // ── Shared: drain MAC pipeline ─────────────────────────────────
                ST_FLUSH_MAC: begin
                    flush_mac_cnt <= flush_mac_cnt + 8'd1;
                    // 1x1 sequential: n_cfg=1 per trip → drain 1+2=3 cycles.
                    // swc (3x3/dw): drain reg_n_config+2 cycles as before.
                    if (flush_mac_cnt >= (swc_en ? {3'd0, reg_n_config} + 8'd2 : 8'd3)) begin
                        if (!swc_en) begin
                            // 1x1 path: loop flush_k 0..n_cfg-1, then advance M-block
                            if ({1'b0, flush_k} < reg_n_config - 5'd1) begin
                                // More NZ slots in current M-block
                                flush_k    <= flush_k + 4'd1;
                                // c3 next slot: idx = wt_rd_idx[flush_k+1] (kept channel).
                                // Dense: wt_rd_idx[k]=k → k*HW = accumulator cũ (bit-exact).
                                // Sparse: kept idx tuỳ ý → đúng CHW channel base. Rev150.
                                c3_idxhw   <= wt_rd_idx[flush_k + 4'd1] * c3_hw;
                                act_ld_cnt <= 6'd0;
                                aw_cnt     <= 3'd0;
                                state      <= c3w ? ST_LOAD_ACT_W : ST_LOAD_ACT_1X1;
                            end else begin
                                // All NZ slots done for this M-block
                                flush_k <= 4'd0;
                                if (ic_blk_cnt < n_ic_blks - 8'd1) begin
                                    ic_blk_cnt <= ic_blk_cnt + 8'd1;
                                    state      <= ST_WAIT_DMA;
                                end else begin
                                    flush_cnt   <= 9'd0;
                                    flush_spec  <= 2'd0;
                                    flush_g     <= 2'd0;
                                    flush_p     <= 4'd0;
                                    wb_done_cnt <= 9'd0;
                                    // Rev139 fastkick: middle LS ic-kicks (no_flush=1) skip the
                                    // NS*64-cycle writeback — their rr outputs were overwritten
                                    // anyway; LS_OUT DMAs out_bram only after the flush kick.
                                    state       <= (fastkick && no_flush) ? ST_DONE : ST_WRITEBACK;
                                end
                            end
                        end else begin
                            // 3x3/dw path — lock-step taps (mf_kh,mf_kw) within input
                            // channel swc_ic; after 9 taps advance channel (ic loop),
                            // accumulating until all Cin channels done. addr includes
                            // channel base swc_ic*tile_chw (CHW layout).
                            if (mf_kh == 2'd2 && mf_kw == 2'd2) begin
                                if (swc_ic < swc_n_ic - 8'd1) begin
                                    // next input channel, tap (0,0); keep accumulating
                                    swc_ic          <= swc_ic + 8'd1;
                                    mf_kh           <= 2'd0;
                                    mf_kw           <= 2'd0;
                                    act_ld_cnt      <= 6'd0;
                                    in_bram_rd_addr <= (swc_ic + 8'd1) * tile_chw;
                                    state           <= ST_ACT_LOAD;
                                end else begin
                                    // all Cin (this chunk) × 9 taps accumulated.
                                    // no_flush: middle chunk → just end, psum persists.
                                    // else: flush 64 outputs (requant).
                                    flush_spec   <= 4'd0;   // B-w FIX: DW/3x3 use SPEC0 only; reset (1x1
                                    // writeback leaves flush_spec=NS since [3:0] no longer wraps at 3→0)
                                    flush_g      <= 2'd0;
                                    flush_p      <= 4'd0;
                                    out_bram_cnt <= 8'd0;
                                    state        <= no_flush ? ST_DONE : ST_FLUSH_RR;
                                end
                            end else if (mf_kw == 2'd2) begin
                                // End of row: next tap = (kh+1, 0) in same channel
                                mf_kw           <= 2'd0;
                                mf_kh           <= mf_kh + 2'd1;
                                act_ld_cnt      <= 6'd0;
                                in_bram_rd_addr <= (swc_ic * tile_chw)
                                                    + (mf_kh + 2'd1) * reg_win[9:0];
                                state           <= ST_ACT_LOAD;
                            end else begin
                                // Next column in same row: next tap = (kh, kw+1)
                                mf_kw           <= mf_kw + 2'd1;
                                act_ld_cnt      <= 6'd0;
                                in_bram_rd_addr <= (swc_ic * tile_chw)
                                                    + mf_kh * reg_win[9:0]
                                                    + {8'd0, mf_kw} + 10'd1;
                                state           <= ST_ACT_LOAD;
                            end
                        end
                    end
                end

                // ── 3x3/dw: flush rr_unit (64 outputs) ───────────────────────
                ST_FLUSH_RR: begin
                    flush_valid <= 1'b1;
                    // Guard advance with flush_valid (registered, lags 1 cycle) so the
                    // first valid psum read is accu_flat[{0,0}] (flush_p=0), not p=1.
                    // Without the guard flush_p advances before flush_valid asserts →
                    // p=0 output dropped → 1-position spatial offset. Mirrors ST_WRITEBACK.
                    if (flush_valid) begin
                        // Rev142 FR4: 4 pos/cycle (lane k = p|k) → 16 nhịp thay 64.
                        flush_p <= flush_p + (fr4_en ? 4'd4 : 4'd1);
                        if (flush_p == (fr4_en ? 4'd12 : 4'd15)) begin
                            flush_p <= 4'd0;
                            if (flush_g == 2'd3) begin
                                flush_g    <= 2'd0;
                                tc_next    <= tc_valid & ~tc_last;
                                pp_out_sel <= ~pp_out_sel;
                                state      <= ST_DONE;
                            end else begin
                                flush_g <= flush_g + 2'd1;
                            end
                        end
                    end
                    if (rr_valid) out_bram_cnt <= out_bram_cnt + 8'd1;
                end

                // ── 1x1: writeback all NS*64 outputs (NS SPEC × G groups × 16 pos) ──────
                ST_WRITEBACK: begin
                    // Đòn 4a' RR4: 4 pos/cycle (flush_p +4, lane k = p|k) → NS*16 nhịp.
                    if (flush_cnt <= (rr4_en ? (NS*16 - 1) : (NS*64 - 1))) begin
                        flush_valid <= 1'b1;
                        flush_cnt   <= flush_cnt + 10'd1;
                    end
                    if (flush_valid) begin
                        if (flush_p == (rr4_en ? 4'd12 : 4'd15)) begin
                            flush_p <= 4'd0;
                            if (flush_g == 2'd3) begin
                                flush_g    <= 2'd0;
                                flush_spec <= flush_spec + 4'd1;   // B-w: [3:0], 0..NS-1
                            end else begin
                                flush_g <= flush_g + 2'd1;
                            end
                        end else begin
                            flush_p <= flush_p + (rr4_en ? 4'd4 : 4'd1);
                        end
                    end
                    if (wb_done_cnt == (rr4_en ? (NS*16 - 1) : (NS*64 - 1)) && rr_valid) begin
                        state <= ST_DONE;
                    end
                    if (rr_valid) wb_done_cnt <= wb_done_cnt + 10'd1;
                end

                // ── 1x1: load per-channel requant params from WT BRAM tail ────
                // Layout (interleaved, 7 bytes per channel):
                //   [m0_lo, m0_hi, sh, bias_b0, bias_b1, bias_b2, bias_b3] × Cout
                // Starts at BRAM address wt_nz_bytes_1x1.
                // BRAM read latency = 2 cycles; rq_load_byte tracks consumer side
                // while wt_bram_rd_addr runs 2 ahead.
                ST_LOAD_RQ: begin
                    // Advance BRAM read address (issue side, 2 cycles ahead of data)
                    wt_bram_rd_addr <= wt_bram_rd_addr + 12'd1;
                    flush_cnt       <= flush_cnt + 9'd1;

                    // Consume data (available 3 cycles after address was issued — RD_LAT=3 Rev139)
                    if (flush_cnt >= 9'd3) begin
                        case (rq_load_byte)
                            3'd0: rq_m0[rq_load_ch][7:0]    <= wt_bram_rd_data;
                            3'd1: rq_m0[rq_load_ch][15:8]   <= wt_bram_rd_data;
                            3'd2: rq_sh[rq_load_ch]         <= wt_bram_rd_data[4:0];
                            3'd3: rq_bias[rq_load_ch][7:0]  <= wt_bram_rd_data;
                            3'd4: rq_bias[rq_load_ch][15:8] <= wt_bram_rd_data;
                            3'd5: rq_bias[rq_load_ch][23:16]<= wt_bram_rd_data;
                            3'd6: begin
                                rq_bias[rq_load_ch][31:24] <= wt_bram_rd_data;
                                if (rq_load_ch < reg_cout[7:0] - 8'd1) begin
                                    rq_load_ch   <= rq_load_ch + 8'd1;
                                    rq_load_byte <= 3'd0;
                                end else begin
                                    rq_load_byte <= 3'd7;  // sentinel: done
                                end
                            end
                            default: ;
                        endcase
                        if (rq_load_byte != 3'd6 && rq_load_byte != 3'd7)
                            rq_load_byte <= rq_load_byte + 3'd1;
                    end

                    // Done when sentinel is set (last channel's last byte consumed).
                    // Transition immediately — data already latched into rq_bias[last_ch][31:24].
                    if (rq_load_byte == 3'd7) begin
                        rq_dirty        <= 1'b0;   // Đòn 4a' RQD: params tươi cho wt_uram hiện tại
                        flush_cnt       <= 9'd0;
                        rq_load_ch      <= 8'd0;
                        rq_load_byte    <= 3'd0;
                        // LAYER_MODE: keep_psum=1 for ic>0 → must NOT clear psum.
                        // OLD MODE: ic_blk_cnt==0 always → ~keep_psum=1 (first ic_blk, correct).
                        acc_clr_r       <= swc_en ? 1'b0 : ~keep_psum;
                        wt_ld_cnt       <= 7'd0;
                        wt_slot_cnt     <= 4'd0;
                        wt_bram_rd_addr <= 12'd0;
                        // swc: requant loaded → start the 3x3/dw pixel scan.
                        // 1x1: continue to weight load (c3 → wide WT load) for the first ic_blk.
                        if (swc_en)      state <= ST_PIX_CLR;
                        else if (c3_act) begin
                            wtw_cnt    <= 7'd0;
                            wtw_active <= 1'b1;
                            state      <= ST_WT_LOAD_W;
                        end else         state <= ST_WT_LOAD;
                    end
                end

                ST_DONE: begin
                    ap_done_r     <= 1'b1;
                    ap_done_latch <= 1'b1;  // hold for SW polling
                    ap_idle_r     <= 1'b1;
                    // Clear flush_valid so rr_unit pipeline drains and rr_valid goes
                    // LOW before the next inference starts ST_WRITEBACK. Without this,
                    // rr_valid stays HIGH permanently after the first run and writes a
                    // stale value to BRAM[0] (wb_done_cnt=0) before the real g[0],
                    // causing a 1-pixel output offset on all runs after the first.
                    flush_valid   <= 1'b0;
                    // 1x1: flip here (no flip in ST_WRITEBACK); 3x3/dw: already flipped in ST_FLUSH_RR
                    if (!swc_en) pp_out_sel <= ~pp_out_sel;
                    state         <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase

            // ── Đòn 4a W11: capture cả 16 lane từ 1 word (addr issue cnt1 → rd_wdata valid cnt3) ──
            if (state == ST_LOAD_ACT_1X1 && w11 && act_ld_cnt == 6'd3) begin : w11_cap
                integer wp;
                for (wp = 0; wp < 16; wp = wp + 1) begin
                    act_g0[wp] <= $signed(act_w_data[wp*8 +: 8]);
                    act_g1[wp] <= $signed(act_w_data[wp*8 +: 8]);
                    act_g2[wp] <= $signed(act_w_data[wp*8 +: 8]);
                    act_g3[wp] <= $signed(act_w_data[wp*8 +: 8]);
                end
            end
            // ── Latch 1x1 activations from BRAM (RD_LAT=3 Rev139: window 3..18 → 4..19) ──
            // c3 pad masking: LS flags out-of-range taps/cols → capture 0 instead of garbage.
            if (state == ST_LOAD_ACT_1X1 && !w11 && act_ld_cnt >= 6'd4 && act_ld_cnt <= 6'd19) begin : act_cap
                reg signed [7:0] cap_d;
                cap_d = (c3_act && (ls_c3_zero
                                    || (ls_c3_maskl && (act_ld_cnt == 6'd4))
                                    || (ls_c3_maskr && ((act_ld_cnt - 6'd4) == {2'd0, ls_c3_lastp}))))
                        ? 8'sd0 : $signed(in_bram_rd_data);
                act_g0[act_ld_cnt - 6'd4] <= cap_d;
                act_g1[act_ld_cnt - 6'd4] <= cap_d;
                act_g2[act_ld_cnt - 6'd4] <= cap_d;
                act_g3[act_ld_cnt - 6'd4] <= cap_d;
            end

            // ── Rev36: registered OUT BRAM write pipeline (runs every cycle) ──
            // owr_en_r / owr_data_r capture rr_valid / rr_out into the same FF
            // stage; owr_ptr counts owr_en_r pulses → address & data locked,
            // immune to Vivado retiming skew that caused the 1-pixel offset.
            // Both 1x1 (ST_WRITEBACK) and 3x3/dw (ST_FLUSH_RR) share this path.
            owr_en_r   <= rr_valid;
            owr_data_r <= rr_out[7:0];
            // Đòn 4a' RR4: 3 lane trên cùng tầng FF (đồng pha owr_data_r — không skew);
            // owr_quad_r registered cùng nhịp với owr_en_r để mode ổn định theo pulse.
            owr_q_r    <= {rr_out_q3, rr_out_q2, rr_out_q1};
            // FIX Rev142b (board-caught 260 det): ST_FLUSH_RR thoát theo phía FEED (advance
            // cuối) — KHÁC ST_WRITEBACK chờ drain (wb_done_cnt) — nên ~5 output cuối tới SAU
            // khi state rời đi → quad rơi giữa chừng → đuôi tile ghi byte-mode ptr+1 (hỏng).
            // Giữ cờ khi pipeline còn output đang bay (pipe/rr/owr bất kỳ còn valid).
            owr_quad_r <= (rr4_en && (state == ST_WRITEBACK))
                       || (fr4_en && (state == ST_FLUSH_RR))
                       || (owr_quad_r && (pipe_accu_valid_r || rr_valid || owr_en_r));
            if (!owr_active && !owr_en_r)
                owr_ptr <= 10'd0;          // reset between writeback phases (per-tile for 3x3/dw)
            else if (owr_en_r)
                owr_ptr <= owr_ptr + (owr_quad_r ? 10'd4 : 10'd1);
            // Đòn 4a' RQD: mọi ghi wt_uram → RQ params stale (đặt SAU case → thắng clear
            // trong ST_LOAD_RQ nếu trùng nhịp — an toàn phía dirty).
            if (dma_wt_w128_en || dma_wt_wr_en) rq_dirty <= 1'b1;
        end
    end

endmodule

// ── out_bram_wide (Đòn 4a') — OUT buffer 784B, LUT-RAM 49×128 ────────────────
// 1 write port 32-bit + byte-lane we (RR4 quad = 4 lane/cycle, legacy byte = 1 lane;
// we16 nội bộ = wr_we << waddr[1:0]*4). Đọc: cùng rd_addr (byte addr từ dma_engine —
// wide mode nhảy +16/beat), trả CẢ byte (RD_LAT=1, giữ Rev37) lẫn word 128b (RD_LAT=1,
// khớp ST_WR_WPRIME 1-cycle). Distributed RAM: né hẳn họ bug RAM-generator (Rev58),
// async read → output reg = đúng 1 cycle, không drama RD_LAT.
module out_bram_wide (
    input  wire         clk,
    input  wire [7:0]   wr_waddr,     // 32-bit word address (0..195)
    input  wire [31:0]  wr_wdata,
    input  wire [3:0]   wr_we,        // byte lanes trong 32-bit word
    input  wire [9:0]   rd_addr,      // BYTE address (dma_out_addr)
    output reg  [7:0]   rd_bdata,     // byte path (1 cycle)
    output reg  [127:0] rd_wdata128   // wide path (1 cycle)
);
    (* ram_style = "distributed" *) reg [127:0] mem [0:48];   // 49 × 16B = 784B
    wire [15:0]  we16    = {12'd0, wr_we} << {wr_waddr[1:0], 2'b00};
    wire [5:0]   wr_w128 = wr_waddr[7:2];
    wire [127:0] wdata_rep = {4{wr_wdata}};
    wire [127:0] rd_word = mem[rd_addr[9:4]];
    integer li;
    always @(posedge clk) begin
        for (li = 0; li < 16; li = li + 1)
            if (we16[li]) mem[wr_w128][li*8 +: 8] <= wdata_rep[li*8 +: 8];
        rd_bdata    <= rd_word[{rd_addr[3:0], 3'b000} +: 8];
        rd_wdata128 <= rd_word;
    end
endmodule
