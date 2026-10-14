// unified_cnn_top.v  --  B-0 IP top: dma_engine + unified_acc_top + layer_sequencer.
// Standalone Vivado BD cell.  AXI4-Lite forwarded to unified_acc_top's internal slave.
// layer_sequencer owns dma_engine when LAYER_MODE=1 (host writes REG_LAYER_MODE=1).
// B-0 GATE: 1-layer conv1x1 board bit-exact.
`timescale 1ns/1ps

module unified_cnn_top (
    input  wire        aclk,
    input  wire        aresetn,

    // ── AXI4-Lite config slave (passed to unified_acc_top.u_axil) ────────────
    input  wire [8:0]  s_awaddr,
    input  wire        s_awvalid,
    output wire        s_awready,
    input  wire [31:0] s_wdata,
    input  wire [3:0]  s_wstrb,
    input  wire        s_wvalid,
    output wire        s_wready,
    output wire [1:0]  s_bresp,
    output wire        s_bvalid,
    input  wire        s_bready,
    input  wire [8:0]  s_araddr,
    input  wire        s_arvalid,
    output wire        s_arready,
    output wire [31:0] s_rdata,
    output wire [1:0]  s_rresp,
    output wire        s_rvalid,
    input  wire        s_rready,

    // ── AXI4 Master (dma_engine → HP0 DDR) ──────────────────────────────────
    output wire [63:0]  m_axi_awaddr,
    output wire [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire [3:0]   m_axi_awcache,
    output wire [2:0]   m_axi_awprot,
    output wire         m_axi_awlock,
    output wire [3:0]   m_axi_awqos,
    output wire         m_axi_awvalid,
    input  wire         m_axi_awready,
    output wire [127:0] m_axi_wdata,
    output wire [15:0]  m_axi_wstrb,
    output wire         m_axi_wlast,
    output wire         m_axi_wvalid,
    input  wire         m_axi_wready,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready,
    output wire [63:0]  m_axi_araddr,
    output wire [7:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    output wire [3:0]   m_axi_arcache,
    output wire [2:0]   m_axi_arprot,
    output wire         m_axi_arlock,
    output wire [3:0]   m_axi_arqos,
    output wire         m_axi_arvalid,
    input  wire         m_axi_arready,
    input  wire [127:0] m_axi_rdata,
    input  wire         m_axi_rlast,
    input  wire         m_axi_rvalid,
    output wire         m_axi_rready,

    output wire        irq_out
);

    // ── DMA engine memory bus ────────────────────────────────────────────────
    wire        dma_active, dma_done;
    wire [9:0]  dma_in_addr;
    wire [7:0]  dma_in_data;
    wire        dma_in_wr_en;
    wire [5:0]  dma_in_w128_addr;
    wire [127:0] dma_in_w128_data;
    wire        dma_in_w128_en;
    wire [11:0] dma_wt_addr;
    wire [7:0]  dma_wt_data;
    wire        dma_wt_wr_en;
    wire [12:0] dma_wt_w128_addr;
    wire [127:0] dma_wt_w128_data;
    wire        dma_wt_w128_en;
    wire [18:0] dma_out_addr;
    wire [7:0]  dma_out_data;
    wire [127:0] dma_out_data128_w;   // Đòn 4a' OW: out bram word → dma_engine wide path
    wire         dma_out_wide_w;

    // ── layer_sequencer → dma_engine request bus ─────────────────────────────
    wire        ls_dma_req;
    wire [1:0]  ls_dma_tgt;
    wire [63:0] ls_dma_addr;
    wire [23:0] ls_dma_bytes;

    // ── Đòn 4b netseq_v2 wires ───────────────────────────────────────────────
    wire        ns_mode_w;                 // feat bit10 & layer_mode (từ acc_top)
    wire        ns_dma_req_w;
    wire [1:0]  ns_dma_tgt_w;
    wire [63:0] ns_dma_addr_w;
    wire [23:0] ns_dma_bytes_w;
    wire        ns_wr_en_w;
    wire [8:0]  ns_wr_addr_w;
    wire [31:0] ns_wr_data_w;
    wire        ns_ls_go_w, ns_busy_w, ns_done_w, ns_err_w;
    wire [3:0]  ns_dbg_w;
    wire        feat_en_ot_w;
    // ── DEINT-ON-HW wires (LS ↔ unified_acc) ──
    wire        deint_feat_w, deint_scat_start_w, deint_scat_done_w, deint_dma_active_w;
    wire [15:0] deint_pos_w, deint_npos_w;
    wire [7:0]  deint_ock_w;
    // ── Rev169 in-place lag scatter (LS ↔ unified_acc) ──
    wire        lag_en_w;
    wire        drain_start_w, drain_done_w;
    wire [15:0] drain_pos_w;
    wire        dma_desc_en_w;
    wire [2:0]  dma_desc_addr_w;
    wire [127:0] dma_desc_data_w;
    // DMA mux: netseq chỉ fetch DESC khi LS idle (by construction) → OR đơn giản
    wire        dma_req_mux   = ls_dma_req | ns_dma_req_w;
    wire [1:0]  dma_tgt_mux   = ns_dma_req_w ? ns_dma_tgt_w   : ls_dma_tgt;
    wire [63:0] dma_addr_mux  = ns_dma_req_w ? ns_dma_addr_w  : ls_dma_addr;
    wire [23:0] dma_bytes_mux = ns_dma_req_w ? ns_dma_bytes_w : ls_dma_bytes;

    // ── unified_acc_top config outputs → layer_sequencer ─────────────────────
    wire        reg_layer_mode_w, dw_mode_w;
    wire [7:0]  reg_n_ocb_w;
    wire [15:0] reg_n_pos_w;
    wire [7:0]  reg_n_icb_w;
    wire [23:0] reg_wt_blk_bytes_w, reg_act_chunk_w;
    wire [23:0] reg_out_bytes_w;   // B-w: REG_OUT_BYTES(0x150) = per-kick OUT tile bytes (OC_PER_KICK*16)
    wire [63:0] reg_in_addr_w, reg_wt_addr_w, reg_out_addr_w;
    wire        k_done_pulse_w, uni_ap_start_w;

    // ── layer_sequencer → unified_acc_top tile interface ─────────────────────
    wire        ls_k_ap_start_w, ls_k_dma_in_done_w;
    wire        ls_k_keep_psum_w, ls_k_no_flush_w;
    wire        ls_busy_w, ls_done_latch_w;
    wire [17:0] ls_wt_uram_off_w;    // Rev139: 18b (wt_uram 256KB)
    wire [20:0] ls_act_uram_off_w;   // B-i/c3: ACT-URAM offset — Rev139: 21b (act_uram 1.5MB)
    wire        feat_en_bi_w;        // B-i: enable (unified_acc_top REG_FEAT_EN → layer_sequencer)
    // ── Rev139 B-c3 9-pass wires ─────────────────────────────────────────────
    wire        feat_en_c3_w;        // c3 enable (REG_FEAT_EN bit1)
    wire [31:0] reg_c3_cfg_w;        // 0x170 c3 config
    wire [15:0] reg_act_total_w;     // 0x180 LS_ACTL chunk count
    wire [20:0] reg_c3_slice_w;      // REG_STRIDE tái dụng: 16*H*W (icc-slice)
    wire        ls_c3_zero_w, ls_c3_maskl_w, ls_c3_maskr_w, ls_c3_stride2_w;
    wire [3:0]  ls_c3_lastp_w;

    // ════════════════════════════════════════════════════════════════════════
    dma_engine u_dma (
        .aclk             (aclk),
        .aresetn          (aresetn),
        .dma_req          (dma_req_mux),
        .dma_req_tgt      (dma_tgt_mux),
        .dma_req_addr     (dma_addr_mux),
        .dma_req_bytes    (dma_bytes_mux),
        .dma_active       (dma_active),
        .dma_done         (dma_done),
        // AXI4 Master
        .m_axi_awaddr     (m_axi_awaddr),
        .m_axi_awlen      (m_axi_awlen),
        .m_axi_awsize     (m_axi_awsize),
        .m_axi_awburst    (m_axi_awburst),
        .m_axi_awcache    (m_axi_awcache),
        .m_axi_awprot     (m_axi_awprot),
        .m_axi_awlock     (m_axi_awlock),
        .m_axi_awqos      (m_axi_awqos),
        .m_axi_awvalid    (m_axi_awvalid),
        .m_axi_awready    (m_axi_awready),
        .m_axi_wdata      (m_axi_wdata),
        .m_axi_wstrb      (m_axi_wstrb),
        .m_axi_wlast      (m_axi_wlast),
        .m_axi_wvalid     (m_axi_wvalid),
        .m_axi_wready     (m_axi_wready),
        .m_axi_bvalid     (m_axi_bvalid),
        .m_axi_bready     (m_axi_bready),
        .m_axi_araddr     (m_axi_araddr),
        .m_axi_arlen      (m_axi_arlen),
        .m_axi_arsize     (m_axi_arsize),
        .m_axi_arburst    (m_axi_arburst),
        .m_axi_arcache    (m_axi_arcache),
        .m_axi_arprot     (m_axi_arprot),
        .m_axi_arlock     (m_axi_arlock),
        .m_axi_arqos      (m_axi_arqos),
        .m_axi_arvalid    (m_axi_arvalid),
        .m_axi_arready    (m_axi_arready),
        .m_axi_rdata      (m_axi_rdata),
        .m_axi_rlast      (m_axi_rlast),
        .m_axi_rvalid     (m_axi_rvalid),
        .m_axi_rready     (m_axi_rready),
        // ACT memory
        .dma_in_addr      (dma_in_addr),
        .dma_in_data      (dma_in_data),
        .dma_in_wr_en     (dma_in_wr_en),
        .dma_in_w128_addr (dma_in_w128_addr),
        .dma_in_w128_data (dma_in_w128_data),
        .dma_in_w128_en   (dma_in_w128_en),
        // WT URAM
        .dma_wt_addr      (dma_wt_addr),
        .dma_wt_data      (dma_wt_data),
        .dma_wt_wr_en     (dma_wt_wr_en),
        .dma_wt_w128_addr (dma_wt_w128_addr),
        .dma_wt_w128_data (dma_wt_w128_data),
        .dma_wt_w128_en   (dma_wt_w128_en),
        // OUT BRAM: byte path + Đòn 4a' OW wide path (feat bit8, out_bram_wide 128b)
        .dma_out_addr     (dma_out_addr),
        .dma_out_data     (dma_out_data),
        .out_wide         (dma_out_wide_w),
        .dma_out_data128  (dma_out_data128_w),
        // Descriptor (Đòn 4b: netseq_v2 nhận 8 beat TGT_DESC)
        .dma_desc_w128_addr (dma_desc_addr_w),
        .dma_desc_w128_data (dma_desc_data_w),
        .dma_desc_w128_en   (dma_desc_en_w)
    );

    // ── Đòn 4b: Task Scheduler on-PL (Phase 1) ───────────────────────────────
    network_seq_v2 u_ns (
        .aclk(aclk), .aresetn(aresetn),
        .ns_go(uni_ap_start_w & ns_mode_w),
        .desc_base(reg_in_addr_w),
        .ns_dma_req(ns_dma_req_w), .ns_dma_tgt(ns_dma_tgt_w),
        .ns_dma_addr(ns_dma_addr_w), .ns_dma_bytes(ns_dma_bytes_w),
        .dma_active(dma_active), .dma_done(dma_done),
        .dma_desc_w128_en(dma_desc_en_w), .dma_desc_w128_addr(dma_desc_addr_w),
        .dma_desc_w128_data(dma_desc_data_w),
        .ns_wr_en(ns_wr_en_w), .ns_wr_addr(ns_wr_addr_w), .ns_wr_data(ns_wr_data_w),
        .ns_ls_go(ns_ls_go_w),
        .ls_busy(ls_busy_w), .ls_done_latch(ls_done_latch_w),
        .ns_busy(ns_busy_w), .ns_done(ns_done_w), .ns_err(ns_err_w),
        .dbg_state(ns_dbg_w)
    );

    // ════════════════════════════════════════════════════════════════════════
    unified_acc_top u_uni (
        .aclk             (aclk),
        .aresetn          (aresetn),
        // AXI4-Lite
        .s_awaddr         (s_awaddr),
        .s_awvalid        (s_awvalid),
        .s_awready        (s_awready),
        .s_wdata          (s_wdata),
        .s_wstrb          (s_wstrb),
        .s_wvalid         (s_wvalid),
        .s_wready         (s_wready),
        .s_bresp          (s_bresp),
        .s_bvalid         (s_bvalid),
        .s_bready         (s_bready),
        .s_araddr         (s_araddr),
        .s_arvalid        (s_arvalid),
        .s_arready        (s_arready),
        .s_rdata          (s_rdata),
        .s_rresp          (s_rresp),
        .s_rvalid         (s_rvalid),
        .s_rready         (s_rready),
        // DMA bus from dma_engine
        .dma_in_addr      (dma_in_addr),
        .dma_in_data      (dma_in_data),
        .dma_in_wr_en     (dma_in_wr_en),
        .dma_in_w128_addr (dma_in_w128_addr),
        .dma_in_w128_data (dma_in_w128_data),
        .dma_in_w128_en   (dma_in_w128_en),
        .dma_in_done      (dma_done),
        .dma_in_active    (dma_active),
        .dma_wt_addr      (dma_wt_addr),
        .dma_wt_data      (dma_wt_data),
        .dma_wt_wr_en     (dma_wt_wr_en),
        .dma_wt_w128_addr (dma_wt_w128_addr),
        .dma_wt_w128_data (dma_wt_w128_data),
        .dma_wt_w128_en   (dma_wt_w128_en),
        .dma_out_addr     (dma_out_addr),
        .dma_out_data     (dma_out_data),
        .dma_out_data128  (dma_out_data128_w),
        .dma_out_wide     (dma_out_wide_w),
        // IRQ
        .irq_out          (irq_out),
        // Config outputs consumed by layer_sequencer; others float
        .reg_cin_out      (),
        .reg_hin_out      (),
        .reg_win_out      (),
        .reg_ncfg_out     (),
        .reg_out_bytes_out(reg_out_bytes_w),   // B-w: → layer_sequencer out_tile_bytes
        .reg_wt_bytes_out (),
        .reg_in_addr_out  (reg_in_addr_w),
        .reg_wt_addr_out  (reg_wt_addr_w),
        .reg_out_addr_out (reg_out_addr_w),
        .reg_cout_out     (),
        .reg_conv_type_out(),
        // No systolic engine in this IP
        .sys_busy_in      (1'b0),
        .sys_done_latch_in(1'b0),
        // ap_start forwarded to layer_sequencer
        .ap_start_out     (uni_ap_start_w),
        .ap_done_out      (),
        .dma_nxt_req_o    (),
        .dbg_seq_state_in (4'd0),
        .dbg_kernel_state_out(),
        // Layer_sequencer config outputs
        .reg_layer_mode_out   (reg_layer_mode_w),
        .dw_mode_out          (dw_mode_w),
        .reg_n_ocb_out        (reg_n_ocb_w),
        .reg_n_pos_out        (reg_n_pos_w),
        .reg_n_icb_out        (reg_n_icb_w),
        .reg_wt_blk_bytes_out (reg_wt_blk_bytes_w),
        .reg_act_chunk_out    (reg_act_chunk_w),
        .k_done_pulse_out     (k_done_pulse_w),
        // Layer_sequencer → kernel per-tile interface
        .ls_k_ap_start    (ls_k_ap_start_w),
        .ls_k_dma_in_done (ls_k_dma_in_done_w),
        .ls_k_keep_psum   (ls_k_keep_psum_w),
        .ls_k_no_flush    (ls_k_no_flush_w),
        .ls_busy          (ls_busy_w),
        .ls_done_latch    (ls_done_latch_w),
        .ls_wt_uram_off   (ls_wt_uram_off_w),
        .ls_act_uram_off  (ls_act_uram_off_w),   // B-i/c3
        .ns_wr_en_in      (ns_wr_en_w),          // Đòn 4b: cổng ghi reg-file từ netseq
        .ns_wr_addr_in    (ns_wr_addr_w),
        .ns_wr_data_in    (ns_wr_data_w),
        .ns_busy_in       (ns_busy_w),
        .ns_done_in       (ns_done_w),
        .ns_err_in        (ns_err_w),
        .ns_dbg_in        (ns_dbg_w),
        .reg_feat_en_ns_out(ns_mode_w),          // NS mode (feat bit10 & layer_mode)
        .reg_feat_en_ot_out(feat_en_ot_w),       // OT out-transpose (bit11)
        // ── DEINT-ON-HW (bit12) ──
        .reg_feat_en_deint_out(deint_feat_w),
        .deint_scat_start (deint_scat_start_w),
        .deint_pos        (deint_pos_w),
        .deint_npos       (deint_npos_w),
        .deint_ock        (deint_ock_w),
        .deint_dma_active (deint_dma_active_w),
        .deint_scat_done  (deint_scat_done_w),
        // ── Rev169 lag ──
        .lag_en_out       (lag_en_w),
        .drain_start      (drain_start_w),
        .drain_pos        (drain_pos_w),
        .drain_done       (drain_done_w),
        .reg_feat_en_bi_out(feat_en_bi_w),       // B-i
        // Rev139 B-c3 9-pass
        .reg_feat_en_c3_out(feat_en_c3_w),
        .reg_c3_cfg_out   (reg_c3_cfg_w),
        .reg_act_total_out(reg_act_total_w),
        .reg_c3_slice_out (reg_c3_slice_w),
        .ls_c3_zero       (ls_c3_zero_w),
        .ls_c3_maskl      (ls_c3_maskl_w),
        .ls_c3_maskr      (ls_c3_maskr_w),
        .ls_c3_lastp      (ls_c3_lastp_w),
        .ls_c3_stride2    (ls_c3_stride2_w)
    );

    // ════════════════════════════════════════════════════════════════════════
    layer_sequencer u_ls (
        .aclk             (aclk),
        .aresetn          (aresetn),
        .layer_mode       (reg_layer_mode_w),
        .dw_mode          (dw_mode_w),
        .feat_en_bi       (feat_en_bi_w),        // B-i: ACT-stationary enable
        .feat_en_ot       (feat_en_ot_w),        // Đòn 4b OT
        // ── DEINT-ON-HW (bit12) ──
        .deint_hw         (deint_feat_w & reg_layer_mode_w),
        .deint_scat_done  (deint_scat_done_w),
        .deint_pos        (deint_pos_w),
        .deint_npos       (deint_npos_w),
        .deint_ock        (deint_ock_w),
        .deint_scat_start (deint_scat_start_w),
        .deint_dma_active (deint_dma_active_w),
        // ── Rev169 lag ──
        .lag_en           (lag_en_w),
        .drain_start      (drain_start_w),
        .drain_done       (drain_done_w),
        .drain_pos        (drain_pos_w),
        // Đòn 4b: NS mode → LS nhận kick từ netseq thay vì AXI ap_start (mux dưới)
        .feat_en_c3       (feat_en_c3_w),        // Rev139: c3 9-pass enable
        .ap_start         (ns_mode_w ? ns_ls_go_w : uni_ap_start_w),
        .k_ap_done        (k_done_pulse_w),
        // Config from unified_acc_top AXI registers
        .n_ocb            (reg_n_ocb_w),
        .n_pos            (reg_n_pos_w),
        .n_icb            (reg_n_icb_w),
        .wt_block_bytes   (reg_wt_blk_bytes_w),
        .act_chunk_bytes  (reg_act_chunk_w),
        .out_tile_bytes   (reg_out_bytes_w),     // B-w: per-kick OUT bytes (OC_PER_KICK*16)
        .c3_cfg           (reg_c3_cfg_w),        // Rev139
        .act_total        (reg_act_total_w),     // Rev139
        .c3_slice         (reg_c3_slice_w),      // Rev139 fix icc-offset
        .act_base         (reg_in_addr_w),
        .wt_base          (reg_wt_addr_w),
        .out_base         (reg_out_addr_w),
        // DMA engine handshake
        .dma_req          (ls_dma_req),
        .dma_req_tgt      (ls_dma_tgt),
        .dma_req_addr     (ls_dma_addr),
        .dma_req_bytes    (ls_dma_bytes),
        .dma_active       (dma_active),
        .dma_done         (dma_done),
        // Kernel per-tile → unified_acc_top
        .k_ap_start       (ls_k_ap_start_w),
        .k_dma_in_done    (ls_k_dma_in_done_w),
        .k_keep_psum      (ls_k_keep_psum_w),
        .k_no_flush       (ls_k_no_flush_w),
        // Rev139 c3 masks → unified_acc_top ACT capture
        .c3_zero          (ls_c3_zero_w),
        .c3_maskl         (ls_c3_maskl_w),
        .c3_maskr         (ls_c3_maskr_w),
        .c3_lastp         (ls_c3_lastp_w),
        .c3_stride2       (ls_c3_stride2_w),
        .irq              (),
        .ls_busy          (ls_busy_w),
        .ls_done_latch    (ls_done_latch_w),
        .wt_uram_off      (ls_wt_uram_off_w),
        .act_uram_off     (ls_act_uram_off_w),   // B-i/c3
        .dbg_ls_state     ()
    );

endmodule
// network_seq_v2.v -- Đòn 4b Phase 1: Task Scheduler on-PL cho đường unified/layer_sequencer.
// "Host on PL": mỗi layer = 1 descriptor 128B trong DDR = ẢNH CHỤP reg-file 0x040..0x1E0
// (word k ↔ offset 0x040 + k*16, k=0..26) + w30=flags + w31=magic. netseq fetch descriptor
// (TGT_DESC, 8 beat×16B qua dma_engine — đường Rev132 board-PASS), GHI vào reg-file qua cổng
// ns_wr của axi_lite_slave (decode chung — mọi consumer không đổi), kick layer_sequencer,
// chờ ls_done_latch, layer kế. Host: ghi desc table 1 lần/segment + 1 ap_start + poll done.
//
// Race-free by construction (kế thừa Rev132): 1 dma_engine, netseq chỉ fetch DESC khi LS idle;
// LS tự quản mọi DMA trong layer; netseq chờ ls_done trước khi đụng reg-file cho layer kế.
//
// flags (w30): bit0 = LAST (segment kết thúc sau layer này).
// magic (w31[15:0]) = 0x4B44 ("KD") — guard chống desc rác; sai → err_latch + done (host đọc dbg).
`timescale 1ns/1ps
`include "cnn_defines.vh"

module network_seq_v2 (
    input  wire        aclk,
    input  wire        aresetn,

    // segment kick (từ unified_cnn_top: ap_start AXI khi feat NS bật)
    input  wire        ns_go,
    input  wire [63:0] desc_base,          // = reg_in_ddr_addr TẠI THỜI ĐIỂM kick (latch ở đây)

    // DMA (shared dma_engine) — chỉ dùng cho TGT_DESC
    output reg         ns_dma_req,
    output reg  [1:0]  ns_dma_tgt,
    output reg  [63:0] ns_dma_addr,
    output reg  [23:0] ns_dma_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    // descriptor beats từ dma_engine (TGT_DESC)
    input  wire         dma_desc_w128_en,
    input  wire [2:0]   dma_desc_w128_addr,
    input  wire [127:0] dma_desc_w128_data,

    // cổng ghi reg-file (axi_lite_slave ns_wr)
    output reg         ns_wr_en,
    output reg  [8:0]  ns_wr_addr,
    output reg  [31:0] ns_wr_data,

    // layer_sequencer handshake
    output reg         ns_ls_go,           // pulse 1 cyc → LS ap_start
    input  wire        ls_busy,
    input  wire        ls_done_latch,

    output reg         ns_busy,
    output reg         ns_done,            // latch tới ns_go kế
    output reg         ns_err,             // magic sai
    output wire [3:0]  dbg_state
);
    localparam S_IDLE  = 4'd0, S_FETCH = 4'd1, S_FWAIT = 4'd2, S_CHECK = 4'd3,
               S_WRITE = 4'd4, S_SETTLE= 4'd5, S_KICK  = 4'd6, S_WAIT  = 4'd7,
               S_NEXT  = 4'd8, S_DONE  = 4'd9;
    localparam TGT_DESC = 2'd3;
    localparam MAGIC = 16'h4B44;

    reg [3:0]   st;
    reg         ns_go_d;                   // Rev145: edge-detect (race S_DONE/!ns_go nuốt kick)
    reg         desc_got;                  // Rev146: beat cuối (addr=7) đã ghi buffer — data-arrival gate
    reg         busy_seen;                 // Rev147: LS busy ĐÃ LÊN sau kick (chống done cũ)
    reg [6:0]   kick_tmo;                  // re-kick nếu pulse bị nuốt (LS chưa nhận sau 64 cyc)
    reg [63:0]  dbase;                     // desc_base latched tại ns_go
    reg [7:0]   li;                        // layer index trong segment
    reg [127:0] desc [0:7];                // 8 beat = 128B
    reg [4:0]   wk;                        // word index 0..26 khi WRITE
    reg [1:0]   settle;

    assign dbg_state = st;

    // word k của descriptor (32b): desc[k>>2][(k&3)*32 +: 32]
    wire [31:0] dw = desc[wk[4:2]][{wk[1:0], 5'b00000} +: 32];
    wire [31:0] w30 = desc[7][95:64];
    wire [31:0] w31 = desc[7][127:96];

    integer di;
    always @(posedge aclk) begin
        if (!aresetn) begin
            st <= S_IDLE; ns_dma_req <= 1'b0; ns_dma_tgt <= 2'd0;
            ns_dma_addr <= 64'd0; ns_dma_bytes <= 24'd0;
            ns_wr_en <= 1'b0; ns_wr_addr <= 9'd0; ns_wr_data <= 32'd0;
            ns_ls_go <= 1'b0; ns_busy <= 1'b0; ns_done <= 1'b0; ns_err <= 1'b0;
            dbase <= 64'd0; li <= 8'd0; wk <= 5'd0; settle <= 2'd0; ns_go_d <= 1'b0; desc_got <= 1'b0;
            busy_seen <= 1'b0; kick_tmo <= 7'd0;
            for (di = 0; di < 8; di = di + 1) desc[di] <= 128'd0;
        end else begin
            ns_go_d  <= ns_go;
            ns_ls_go <= 1'b0;             // default: pulse
            ns_wr_en <= 1'b0;

            // descriptor beats từ dma_engine (bất kỳ lúc nào TGT_DESC active)
            if (dma_desc_w128_en) desc[dma_desc_w128_addr] <= dma_desc_w128_data;

            case (st)
                S_IDLE: if (ns_go & ~ns_go_d) begin   // Rev145: RISING EDGE — mỗi kick đúng 1 segment
                    dbase   <= desc_base;
                    li      <= 8'd0;
                    ns_busy <= 1'b1; ns_done <= 1'b0; ns_err <= 1'b0;
                    st      <= S_FETCH;
                end

                // fetch descriptor layer li (chờ DMA + LS rảnh — an toàn tuyệt đối)
                S_FETCH: if (!dma_active && !ls_busy && !ns_dma_req) begin
                    ns_dma_req   <= 1'b1;
                    ns_dma_tgt   <= TGT_DESC;
                    ns_dma_addr  <= dbase + {49'd0, li, 7'd0};   // + li*128
                    ns_dma_bytes <= 24'd128;
                    st           <= S_FWAIT;
                end

                // Rev146 FIX: dma_done là PULSE 1-cycle (dma_engine clear mỗi nhịp) — chờ theo
                // pulse có cửa sổ bắt done CŨ/lỡ pulse → S_CHECK sớm khi desc chưa về (board:
                // segment đầu sau host-kick đọc desc zeros/cũ; trials pass vì desc giống nhau).
                // Gate theo DATA-ARRIVAL: beat 8/8 (addr==7) phải GHI THẬT vào buffer.
                S_FWAIT: begin
                    if (ns_dma_req && dma_active) ns_dma_req <= 1'b0;   // req được nhận
                    if (dma_desc_w128_en && dma_desc_w128_addr == 3'd7) desc_got <= 1'b1;
                    if (desc_got && !dma_active && !ns_dma_req) begin
                        desc_got <= 1'b0;
                        st       <= S_CHECK;
                    end
                end

                S_CHECK: begin
                    if (w31[15:0] != MAGIC) begin
                        ns_err <= 1'b1; st <= S_DONE;
                    end else begin
                        wk <= 5'd0; st <= S_WRITE;
                    end
                end

                // ghi 27 thanh ghi 0x040..0x1E0, 1 reg/cyc
                S_WRITE: begin
                    ns_wr_en   <= 1'b1;
                    ns_wr_addr <= 9'h040 + {wk, 4'b0000};
                    ns_wr_data <= dw;
                    if (wk == 5'd26) begin wk <= 5'd0; settle <= 2'd0; st <= S_SETTLE; end
                    else wk <= wk + 5'd1;
                end

                S_SETTLE: begin                     // 2 cyc cho reg settle trước kick
                    settle <= settle + 2'd1;
                    if (settle == 2'd1) st <= S_KICK;
                end

                S_KICK: begin
                    ns_ls_go  <= 1'b1;              // pulse — LS vào việc
                    busy_seen <= 1'b0;
                    kick_tmo  <= 7'd0;
                    st        <= S_WAIT;
                end

                // Rev147 FIX (board-caught: NS-sau-NS sai, NS-đầu đúng): ls_done_latch của
                // layer TRƯỚC còn treo 1-2 cycle sau kick (LS chưa kịp clear/raise busy) →
                // netseq tưởng xong ngay → done sớm → host đọc arena CHƯA ĐƯỢC GHI.
                // Gate: phải THẤY busy LÊN rồi XUỐNG mới tin done. Re-kick nếu pulse bị nuốt.
                S_WAIT: begin
                    if (ls_busy) busy_seen <= 1'b1;
                    if (!busy_seen && !ls_busy) begin
                        kick_tmo <= kick_tmo + 7'd1;
                        if (kick_tmo == 7'd64) begin ns_ls_go <= 1'b1; kick_tmo <= 7'd0; end
                    end
                    if (busy_seen && !ls_busy && ls_done_latch) begin
                        st <= (w30[0]) ? S_DONE : S_NEXT;   // flags.LAST?
                    end
                end

                S_NEXT: begin
                    li <= li + 8'd1;
                    st <= S_FETCH;
                end

                S_DONE: begin
                    ns_busy <= 1'b0; ns_done <= 1'b1;
                    st <= S_IDLE;                   // Rev145: về IDLE ngay (edge-detect chống re-trigger;
                                                    // bản cũ chờ !ns_go → kick kế trùng cửa sổ bị NUỐT
                                                    // → host wait_done ăn ns_done CŨ = đọc data stale)
                end

                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
