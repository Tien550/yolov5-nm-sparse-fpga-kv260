// cnn_acc_top.v -- MINIMAL systolic-only top (Rev70+). The old unified kernel (i_spec×4,
// mul_group_array, psum_array, conv_tile_r1, conv_kernel_r1, accu_top, weight_decoder,
// depth_shuffle, tile_ctrl, dma_sequencer, layer_sequencer, unified_acc_top) is REMOVED — it
// became fully unused once conv1x1+conv3x3 route to the systolic datapath (Cout-tiling) and dw+fp32
// run on the host CPU. Removing it frees ~40% LUT for a wider pe_array.
//
// Kept: axi_lite_slave (SAME register map -> host unchanged) + dma_engine (AXI4 master) + the
// systolic datapath (systolic_seq + conv1x1_systolic_top + routing). Everything is systolic now.
//
// AXI base: 0xa0000000 (FROZEN). Host: LAYER_TYPE ignored (always systolic); set CIN/COUT/HIN(=HW)/
// IN/WT/OUT addrs + WT_BYTES([wmem][params]) + OUT_BYTES, then ap_start; poll ap_done.
`timescale 1ns/1ps
`include "cnn_defines.vh"

(* keep_hierarchy = "yes" *)
module cnn_acc_top (
    input  wire        aclk,
    input  wire        aresetn,
    // ── AXI4-Lite slave ──
    input  wire [8:0]  s_awaddr,  input  wire        s_awvalid, output wire        s_awready,
    input  wire [31:0] s_wdata,   input  wire [3:0]  s_wstrb,
    input  wire        s_wvalid,  output wire        s_wready,
    output wire [1:0]  s_bresp,   output wire        s_bvalid,  input  wire        s_bready,
    input  wire [8:0]  s_araddr,  input  wire        s_arvalid, output wire        s_arready,
    output wire [31:0] s_rdata,   output wire [1:0]  s_rresp,
    output wire        s_rvalid,  input  wire        s_rready,
    // ── AXI4 Master (128-bit DMA to DDR via HP0) ──
    output wire [63:0]  m_axi_awaddr,  output wire [7:0]   m_axi_awlen,  output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst, output wire [3:0]   m_axi_awcache,output wire [2:0]   m_axi_awprot,
    output wire         m_axi_awlock,  output wire [3:0]   m_axi_awqos,
    output wire         m_axi_awvalid, input  wire         m_axi_awready,
    output wire [127:0] m_axi_wdata,   output wire [15:0]  m_axi_wstrb,  output wire         m_axi_wlast,
    output wire         m_axi_wvalid,  input  wire         m_axi_wready,
    input  wire         m_axi_bvalid,  output wire         m_axi_bready,
    output wire [63:0]  m_axi_araddr,  output wire [7:0]   m_axi_arlen,  output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst, output wire [3:0]   m_axi_arcache,output wire [2:0]   m_axi_arprot,
    output wire         m_axi_arlock,  output wire [3:0]   m_axi_arqos,
    output wire         m_axi_arvalid, input  wire         m_axi_arready,
    input  wire [127:0] m_axi_rdata,   input  wire         m_axi_rlast,
    input  wire         m_axi_rvalid,  output wire         m_axi_rready,
    output wire        irq_out
);
    localparam SYS_N=16, SYS_LOGN=4, SYS_MAW=14;

    // ── config registers (AXI-lite) ──
    wire [31:0] reg_cin, reg_cout, reg_hin, reg_win;
    wire [63:0] reg_in_addr, reg_wt_addr, reg_out_addr;
    wire [23:0] reg_out_bytes, reg_wt_bytes;
    // Network sequencer (LT14) reuses existing registers: desc_base=reg_in_addr, n_layers=reg_cin[3:0]
    wire        reg_apply_relu;                 // per-layer relu (no-relu int8 convs exist)
    wire [3:0]  reg_conv_type;                  // 4-bit (was 3-bit: LT8-11 aliased to LT0-3 -> conv1x1 break)
    // B1 systolic layer-mode config (on-chip conv1x1 HW-tile/Cout-tile loop)
    wire        reg_layer_mode;
    wire [7:0]  reg_n_ocb;                       // n_cot (Cout-tiles)
    wire [15:0] reg_n_pos;                       // n_hwt (HW-tiles)
    wire [7:0]  reg_n_icb;                        // unused by systolic lseq (conv1x1 = full Cin/kick)
    wire [23:0] reg_wt_blk_bytes;                 // wt_tile_bytes ([wmem][pmem] blob, uniform)
    wire [23:0] reg_act_chunk;                    // act_tile_bytes (HW-chunk, 4KB-safe stride)
    wire [3:0]  reg_n_config;                     // I-SPEC sparse N (NZ per M-block); 0x080
    wire        ap_start;
    wire        sys_busy_w;
    reg         sys_done_latch_r;
    wire        is_dw = (reg_conv_type == 4'd4); // dw_layer
    wire        is_dwoc = (reg_conv_type == 4'd10); // T2-3 dw on-chip input (pad_rebank -> dw_engine)
    wire        is_dwocamem = (reg_conv_type == 4'd11); // C/T2-4 dw on-chip OUTPUT -> c1 amem (DMA input, NO DMA out)
    wire        is_dwany = is_dw | is_dwoc | is_dwocamem; // dw_layer active (any mode)
    wire        is_c3 = (reg_conv_type == 4'd5); // conv3x3 on-chip im2col engine (vs conv1x1 systolic)
    // ── B1: on-chip conv1x1 layer loop (systolic_layer_seq). reg_layer_mode=1 + plain conv1x1
    // (not dw, not conv3x3-onchip) → this sequencer owns dma_engine + engine kick, sweeps ALL
    // HW-tiles × Cout-tiles on ONE ap_start. Declared early (used by ap_done_mux below).
    // B1 REVERTED (rev120 = rev118 logic): measured rev119 B1 gave NO FPS gain (forward 0.9s both;
    // MMIO kick overhead is only 0.01s so there was nothing to remove — host is already optimized).
    // Tie 0 so Vivado prunes u_sys_lseq (frees 5 DSP + 327 LUT). Default = systolic per-tile path.
    wire        eng_lseq = 1'b0;
    wire        lseq_dma_req_w;  wire [1:0] lseq_dma_tgt_w;
    wire [63:0] lseq_dma_addr_w; wire [23:0] lseq_dma_bytes_w;
    wire        lseq_start_w, lseq_irq_w, lseq_busy_w, lseq_done_latch_w;
    // ── Tier 2: on-chip cross-engine copy conv3x3.omem -> conv1x1.amem (LAYER_TYPE=6). xcopy reads c3
    // omem via the shared sys_o_rd_addr port and writes c1 amem via its ld_a port. No DDR round-trip.
    wire        is_xc = (reg_conv_type == 4'd6);
    wire        is_c1na = (reg_conv_type == 4'd7);   // conv1x1 reading PRE-FILLED amem (rebank chain): skip ACT DMA
    wire        is_padxc = (reg_conv_type == 4'd8);  // T2-2 PAD-rebank: c1 omem -> c3 amem PADDED (no DDR)
    wire        is_c3na  = (reg_conv_type == 4'd9);   // conv3x3 reading PRE-FILLED amem (pad_rebank): skip ACT DMA
    wire        is_c3any = is_c3 | is_c3na;           // conv3x3 datapath selected (normal or skip-act)
    // ── I-SPEC sparse conv1x1 (LAYER_TYPE=12): paper-faithful N:16 on-chip COO engine.
    // Reuses systolic_seq (per-tile DMA load act→amem / wt→coo+params, kick, read omem out).
    // One kick = one spatial-tile (16 pos), full Cin (sparse N/M-tile) × full Cout (16 oc/pass).
    wire        is_sp = (reg_conv_type == 4'd12);
    wire        sp_done_w;   // systolic_seq waits this via sys_done_w mux (sys_done_latch_r handles ap_done)
    // ── c1_copy (LAYER_TYPE=13): c1 omem → c1 amem on-chip (no DDR). Phase 2 on-chip dataflow.
    // Follow with LT7 (is_c1na) so next conv1x1 reads pre-filled amem instead of DDR.
    wire        is_c1c = (reg_conv_type == 4'd13);
    wire        c1c_busy_w, c1c_done_latch_w;
    wire [13:0] c1c_rd_addr_w;
    wire        c1c_wr_en_w;
    wire [13:0] c1c_wr_addr_w;
    wire [SYS_N*8-1:0] c1c_wr_data_w;
    // ── Network sequencer (LAYER_TYPE=14): runs N=2 conv1x1 layers under ONE ap_start, no host
    // round-trip. network_seq drives engine kick + DMA; cnn_acc_top muxes engine dims on ns_cur_layer.
    wire        is_netseq = (reg_conv_type == 4'd14);
    wire        ns_dma_req_w; wire [1:0] ns_dma_tgt_w;
    wire [63:0] ns_dma_addr_w; wire [23:0] ns_dma_bytes_w;
    wire        ns_sys_start_w, ns_busy_w, ns_done_latch_w;
    wire [3:0]  ns_cur_layer;
    wire [13:0] ns_cur_cin;  wire [9:0] ns_cur_cout, ns_cur_hw;   // descriptor-parsed dims (Rev132)
    // ── I-SPEC on-chip layer loop (spec_layer_seq): is_sp + reg_layer_mode=1. 1 ap_start/layer,
    // engine loops spatial-tiles + owns dma_engine (eliminates host per-kick loop, 5890→1 kicks).
    wire        is_sp_lseq = is_sp & reg_layer_mode;
    // ── conv3x3 on-chip layer sequencer (LAYER_TYPE=5 + LAYER_MODE=1): one ap_start runs all
    // (band × Cout-chunk) iterations on-chip, eliminating host per-kick overhead.
    wire        is_c3_lseq = is_c3 & reg_layer_mode;
    wire        c3seq_dma_req_w;  wire [1:0]  c3seq_dma_tgt_w;
    wire [63:0] c3seq_dma_addr_w; wire [23:0] c3seq_dma_bytes_w;
    wire        c3seq_start_w, c3seq_busy_w, c3seq_done_latch_w;
    wire        spl_dma_req_w;  wire [1:0]  spl_dma_tgt_w;
    wire [63:0] spl_dma_addr_w; wire [23:0] spl_dma_bytes_w;
    wire        spl_start_w, spl_irq_w, spl_busy_w, spl_done_latch_w;
    wire        xc_busy, xc_done_latch, xc_wr_en;
    wire [SYS_MAW-1:0] xc_rd_addr, xc_wr_addr;
    wire [SYS_N*8-1:0] xc_wr_data;               // rebanked word -> c1 amem
    wire [SYS_N*8-1:0] c3_o_rd_data;             // conv3x3 omem read data (declared early: rebank source)
    wire        padxc_busy, padxc_done_latch, padxc_wr_en;
    wire [SYS_MAW-1:0] padxc_rd_addr, padxc_wr_addr;
    wire [SYS_N*8-1:0] padxc_wr_data;            // padded word -> c3 amem
    wire [SYS_MAW-1:0] dw_omem_rd_addr;          // T2-3: dw_layer reads c1 omem (declared early for sys_o_rd_addr mux)
    // C/T2-4: dw_layer writes its OUTPUT to c1 amem ON-CHIP (declared early for the c1_ld_a mux below).
    // Goes through the SAME c1 ld_a port as DMA/xcopy (1 write-port -> URAM holds; no 2nd write port).
    wire               dw_amem_we_w;
    wire [SYS_MAW-1:0] dw_amem_waddr_w;
    wire [SYS_N*8-1:0] dw_amem_wdata_w;
    // dw_layer nets (declared early for the muxes feeding axi_lite + dma_engine)
    wire        dw_dma_req_w;  wire [1:0] dw_dma_tgt_w;
    wire [63:0] dw_dma_addr_w; wire [23:0] dw_dma_bytes_w;
    wire        dw_busy_w, dw_irq_w; wire [7:0] dw_out_data_w;
    reg         dw_done_latch_r;
    // 3-way: dw_layer | conv1x1 layer-mode (systolic_layer_seq) | per-tile (systolic_seq).
    // eng_lseq + lseq_* are driven below (forward net refs, same style as sys_busy_w/sys_done_latch_r).
    wire        ap_done_mux = is_netseq ? ns_done_latch_w : is_sp_lseq ? spl_done_latch_w : is_c3_lseq ? c3seq_done_latch_w : is_c1c ? c1c_done_latch_w : is_dwany ? dw_done_latch_r : is_padxc ? padxc_done_latch : is_xc ? xc_done_latch : (eng_lseq ? lseq_done_latch_w : sys_done_latch_r);
    wire        busy_mux    = is_netseq ? ns_busy_w       : is_sp_lseq ? spl_busy_w    : is_c3_lseq ? c3seq_busy_w    : is_c1c ? c1c_busy_w    : is_dwany ? dw_busy_w       : is_padxc ? padxc_busy       : is_xc ? xc_busy       : (eng_lseq ? lseq_busy_w       : sys_busy_w);
    // NOTE: sys_done_latch_r already tracks the spec engine when is_sp (sys_done_w muxes sp_done_w),
    // and sys_busy_w comes from systolic_seq which owns the DMA+kick for sp too. No extra mux needed
    // here; the spec datapath plugs in via sys_done_w / dma_out_mux below.

    axi_lite_slave #(.IP_TYPE(1)) u_axil (
        .aclk(aclk), .aresetn(aresetn),
        .awaddr(s_awaddr), .awvalid(s_awvalid), .awready(s_awready),
        .wdata(s_wdata),   .wstrb(s_wstrb),     .wvalid(s_wvalid),  .wready(s_wready),
        .bresp(s_bresp),   .bvalid(s_bvalid),   .bready(s_bready),
        .araddr(s_araddr), .arvalid(s_arvalid), .arready(s_arready),
        .rdata(s_rdata),   .rresp(s_rresp),     .rvalid(s_rvalid),  .rready(s_rready),
        .ap_done_in(ap_done_mux), .ap_idle_in(~busy_mux),
        .dbg_seq_state_in(4'd0), .dbg_kernel_state_in(4'd0),
        .reg_cin(reg_cin), .reg_cout(reg_cout), .reg_hin(reg_hin), .reg_win(reg_win),
        .reg_conv_type(reg_conv_type),
        .reg_wt_ddr_addr(reg_wt_addr), .reg_in_ddr_addr(reg_in_addr), .reg_out_ddr_addr(reg_out_addr),
        .reg_out_bytes(reg_out_bytes), .reg_wt_bytes(reg_wt_bytes), .reg_apply_relu(reg_apply_relu),
        .reg_layer_mode(reg_layer_mode), .reg_n_ocb(reg_n_ocb), .reg_n_pos(reg_n_pos),
        .reg_n_icb(reg_n_icb), .reg_wt_blk_bytes(reg_wt_blk_bytes), .reg_act_chunk(reg_act_chunk),
        .reg_n_config(reg_n_config),   // I-SPEC sparse N (NZ/M-block); 0x080
        .ap_start(ap_start), .irq_out(irq_out)
    );

    // ── dma_engine nets ──
    wire        dma_active_w, dma_done_w;
    wire [9:0]  dma_in_addr_w;  wire [7:0] dma_in_data_w;  wire dma_in_wr_en_w;
    wire [5:0]  dma_in_w128_addr_w;  wire [127:0] dma_in_w128_data_w;  wire dma_in_w128_en_w;
    wire [11:0] dma_wt_addr_w;  wire [7:0] dma_wt_data_w;  wire dma_wt_wr_en_w;
    wire [12:0] dma_wt_w128_addr_w; wire [127:0] dma_wt_w128_data_w; wire dma_wt_w128_en_w;
    wire [9:0]  dma_out_addr_w;
    wire [1:0]  dma_desc_w128_addr_w; wire [127:0] dma_desc_w128_data_w; wire dma_desc_w128_en_w;

    // =========================================================================
    // SYSTOLIC datapath (the only compute)
    // =========================================================================
    wire        sys_dma_req_w;  wire [1:0] sys_dma_tgt_w;
    wire [63:0] sys_dma_addr_w; wire [23:0] sys_dma_bytes_w;
    wire        sys_start_w, sys_irq_w;
    // CONFIG-MUX (netseq): network_seq parses per-layer dims from the DDR descriptor and exposes them
    // as cur_cin/cout/hw; the engine sees those during is_netseq. sys_wb/actb/nwmem auto-derive.
    wire [9:0]  sys_cout = is_netseq ? ns_cur_cout : reg_cout[9:0];
    wire [13:0] sys_cin  = is_netseq ? ns_cur_cin  : reg_cin[13:0];   // conv3x3 im2col Cin'=9*Cin up to ~4600
    wire [9:0]  sys_hw   = is_netseq ? ns_cur_hw   : reg_hin[9:0];    // host sets HIN = HW (H*W)
    wire [9:0]  sys_wb   = (sys_hw + SYS_N-1) >> SYS_LOGN;
    reg  [23:0] sys_actb;                               // H3: registered (config-stable, breaks mult path)
    always @(posedge aclk) sys_actb <= sys_cin * sys_wb * SYS_N;

    // ── conv3x3 on-chip config (LAYER_TYPE=5): HIN=Hp|Wp<<10, WIN=Ho|Wo<<10, CIN=real Cin ──
    wire [9:0]  c3_Hp = reg_hin[9:0], c3_Wp = reg_hin[19:10];
    wire [9:0]  c3_Ho = reg_win[9:0], c3_Wo = reg_win[19:10];
    // H3 200MHz: REGISTER the config multiply chain (Hp*Wp -> WBin -> actb, + nwmem). These are stable
    // per-layer (written before ap_start, host gaps >> 3 cycles) so registering is functionally safe and
    // breaks the long combinational multiply path (c3_HpWp -> dma_req_addr, was 5.3ns logic).
    reg [19:0] c3_HpWp; reg [9:0] c3_WBin, c3_WBrow; reg [23:0] c3_actb; reg [SYS_MAW-1:0] c3_nwmem;
    always @(posedge aclk) begin
        c3_HpWp  <= c3_Hp * c3_Wp;
        c3_WBin  <= (c3_HpWp + 20'd15) >> 4;                // amem words/channel (uses registered HpWp)
        c3_WBrow <= (c3_Wo + 10'd15) >> 4;                  // omem words/output row
        c3_actb  <= sys_cin * c3_WBin * SYS_N;              // raw act bytes (uses registered WBin)
        c3_nwmem <= ((sys_cout + SYS_N-1) >> SYS_LOGN) * (sys_cin * 14'd9);
    end
    wire c1_done_w, c3_done_w;
    wire sys_done_w = is_sp ? sp_done_w : (is_c3any ? c3_done_w : c1_done_w); // systolic_seq waits the active engine (sp/c3/c1)

    // ── Network sequencer (LT14, DESCRIPTOR mode Rev132): desc_base=reg_in_addr, n_layers=reg_cin[3:0].
    // Per-layer config read from DDR descriptor table; ZERO new AXI registers (see addr-wall finding).
    network_seq u_netseq (
        .aclk(aclk), .aresetn(aresetn), .ap_start(ap_start & is_netseq),
        .desc_base(reg_in_addr), .n_layers(reg_cin[3:0]),
        .dma_req(ns_dma_req_w), .dma_req_tgt(ns_dma_tgt_w),
        .dma_req_addr(ns_dma_addr_w), .dma_req_bytes(ns_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .dma_desc_w128_en(dma_desc_w128_en_w), .dma_desc_w128_addr(dma_desc_w128_addr_w),
        .dma_desc_w128_data(dma_desc_w128_data_w),
        .sys_start(ns_sys_start_w), .sys_done(c1_done_w),
        .cur_cin(ns_cur_cin), .cur_cout(ns_cur_cout), .cur_hw(ns_cur_hw),
        .cur_layer(ns_cur_layer), .busy(ns_busy_w), .done_latch(ns_done_latch_w), .dbg_state()
    );

    systolic_seq u_sys_seq (
        .aclk(aclk), .aresetn(aresetn), .ap_start(ap_start & ~is_dwany & ~eng_lseq & ~is_xc & ~is_padxc & ~is_sp_lseq & ~is_c3_lseq & ~is_c1c & ~is_netseq),
        .skip_act(is_c1na | is_c3na),
        .act_addr(reg_in_addr), .wt_addr(reg_wt_addr), .par_addr(64'd0), .out_addr(reg_out_addr),
        .act_bytes(is_c3any ? c3_actb : sys_actb), .wt_bytes(reg_wt_bytes), .par_bytes(24'd0), .out_bytes(reg_out_bytes),
        .dma_req(sys_dma_req_w), .dma_req_tgt(sys_dma_tgt_w),
        .dma_req_addr(sys_dma_addr_w), .dma_req_bytes(sys_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .sys_start(sys_start_w), .sys_done(sys_done_w),
        .busy(sys_busy_w), .irq(sys_irq_w), .dbg_state()
    );
    always @(posedge aclk)
        if (!aresetn) sys_done_latch_r <= 1'b0;
        else if (ap_start) sys_done_latch_r <= 1'b0;
        else if (sys_irq_w) sys_done_latch_r <= 1'b1;

    // conv1x1 params travel inside the WT blob ([wmem][pmem]) → par_tile_bytes=0. All config maps to
    // existing B0 layer regs: n_hwt←n_pos, n_cot←n_ocb, act_tile←act_chunk, wt_tile←wt_blk_bytes,
    // out_tile←out_bytes. Tiles are uniform (host pads Cout/HW to tile multiples).
    systolic_layer_seq u_sys_lseq (
        .aclk(aclk), .aresetn(aresetn),
        .layer_mode(eng_lseq), .ap_start(ap_start), .sys_done(c1_done_w),
        .n_hwt(reg_n_pos), .n_cot(reg_n_ocb),
        .act_tile_bytes(reg_act_chunk), .wt_tile_bytes(reg_wt_blk_bytes),
        .par_tile_bytes(24'd0), .out_tile_bytes(reg_out_bytes),
        .act_base(reg_in_addr), .wt_base(reg_wt_addr), .par_base(64'd0), .out_base(reg_out_addr),
        .dma_req(lseq_dma_req_w), .dma_req_tgt(lseq_dma_tgt_w),
        .dma_req_addr(lseq_dma_addr_w), .dma_req_bytes(lseq_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .sys_start(lseq_start_w),
        .busy(lseq_busy_w), .irq(lseq_irq_w), .ls_done_latch(lseq_done_latch_w),
        .ls_busy(), .dbg_state()
    );

    // ── I-SPEC on-chip layer sequencer (LAYER_TYPE=12 + LAYER_MODE=1) ─────────────
    // act_bytes/out_bytes = Cin*16 / Cout*16 (per-tile DMA); strides are 4KB-aligned (host-set regs:
    // act_chunk=act_stride, wt_blk_bytes=out_stride, n_pos=n_st, wt_bytes=COO blob).
    spec_layer_seq u_sp_lseq (
        .aclk(aclk), .aresetn(aresetn),
        .layer_mode(is_sp_lseq), .ap_start(ap_start), .sp_done(sp_done_w),
        .n_st(reg_n_pos),
        .act_stride(reg_act_chunk), .act_bytes({6'd0, sys_cin, 4'd0}),    // 24-bit = Cin*16
        .wt_bytes(reg_wt_bytes),
        .out_stride(reg_wt_blk_bytes), .out_bytes({10'd0, sys_cout, 4'd0}), // 24-bit = Cout*16 (was 22→Z)
        .act_base(reg_in_addr), .wt_base(reg_wt_addr), .out_base(reg_out_addr),
        .dma_req(spl_dma_req_w), .dma_req_tgt(spl_dma_tgt_w),
        .dma_req_addr(spl_dma_addr_w), .dma_req_bytes(spl_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .sp_start(spl_start_w),
        .busy(spl_busy_w), .irq(spl_irq_w), .ls_done_latch(spl_done_latch_w), .dbg_state()
    );

    // ── conv3x3 on-chip layer sequencer instantiation ──────────────────────────
    conv3x3_layer_seq u_c3_lseq (
        .aclk(aclk), .aresetn(aresetn),
        .layer_mode(is_c3_lseq), .ap_start(ap_start & is_c3),
        .c3_done(c3_done_w),
        .n_bands(reg_n_pos[7:0]), .n_cout(reg_n_ocb[7:0]),
        .act_bytes(reg_act_chunk), .wt_bytes(reg_wt_bytes), .out_bytes(reg_out_bytes),
        .act_base(reg_in_addr), .wt_base(reg_wt_addr), .out_base(reg_out_addr),
        .dma_req(c3seq_dma_req_w), .dma_req_tgt(c3seq_dma_tgt_w),
        .dma_req_addr(c3seq_dma_addr_w), .dma_req_bytes(c3seq_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .c3seq_start(c3seq_start_w),
        .busy(c3seq_busy_w), .c3seq_done_latch(c3seq_done_latch_w), .dbg_state()
    );

    // engine kick + systolic-memory DMA control: muxed lseq (layer mode) vs sys_seq (per-tile)
    // is_c3_lseq: c3seq owns DMA + kicks c3 engine; sys_start_w is blocked (0) to prevent systolic_seq
    // from starting any engine while c3seq runs.
    wire        eng_start_w = is_netseq ? ns_sys_start_w : is_sp_lseq ? spl_start_w : is_c3_lseq ? 1'b0 : eng_lseq ? lseq_start_w : sys_start_w;
    // eng_dma_req/tgt: feed sys_a_cnt/sys_w_cnt/sys_cur_tgt counters — must include c3seq so that
    // amem write counter resets correctly at start of each ACT DMA and wmem at each WT DMA.
    wire        eng_dma_req = is_netseq ? ns_dma_req_w : is_sp_lseq ? spl_dma_req_w : is_c3_lseq ? c3seq_dma_req_w : eng_lseq ? lseq_dma_req_w : sys_dma_req_w;
    wire [1:0]  eng_dma_tgt = is_netseq ? ns_dma_tgt_w : is_sp_lseq ? spl_dma_tgt_w : is_c3_lseq ? c3seq_dma_tgt_w : eng_lseq ? lseq_dma_tgt_w : sys_dma_tgt_w;

    // wide-memory write address counters (count en pulses; reset at each transfer start)
    reg [1:0] sys_cur_tgt;
    reg [SYS_MAW-1:0] sys_a_cnt, sys_w_cnt;
    always @(posedge aclk) begin
        if (!aresetn) begin sys_a_cnt<=0; sys_w_cnt<=0; sys_cur_tgt<=0; end
        else begin
            if (eng_dma_req) begin
                sys_cur_tgt <= eng_dma_tgt;
                if (eng_dma_tgt==2'd0) sys_a_cnt<=0;
                if (eng_dma_tgt==2'd1) sys_w_cnt<=0;
            end
            if (dma_in_w128_en_w) sys_a_cnt <= sys_a_cnt + 1'b1;
            if (dma_wt_w128_en_w) sys_w_cnt <= sys_w_cnt + 1'b1;
        end
    end
    reg  [SYS_MAW-1:0] sys_nwmem;   // H3: registered (config-stable) — was reg_cout->mult->wmem-BWE worst path
    always @(posedge aclk) sys_nwmem <= ((sys_cout + SYS_N-1) >> SYS_LOGN) * sys_cin;
    wire sys_ldw_we = dma_wt_w128_en_w & (sys_w_cnt <  sys_nwmem);
    wire sys_ldp_we = dma_wt_w128_en_w & (sys_w_cnt >= sys_nwmem);

    // OUT serialise: omem 128-bit word -> byte (1-cyc read latency); OUT-chunk omem_base += 64/DMA
    wire [SYS_MAW-1:0] sys_o_rd_addr;
    wire [SYS_N*8-1:0] sys_o_rd_data;
    reg  [3:0] sys_byte_sel_d;
    always @(posedge aclk) sys_byte_sel_d <= dma_out_addr_w[3:0];
    wire [7:0] sys_out_data = sys_o_rd_data[sys_byte_sel_d*8 +: 8];
    reg [SYS_MAW-1:0] sys_omem_base;
    always @(posedge aclk) begin
        if (!aresetn) sys_omem_base <= 0;
        else if (eng_start_w || (c3seq_start_w & is_c3_lseq)) sys_omem_base <= 0;
        else if (dma_done_w && sys_cur_tgt==2'd2) sys_omem_base <= sys_omem_base + 14'd64;
    end
    // sys_o_rd_addr: DMA-out reads c1/c3 omem; during xcopy sweeps c3 omem (rebank); during padxc sweeps
    // c1 omem (pad_rebank source -> sys_o_rd_data). Both cross-engine copies own the shared read port.
    // c1_copy (LT13): on-chip c1 omem→amem. rd_addr sweeps c1 omem; c1_ld_a receives the copy.
    // Registered Cout×WB (n_words) for the copy counter; config-stable so register breaks mult path.
    reg [19:0] c1c_n_words;
    always @(posedge aclk) c1c_n_words <= {10'd0, sys_cout} * {10'd0, sys_wb};
    c1_copy u_c1c (
        .clk(aclk), .rst_n(aresetn),
        .start(ap_start & is_c1c), .n_words(c1c_n_words),
        .rd_addr(c1c_rd_addr_w), .rd_data(sys_o_rd_data),
        .wr_en(c1c_wr_en_w), .wr_addr(c1c_wr_addr_w), .wr_data(c1c_wr_data_w),
        .busy(c1c_busy_w), .done_latch(c1c_done_latch_w), .dbg_state()
    );
    // sys_o_rd_addr mux: c1_copy reads c1 omem (sys_o_rd_data) via c1c_rd_addr
    assign sys_o_rd_addr = is_dwoc ? dw_omem_rd_addr : is_padxc ? padxc_rd_addr : is_xc ? xc_rd_addr : is_c1c ? {c1c_rd_addr_w} : (sys_omem_base + {{(SYS_MAW-6){1'b0}}, dma_out_addr_w[9:4]});
    // c1 amem write port: DMA load, or cross-engine copy (xcopy/c1_copy), or rebank
    wire               c1_ld_a_we   = is_dwocamem ? dw_amem_we_w    : is_c1c ? c1c_wr_en_w   : is_xc ? xc_wr_en   : dma_in_w128_en_w;
    wire [SYS_MAW-1:0] c1_ld_a_addr = is_dwocamem ? dw_amem_waddr_w : is_c1c ? c1c_wr_addr_w : is_xc ? xc_wr_addr : sys_a_cnt;
    wire [SYS_N*8-1:0] c1_ld_a_data = is_dwocamem ? dw_amem_wdata_w : is_c1c ? c1c_wr_data_w : is_xc ? xc_wr_data : dma_in_w128_data_w;

    conv1x1_systolic_top #(.N(SYS_N),.LOGN(SYS_LOGN),.AW(8),.ACCW(32),.MAW(SYS_MAW)) u_sys_eng (
        .clk(aclk), .rst_n(aresetn),
        .ld_w_we(sys_ldw_we), .ld_w_addr(sys_w_cnt), .ld_w_data(dma_wt_w128_data_w),
        .ld_a_we(c1_ld_a_we), .ld_a_addr(c1_ld_a_addr), .ld_a_data(c1_ld_a_data),
        .ld_p_we(sys_ldp_we), .ld_p_addr(sys_w_cnt - sys_nwmem), .ld_p_data(dma_wt_w128_data_w),
        .start(eng_start_w & ~is_c3 & ~is_c3na & ~is_sp), .Cout(sys_cout), .Cin(sys_cin), .HW(sys_hw), .WB(sys_wb), .apply_relu(reg_apply_relu),
        .done(c1_done_w), .rd_o_addr(sys_o_rd_addr), .rd_o_data(sys_o_rd_data)
    );

    // ── I-SPEC sparse conv1x1 engine (LAYER_TYPE=12) ──────────────────────────
    // DMA fills: act stream → amem (ld_a_addr=sys_a_cnt), wt stream → coo (first sp_ncoo words)
    // then params. coo512 = 4×128-bit lanes per (ob16,mt); sp_ncoo = ob16_blks*m_tiles*NSPEC.
    wire [13:0] sp_mtiles = sys_cin >> 4;             // Cin/16
    wire [9:0]  sp_ob16   = (sys_cout + 15) >> 4;      // ceil(Cout/16)
    reg  [SYS_MAW-1:0] sp_ncoo;
    always @(posedge aclk) sp_ncoo <= sp_ob16 * sp_mtiles * 3'd4;  // 4 lanes/coo512 word
    wire sp_ldw_we = is_sp & dma_wt_w128_en_w & (sys_w_cnt <  sp_ncoo);
    wire sp_ldp_we = is_sp & dma_wt_w128_en_w & (sys_w_cnt >= sp_ncoo);
    wire [127:0] sp_rd_o_data;
    reg  [7:0]   sp_out_data;
    always @(posedge aclk) sp_out_data <= sp_rd_o_data[dma_out_addr_w[3:0]*8 +: 8];

    // CDEPTH=1024 (max ob16*m_tiles=32*32); ODEPTH=512 (omem now 128-bit/oc word, max Cout); ADEPTH=64 (Cin/16).
    spec_conv1x1_top #(.MAW(SYS_MAW),.NSPEC(4),.ADEPTH(64),.CDEPTH(1024),.ODEPTH(512)) u_sp_eng (
        .clk(aclk), .rst_n(aresetn),
        .ld_a_we(is_sp & dma_in_w128_en_w), .ld_a_addr(sys_a_cnt), .ld_a_data(dma_in_w128_data_w),
        .ld_w_we(sp_ldw_we), .ld_w_addr(sys_w_cnt), .ld_w_data(dma_wt_w128_data_w),
        .ld_p_we(sp_ldp_we), .ld_p_addr(sys_w_cnt - sp_ncoo), .ld_p_data(dma_wt_w128_data_w),
        .start(eng_start_w & is_sp), .Cout(sys_cout), .Cin(sys_cin), .HW(sys_hw),
        .n_cfg(reg_n_config), .apply_relu(reg_apply_relu),  // host writes N into N_CONFIG reg (0x080)
        .done(sp_done_w),
        .rd_o_addr(sys_omem_base + {{(SYS_MAW-6){1'b0}}, dma_out_addr_w[9:4]}), .rd_o_data(sp_rd_o_data)
    );

    // ── conv3x3 on-chip im2col engine (LAYER_TYPE=5): raw act, barrel-shift windowed feed ──
    wire [SYS_MAW-1:0] c3_nwmem_m = (c3_nwmem > {SYS_MAW{1'b1}}) ? {SYS_MAW{1'b1}} : c3_nwmem;
    wire c3_ldw_we = dma_wt_w128_en_w & (sys_w_cnt <  c3_nwmem_m);
    wire c3_ldp_we = dma_wt_w128_en_w & (sys_w_cnt >= c3_nwmem_m);
    // c3 amem write port: DMA load, or (during padxc) the pad-rebanked c1 omem (cross-engine, no DDR).
    wire               c3_ld_a_we   = is_padxc ? padxc_wr_en   : dma_in_w128_en_w;
    wire [SYS_MAW-1:0] c3_ld_a_addr = is_padxc ? padxc_wr_addr : sys_a_cnt;
    wire [SYS_N*8-1:0] c3_ld_a_data = is_padxc ? padxc_wr_data : dma_in_w128_data_w;
    conv3x3_systolic_top #(.N(SYS_N),.LOGN(SYS_LOGN),.AW(8),.ACCW(32),.MAW(SYS_MAW),.ADEPTH(8192)) u_c3_eng (
        .clk(aclk), .rst_n(aresetn),
        .ld_w_we(c3_ldw_we), .ld_w_addr(sys_w_cnt), .ld_w_data(dma_wt_w128_data_w),
        .ld_a_we(c3_ld_a_we), .ld_a_addr(c3_ld_a_addr), .ld_a_data(c3_ld_a_data),
        .ld_p_we(c3_ldp_we), .ld_p_addr(sys_w_cnt - c3_nwmem_m), .ld_p_data(dma_wt_w128_data_w),
        .start((is_c3_lseq ? c3seq_start_w : eng_start_w) & is_c3any), .Cout(sys_cout), .Cin(sys_cin),
        .Hp(c3_Hp), .Wp(c3_Wp), .Ho(c3_Ho), .Wo(c3_Wo), .WBin(c3_WBin), .WBrow(c3_WBrow),
        .apply_relu(reg_apply_relu),
        .done(c3_done_w), .rd_o_addr(sys_o_rd_addr), .rd_o_data(c3_o_rd_data)
    );
    wire [7:0] c3_out_data = c3_o_rd_data[sys_byte_sel_d*8 +: 8];

    // =========================================================================
    // dw_layer (LAYER_TYPE=4): on-chip depthwise channel loop, shares the dma_engine.
    // Config packed in regs: CIN=C, HIN[9:0]=Hp/[19:10]=Wp, WIN[9:0]=Ho/[19:10]=Wo, COUT[1:0]=stride.
    // =========================================================================
    wire [9:0] dw_Hp = reg_hin[9:0],  dw_Wp = reg_hin[19:10];
    wire [9:0] dw_Ho = reg_win[9:0],  dw_Wo = reg_win[19:10];
    wire [8:0] dw_C  = reg_cin[8:0];
    wire [1:0] dw_str= reg_cout[1:0];
    // per-channel DMA base must avoid AXI 4KB-boundary crossing. Stride = 1024 if the channel fits
    // in 1024B (1024|4096 => a <=1024B burst from a 1024-aligned base stays inside one 4KB page),
    // else round up to 4096 (4KB-aligned base; the dma_engine splits internal 4KB crossings like the
    // systolic). 1024 tier avoids the 4-18x over-DMA on the small-spatial many-channel layers (13²/26²).
    wire [23:0] dw_act_raw = dw_Hp * dw_Wp;
    wire [23:0] dw_out_raw = dw_Ho * dw_Wo;
    // H3: register the channel-byte config (mult + 4KB-align clamp) — config-stable, breaks the
    // dw_Hp*dw_Wp -> dma_req_addr multiply path (u_dw worst path at 200MHz).
    reg  [23:0] dw_act_cb, dw_out_cb;
    always @(posedge aclk) begin
        dw_act_cb <= (dw_act_raw <= 24'd1024) ? 24'd1024 : ((dw_act_raw + 24'd4095) & 24'hFFF000);
        dw_out_cb <= (dw_out_raw <= 24'd1024) ? 24'd1024 : ((dw_out_raw + 24'd4095) & 24'hFFF000);
    end
    wire [23:0] dw_wt_b   = {11'd0, dw_C, 4'd0};       // C*16
    // T2-3 dwoc: conv1x1-output WB (source words/channel in c1 omem) = ceil(Ho*Wo/16); c1 omem read addr.
    reg  [SYS_MAW-1:0] dwoc_WBsrc;
    always @(posedge aclk) dwoc_WBsrc <= (dw_out_raw + 24'd15) >> 4;
    dw_layer #(.ADW(15),.AW128(11),.ODW(15),.MAXWP(512),.PRMAW(SYS_MAW)) u_dw (
        .aclk(aclk), .aresetn(aresetn), .ap_start(ap_start & is_dwany),
        .C(dw_C), .Hp(dw_Hp), .Wp(dw_Wp), .Ho(dw_Ho), .Wo(dw_Wo), .stride(dw_str),
        .apply_relu(reg_apply_relu),
        .dwoc(is_dwoc), .omem_WBsrc(dwoc_WBsrc),
        .omem_rd_addr(dw_omem_rd_addr), .omem_rd_data(sys_o_rd_data),
        // C/T2-4: on-chip OUTPUT -> c1 amem (out_WB == dwoc_WBsrc == ceil(Ho*Wo/16), same value)
        .dwout_amem(is_dwocamem), .out_WB(dwoc_WBsrc),
        .amem_we(dw_amem_we_w), .amem_waddr(dw_amem_waddr_w), .amem_wdata(dw_amem_wdata_w),
        .act_addr(reg_in_addr), .wt_addr(reg_wt_addr), .out_addr(reg_out_addr),
        .act_chan_bytes(dw_act_cb), .out_chan_bytes(dw_out_cb), .wt_bytes(dw_wt_b),
        .dma_req(dw_dma_req_w), .dma_req_tgt(dw_dma_tgt_w),
        .dma_req_addr(dw_dma_addr_w), .dma_req_bytes(dw_dma_bytes_w),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .dma_in_w128_en(dma_in_w128_en_w), .dma_in_w128_data(dma_in_w128_data_w),
        .dma_wt_w128_en(dma_wt_w128_en_w), .dma_wt_w128_data(dma_wt_w128_data_w),
        .dma_out_addr(dma_out_addr_w), .dma_out_data(dw_out_data_w),
        .busy(dw_busy_w), .irq(dw_irq_w), .dbg_state()
    );
    always @(posedge aclk)
        if (!aresetn) dw_done_latch_r <= 1'b0;
        else if (ap_start & is_dwany) dw_done_latch_r <= 1'b0;
        else if (dw_irq_w) dw_done_latch_r <= 1'b1;

    // ── Tier 2 rebank (LAYER_TYPE=6): re-bank c3 omem (ROW-padded, via sys_o_rd_addr->c3_o_rd_data)
    // into c1 amem (CONTIGUOUS, via c1_ld_a port). Handles general Wo. Host: conv3x3(LT5)->c3 omem;
    // then LT6 rebank->c1 amem; then conv1x1(LT3, no ACT DMA). Dims from regs (Cout/Ho/WBrow/wo_last).
    wire [4:0] c3_wo_last = c3_Wo - ((c3_WBrow - 10'd1) << 4);     // valid lanes in last word of row (1..16)
    rebank #(.MAW(SYS_MAW), .N(SYS_N), .AW(8)) u_rebank (
        .clk(aclk), .rst_n(aresetn), .start(ap_start & is_xc),
        .Cout(sys_cout), .Ho(c3_Ho), .WBrow(c3_WBrow), .wo_last(c3_wo_last),
        .rd_addr(xc_rd_addr), .rd_data(c3_o_rd_data),
        .wr_en(xc_wr_en), .wr_addr(xc_wr_addr), .wr_data(xc_wr_data),
        .busy(xc_busy), .done_latch(xc_done_latch)
    );

    // ── Tier 2 pad_rebank (LAYER_TYPE=8): PAD-WHILE-COPY c1 omem (CONTIGUOUS, via sys_o_rd_addr->
    // sys_o_rd_data) into c3 amem (PADDED, via c3_ld_a port), zero borders. Feeds a conv3x3/dw from a
    // conv1x1 output with NO DDR round-trip. Host: conv1x1(LT3)->c1 omem; then LT8 pad_rebank->c3 amem;
    // then conv3x3(LT5)/dw. Dims: Cout=sys_cout, Ho/Wo=reg_win (conv1x1 out dims), pad=1 (3x3).
    reg [19:0] padxc_HoWo; reg [SYS_MAW-1:0] padxc_WBsrc;     // ceil(Ho*Wo/16) src words/ch (registered)
    always @(posedge aclk) begin
        padxc_HoWo  <= c3_Ho * c3_Wo;
        padxc_WBsrc <= (padxc_HoWo + 20'd15) >> 4;
    end
    pad_rebank #(.MAW(SYS_MAW), .N(SYS_N), .AW(8)) u_pad_rebank (
        .clk(aclk), .rst_n(aresetn), .start(ap_start & is_padxc),
        .Cout(sys_cout), .Ho(c3_Ho), .Wo(c3_Wo), .WBsrc(padxc_WBsrc), .src_off({SYS_MAW{1'b0}}),
        .rd_addr(padxc_rd_addr), .rd_data(sys_o_rd_data),
        .wr_en(padxc_wr_en), .wr_addr(padxc_wr_addr), .wr_data(padxc_wr_data),
        .busy(padxc_busy), .done_latch(padxc_done_latch)
    );

    // dma_engine input mux (dw | conv1x1 layer-mode | per-tile systolic share the single engine)
    wire        dma_req_mux   = is_netseq ? ns_dma_req_w   : is_sp_lseq ? spl_dma_req_w   : is_c3_lseq ? c3seq_dma_req_w   : is_dwany ? dw_dma_req_w   : (eng_lseq ? lseq_dma_req_w   : sys_dma_req_w);
    wire [1:0]  dma_tgt_mux   = is_netseq ? ns_dma_tgt_w   : is_sp_lseq ? spl_dma_tgt_w   : is_c3_lseq ? c3seq_dma_tgt_w   : is_dwany ? dw_dma_tgt_w   : (eng_lseq ? lseq_dma_tgt_w   : sys_dma_tgt_w);
    wire [63:0] dma_addr_mux  = is_netseq ? ns_dma_addr_w  : is_sp_lseq ? spl_dma_addr_w  : is_c3_lseq ? c3seq_dma_addr_w  : is_dwany ? dw_dma_addr_w  : (eng_lseq ? lseq_dma_addr_w  : sys_dma_addr_w);
    wire [23:0] dma_bytes_mux = is_netseq ? ns_dma_bytes_w : is_sp_lseq ? spl_dma_bytes_w : is_c3_lseq ? c3seq_dma_bytes_w : is_dwany ? dw_dma_bytes_w : (eng_lseq ? lseq_dma_bytes_w : sys_dma_bytes_w);
    wire [7:0]  dma_out_mux   = is_dwany ? dw_out_data_w  : is_sp ? sp_out_data : (is_c3any ? c3_out_data : sys_out_data);
    // WIDE OUTPUT (Rev130): 128-bit omem word per beat for sys/c3/spec (omem is 128-bit). dw → byte path.
    // omem word read uses the SAME sys_o_rd_addr mapping (sys_omem_base + dma_out_addr[9:4]); the engine
    // streams 1 word/beat. The 128-bit data is already produced by each engine's omem read port.
    wire [SYS_N*8-1:0] dma_out_data128_mux = is_sp ? sp_rd_o_data : (is_c3any ? c3_o_rd_data : sys_o_rd_data);
    wire        dma_out_wide = ~is_dwany;   // dw output is per-channel byte-strided → keep byte path

    // =========================================================================
    // dma_engine
    // =========================================================================
    dma_engine u_dma (
        .aclk(aclk), .aresetn(aresetn),
        .dma_req(dma_req_mux), .dma_req_tgt(dma_tgt_mux),
        .dma_req_addr(dma_addr_mux), .dma_req_bytes(dma_bytes_mux),
        .dma_active(dma_active_w), .dma_done(dma_done_w),
        .m_axi_awaddr(m_axi_awaddr), .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot),
        .m_axi_awlock(m_axi_awlock), .m_axi_awqos(m_axi_awqos),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata), .m_axi_wstrb(m_axi_wstrb), .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst), .m_axi_arcache(m_axi_arcache), .m_axi_arprot(m_axi_arprot),
        .m_axi_arlock(m_axi_arlock), .m_axi_arqos(m_axi_arqos),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready),
        .dma_in_addr(dma_in_addr_w), .dma_in_data(dma_in_data_w), .dma_in_wr_en(dma_in_wr_en_w),
        .dma_in_w128_addr(dma_in_w128_addr_w), .dma_in_w128_data(dma_in_w128_data_w), .dma_in_w128_en(dma_in_w128_en_w),
        .dma_wt_addr(dma_wt_addr_w), .dma_wt_data(dma_wt_data_w), .dma_wt_wr_en(dma_wt_wr_en_w),
        .dma_wt_w128_addr(dma_wt_w128_addr_w), .dma_wt_w128_data(dma_wt_w128_data_w), .dma_wt_w128_en(dma_wt_w128_en_w),
        .dma_out_addr(dma_out_addr_w), .dma_out_data(dma_out_mux),
        .out_wide(dma_out_wide), .dma_out_data128(dma_out_data128_mux),
        .dma_desc_w128_addr(dma_desc_w128_addr_w), .dma_desc_w128_data(dma_desc_w128_data_w),
        .dma_desc_w128_en(dma_desc_w128_en_w)
    );
endmodule

// =============================================================================================
// Tier 2 rebank (inlined here so it ships in the cnn_acc_top fileset; standalone src/rebank.v +
// src/xcopy.v are kept for the unit testbenches). conv3x3 omem (ROW-padded) -> conv1x1 amem
// (CONTIGUOUS), general Wo: barrel accumulator re-packs valid pixels row-by-row, ~2 cyc/word.
// =============================================================================================
module rebank #(
    parameter integer MAW = 14,
    parameter integer N   = 16,
    parameter integer AW  = 8
) (
    input  wire             clk, rst_n, start,
    input  wire [9:0]       Cout, Ho, WBrow,
    input  wire [4:0]       wo_last,
    output wire [MAW-1:0]   rd_addr,
    input  wire [N*AW-1:0]  rd_data,
    output reg              wr_en,
    output reg  [MAW-1:0]   wr_addr,
    output reg  [N*AW-1:0]  wr_data,
    output wire             busy,
    output reg              done_latch
);
    localparam S_IDLE=2'd0, S_RD=2'd1, S_PROC=2'd2, S_FLUSH=2'd3;
    reg [1:0]  st;
    reg [9:0]  co, ho, wblk;
    reg [MAW-1:0] ra, out_addr;
    reg [2*N*AW-1:0] acc;
    reg [5:0]  fill;

    assign busy    = (st != S_IDLE);
    assign rd_addr = ra;

    wire last_in_row = (wblk == WBrow - 1);
    wire chan_end    = last_in_row && (ho == Ho-1);
    wire [4:0] vcnt  = last_in_row ? wo_last : N[4:0];
    wire [N*AW-1:0] vmask = (vcnt >= N) ? {N*AW{1'b1}} : ~({N*AW{1'b1}} << (vcnt*AW));
    wire [N*AW-1:0] mword = rd_data & vmask;
    wire [2*N*AW-1:0] acc_ins = acc | ({{(N*AW){1'b0}}, mword} << (fill*AW));
    wire [5:0] fill_after = fill + vcnt;
    wire emit_now = (fill_after >= N);

    always @(posedge clk) begin
        if (!rst_n) begin
            st<=S_IDLE; co<=0; ho<=0; wblk<=0; ra<=0; out_addr<=0; acc<=0; fill<=0;
            wr_en<=0; wr_addr<=0; wr_data<=0; done_latch<=0;
        end else begin
            wr_en<=0;
            case (st)
                S_IDLE: if (start) begin
                    co<=0; ho<=0; wblk<=0; ra<=0; out_addr<=0; acc<=0; fill<=0; done_latch<=0;
                    st<=S_RD;
                end
                S_RD: st<=S_PROC;
                S_PROC: begin
                    if (emit_now) begin
                        wr_en<=1; wr_addr<=out_addr; wr_data<=acc_ins[N*AW-1:0]; out_addr<=out_addr+1'b1;
                        acc<=acc_ins >> (N*AW); fill<=fill_after - N;
                    end else begin
                        acc<=acc_ins; fill<=fill_after;
                    end
                    ra<=ra+1'b1;
                    if (last_in_row) begin
                        wblk<=0;
                        if (ho==Ho-1) begin ho<=0; co<=co+1'b1; end else ho<=ho+1'b1;
                    end else wblk<=wblk+1'b1;
                    st <= chan_end ? S_FLUSH : S_RD;
                end
                S_FLUSH: begin
                    if (fill != 0) begin
                        wr_en<=1; wr_addr<=out_addr; wr_data<=acc[N*AW-1:0]; out_addr<=out_addr+1'b1;
                    end
                    acc<=0; fill<=0;
                    if (co == Cout) begin done_latch<=1; st<=S_IDLE; end
                    else st<=S_RD;
                end
            endcase
        end
    end
endmodule

// ============================================================================================
// pad_rebank (T2-2) -- INLINED (IP-fileset only rebuilds cnn_acc_top.v). Standalone copy in
// ip_repo/src/pad_rebank.v is for the unit TB (tb_pad_rebank.v, all 8 cases err=0). PAD-WHILE-COPY:
// conv1x1 omem (banked CONTIGUOUS) -> conv3x3 amem (banked PADDED, zero borders). Inverse of rebank.
// SOURCE omem[c*WBsrc+(ho*Wo+wo)>>4][&15]=O; DEST amem[c*WBin+(hp*Wp+wp)>>4][&15], interior=O[hp-1][wp-1]
// (1<=hp<=Ho,1<=wp<=Wo), border=0, pad=1. Interior elements appear in linear src order -> one src_lin
// counter + a cached 16-lane source word; barrel-pack 16/word (zeros at borders) + flush partial/channel.
// ============================================================================================
module pad_rebank #(
    parameter integer MAW = 14,
    parameter integer N   = 16,
    parameter integer AW  = 8
) (
    input  wire             clk, rst_n, start,
    input  wire [9:0]       Cout, Ho, Wo,
    input  wire [MAW-1:0]   WBsrc,
    input  wire [MAW-1:0]   src_off,
    output wire [MAW-1:0]   rd_addr,
    input  wire [N*AW-1:0]  rd_data,
    output reg              wr_en,
    output reg  [MAW-1:0]   wr_addr,
    output reg  [N*AW-1:0]  wr_data,
    output wire             busy,
    output reg              done_latch
);
    localparam LOGN = 4;
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_FETCH=2'd2, S_FLUSH=2'd3;
    reg [1:0] st;
    reg [9:0] co, hp, wp;
    reg [MAW-1:0] src_lin, src_base, dst_word, ra, swidx;
    reg [N*AW-1:0] wacc, sword;
    reg [4:0] wfill;
    reg swvalid;

    wire [9:0] Hp = Ho + 10'd2;
    wire [9:0] Wp = Wo + 10'd2;
    wire interior = (hp>=10'd1) && (hp<=Ho) && (wp>=10'd1) && (wp<=Wo);
    wire [MAW-1:0] need_word = src_base + (src_lin >> LOGN);
    wire [LOGN-1:0] src_lane = src_lin[LOGN-1:0];
    wire hit = swvalid && (swidx == need_word);
    wire [AW-1:0] elem = interior ? sword[src_lane*AW +: AW] : {AW{1'b0}};
    wire last_col  = (wp == Wp - 10'd1);
    wire last_row  = (hp == Hp - 10'd1);
    wire chan_last = last_col && last_row;
    wire emit_full = (wfill == N-1);

    reg [N*AW-1:0] wacc_ins;
    always @* begin
        wacc_ins = wacc;
        wacc_ins[wfill*AW +: AW] = elem;
    end

    assign busy    = (st != S_IDLE);
    assign rd_addr = (st==S_RUN) ? need_word : ra;

    always @(posedge clk) begin
        if (!rst_n) begin
            st<=S_IDLE; wr_en<=0; done_latch<=0;
            co<=0; hp<=0; wp<=0; src_lin<=0; src_base<=0; dst_word<=0;
            wacc<=0; wfill<=0; swvalid<=0; ra<=0; swidx<=0; sword<=0; wr_addr<=0; wr_data<=0;
        end else begin
            wr_en<=0;
            case (st)
                S_IDLE: if (start) begin
                    co<=0; hp<=0; wp<=0; src_lin<=0; src_base<=src_off; dst_word<=0;
                    wacc<=0; wfill<=0; swvalid<=0; done_latch<=0; st<=S_RUN;
                end
                S_RUN: begin
                    if (interior && !hit) begin
                        ra <= need_word; st <= S_FETCH;
                    end else begin
                        if (emit_full) begin
                            wr_en<=1; wr_addr<=dst_word; wr_data<=wacc_ins; dst_word<=dst_word+1'b1;
                            wacc<=0; wfill<=0;
                        end else begin
                            wacc<=wacc_ins; wfill<=wfill+1'b1;
                        end
                        if (interior) src_lin<=src_lin+1'b1;
                        if (chan_last)      st<=S_FLUSH;
                        else if (last_col)  begin wp<=10'd0; hp<=hp+1'b1; end
                        else                wp<=wp+1'b1;
                    end
                end
                S_FETCH: begin
                    sword<=rd_data; swidx<=ra; swvalid<=1'b1; st<=S_RUN;
                end
                S_FLUSH: begin
                    if (wfill != 5'd0) begin
                        wr_en<=1; wr_addr<=dst_word; wr_data<=wacc; dst_word<=dst_word+1'b1;
                    end
                    wacc<=0; wfill<=0; swvalid<=0;
                    if (co == Cout-10'd1) begin done_latch<=1; st<=S_IDLE; end
                    else begin
                        co<=co+1'b1; hp<=10'd0; wp<=10'd0; src_lin<=0; src_base<=src_base+WBsrc; st<=S_RUN;
                    end
                end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
