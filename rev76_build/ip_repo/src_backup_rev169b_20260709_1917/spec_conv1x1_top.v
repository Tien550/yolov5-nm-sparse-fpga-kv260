// spec_conv1x1_top.v -- I-SPEC (paper-faithful) conv1x1 engine, N_SPEC=4 (16 oc/pass).
// B2: 4 SPECs share ONE depth_shuffle (same activations + same NZ indices, structured N:16),
// each SPEC has its own COO weight stream (4 oc) → 16 output channels in parallel.
// 4 SPEC × G*P(=64) = 256 MAC (= systolic budget). Port style mirrors conv1x1_systolic_top.
//
// Processing (one spatial-tile of 16 positions in amem, reused across oc-blocks-of-16):
//   for ob16 in ceil(Cout/16):  acc_clr; for mt in M-tiles: load DS(amem[mt]) + wt_load(coo512[ob16][mt]) + drain N;
//                               requant 16oc×16pos psum -> omem.
//
// Memory load (128-bit DMA lanes):
//   amem  : ADEPTH words 2048-bit (1 MM-tile=256B). ld_a_addr={mt,lane4}; 16 lanes/word.
//   coo512: per (ob16,mt) 512-bit = 4 SPEC words. ld_w_addr={(ob16*MT+mt),lane2}; 4 lanes/word.
//           specword[s] = {idx(12b @11:0), g0vals(@35:12), g1(@59:36), g2(@83:60), g3(@107:84)}.
//   parm  : 2 oc per 128-bit word (like systolic ld_p).

`timescale 1ns/1ps
`include "cnn_defines.vh"

module spec_conv1x1_top #(
    parameter integer MAW    = 14,
    parameter integer NSPEC  = 4,
    parameter integer ADEPTH = 64,    // M-tiles per spatial-tile
    parameter integer CDEPTH = 4096,  // coo512 words = oc16_blocks * M-tiles
    parameter integer ODEPTH = 16384  // omem bytes
)(
    input  wire                 clk,
    input  wire                 rst_n,
    input  wire                 ld_a_we,
    input  wire [MAW-1:0]       ld_a_addr,   // {mt, lane[3:0]}
    input  wire [127:0]         ld_a_data,
    input  wire                 ld_w_we,
    input  wire [MAW-1:0]       ld_w_addr,   // {coo_idx, lane[1:0]}
    input  wire [127:0]         ld_w_data,
    input  wire                 ld_p_we,
    input  wire [MAW-1:0]       ld_p_addr,
    input  wire [127:0]         ld_p_data,
    input  wire                 start,
    input  wire [9:0]           Cout,        // multiple of 16
    input  wire [13:0]          Cin,         // multiple of 16
    input  wire [9:0]           HW,          // positions this tile (==16)
    input  wire [3:0]           n_cfg,
    input  wire                 apply_relu,
    output reg                  done,
    input  wire [MAW-1:0]       rd_o_addr,
    output wire [127:0]         rd_o_data
);
    localparam P = `P, G = `G, ACCW = `ACC_W;

    // ── memories ─────────────────────────────────────────────────────────────
    // amem is 2048-bit wide but shallow (64) → distributed LUTRAM (avoids 29-BRAM width waste).
    (* ram_style="distributed" *) reg [P*`M*`DATA_W-1:0] amem [0:ADEPTH-1];  // 2048-bit MM-tiles
    (* ram_style="block" *) reg [NSPEC*128-1:0] coo [0:CDEPTH-1];            // 512-bit (4 SPEC words)
    (* ram_style="block" *) reg [127:0]         omem[0:ODEPTH-1];            // 128-bit/oc (16 pos), idx=oc
    // Requant params split into EVEN/ODD oc banks: each bank single-write-port → infers as
    // distributed RAM (LUTRAM) instead of dissolving 1024-deep array to FF + 1024:1 mux (~40k LUT).
    (* ram_style="distributed" *) reg [`SCALE_W-1:0]  rq_m0_e   [0:511], rq_m0_o   [0:511];
    (* ram_style="distributed" *) reg [`SHIFT_W-1:0]  rq_shift_e[0:511], rq_shift_o[0:511];
    (* ram_style="distributed" *) reg signed [`ACC_W-1:0] rq_bias_e [0:511], rq_bias_o [0:511];

    integer li;
    always @(posedge clk) begin
        if (ld_a_we) amem[ld_a_addr[MAW-1:4]][ld_a_addr[3:0]*128 +: 128] <= ld_a_data;
        if (ld_w_we) coo[ld_w_addr[MAW-1:2]][ld_w_addr[1:0]*128 +: 128]  <= ld_w_data;
        if (ld_p_we) begin   // ld_p_addr = oc-pair index; even oc=2*addr, odd oc=2*addr+1
            rq_m0_e   [ld_p_addr[8:0]] <= ld_p_data[15:0];
            rq_shift_e[ld_p_addr[8:0]] <= ld_p_data[20:16];
            rq_bias_e [ld_p_addr[8:0]] <= ld_p_data[55:24];
            rq_m0_o   [ld_p_addr[8:0]] <= ld_p_data[64+15:64];
            rq_shift_o[ld_p_addr[8:0]] <= ld_p_data[64+20:64+16];
            rq_bias_o [ld_p_addr[8:0]] <= ld_p_data[64+55:64+24];
        end
    end

    wire [13:0] m_tiles  = Cin >> 4;            // Cin/16
    wire [9:0]  ob_blks  = (Cout + 15) >> 4;     // ceil(Cout/16)

    // ── shared engine drive ────────────────────────────────────────────────────
    reg        acc_clr_r, wt_load_r;
    reg [1:0]  shuf_mode_r;
    reg [P*`M*`DATA_W-1:0] ds_bus_r;
    reg [3:0]  idx0_r, idx1_r, idx2_r, idx3_r;
    // per-SPEC weight regs (4 oc each)
    reg signed [`DATA_W-1:0] wv0[0:NSPEC-1], wv1[0:NSPEC-1], wv2[0:NSPEC-1], wv3[0:NSPEC-1];
    reg signed [`DATA_W-1:0] wa0[0:NSPEC-1], wa1[0:NSPEC-1], wa2[0:NSPEC-1], wa3[0:NSPEC-1]; // entry1
    reg signed [`DATA_W-1:0] wb0[0:NSPEC-1], wb1[0:NSPEC-1], wb2[0:NSPEC-1], wb3[0:NSPEC-1]; // entry2

    wire [G*P*ACCW-1:0] ps_bus [0:NSPEC-1];
    wire [NSPEC-1:0]    mac_v;

    genvar s;
    generate for (s=0; s<NSPEC; s=s+1) begin : SPEC
        i_spec_mm u_eng (
            .clk(clk),.rst_n(rst_n),.n_cfg(n_cfg),.acc_clr(acc_clr_r),
            .shuf_mode(shuf_mode_r),.act_load_bus(ds_bus_r),.wt_load(wt_load_r),
            .idx_0(idx0_r),.idx_1(idx1_r),.idx_2(idx2_r),.idx_3(idx3_r),
            .idx_4(4'd0),.idx_5(4'd0),.idx_6(4'd0),.idx_7(4'd0),.idx_8(4'd0),.idx_9(4'd0),
            .idx_10(4'd0),.idx_11(4'd0),.idx_12(4'd0),.idx_13(4'd0),.idx_14(4'd0),.idx_15(4'd0),
            .wv_g0_0(wv0[s]),.wv_g0_1(wa0[s]),.wv_g0_2(wb0[s]),.wv_g0_3(8'd0),
            .wv_g1_0(wv1[s]),.wv_g1_1(wa1[s]),.wv_g1_2(wb1[s]),.wv_g1_3(8'd0),
            .wv_g2_0(wv2[s]),.wv_g2_1(wa2[s]),.wv_g2_2(wb2[s]),.wv_g2_3(8'd0),
            .wv_g3_0(wv3[s]),.wv_g3_1(wa3[s]),.wv_g3_2(wb3[s]),.wv_g3_3(8'd0),
            // flat psum ports unused (read packed bus instead)
            .psum_g0_p0(),.psum_g0_p1(),.psum_g0_p2(),.psum_g0_p3(),.psum_g0_p4(),.psum_g0_p5(),.psum_g0_p6(),.psum_g0_p7(),
            .psum_g0_p8(),.psum_g0_p9(),.psum_g0_p10(),.psum_g0_p11(),.psum_g0_p12(),.psum_g0_p13(),.psum_g0_p14(),.psum_g0_p15(),
            .psum_g1_p0(),.psum_g1_p1(),.psum_g1_p2(),.psum_g1_p3(),.psum_g1_p4(),.psum_g1_p5(),.psum_g1_p6(),.psum_g1_p7(),
            .psum_g1_p8(),.psum_g1_p9(),.psum_g1_p10(),.psum_g1_p11(),.psum_g1_p12(),.psum_g1_p13(),.psum_g1_p14(),.psum_g1_p15(),
            .psum_g2_p0(),.psum_g2_p1(),.psum_g2_p2(),.psum_g2_p3(),.psum_g2_p4(),.psum_g2_p5(),.psum_g2_p6(),.psum_g2_p7(),
            .psum_g2_p8(),.psum_g2_p9(),.psum_g2_p10(),.psum_g2_p11(),.psum_g2_p12(),.psum_g2_p13(),.psum_g2_p14(),.psum_g2_p15(),
            .psum_g3_p0(),.psum_g3_p1(),.psum_g3_p2(),.psum_g3_p3(),.psum_g3_p4(),.psum_g3_p5(),.psum_g3_p6(),.psum_g3_p7(),
            .psum_g3_p8(),.psum_g3_p9(),.psum_g3_p10(),.psum_g3_p11(),.psum_g3_p12(),.psum_g3_p13(),.psum_g3_p14(),.psum_g3_p15(),
            .psum_bus(ps_bus[s]),.mac_valid(mac_v[s])
        );
    end endgenerate

    // ── FSM state (declared before coo unpack) ──────────────────────────────────
    localparam S_IDLE=0,S_LDDS=1,S_WT=2,S_DRAIN=3,S_NEXTMT=4,S_RQ=5,S_RQWAIT=6,S_NEXTOB=7,S_DONE=8;
    reg [3:0]  st;
    reg [13:0] mt_cnt;
    reg [9:0]  cur_ob;     // oc-block-of-16 index
    reg [3:0]  drain_cnt;
    reg [7:0]  rq_cnt;     // 0..255
    reg [3:0]  rqdrain;

    // coo512 word for (cur_ob, mt)
    wire [13:0] coo_idx = cur_ob*m_tiles + mt_cnt;
    wire [NSPEC*128-1:0] cw = coo[coo_idx];

    // ── requant: P=16 PARALLEL rr_units (one per spatial pos) → 16x faster than serial.
    // Per cycle requant all 16 positions of one (spec,oc) → 16 (spec,oc)/ob16 = 16 cyc (vs 256).
    reg [1:0] rq_spec, rq_oc;
    reg       rq_vin;
    wire [9:0] oc_abs = (cur_ob<<4) + (rq_spec<<2) + rq_oc;     // ob16*16 + spec*4 + oc
    // banked param read: bank = oc_abs[0], idx = oc_abs>>1 (one read shared by all 16 rr_units)
    wire [8:0] oc_idx = oc_abs[9:1];
    wire       oc_bk  = oc_abs[0];
    wire [`SCALE_W-1:0]     cur_m0   = oc_bk ? rq_m0_o[oc_idx]    : rq_m0_e[oc_idx];
    wire [`SHIFT_W-1:0]     cur_sh   = oc_bk ? rq_shift_o[oc_idx] : rq_shift_e[oc_idx];
    wire signed [`ACC_W-1:0] cur_bias = oc_bk ? rq_bias_o[oc_idx] : rq_bias_e[oc_idx];
    wire [G*P*ACCW-1:0] sel_bus = ps_bus[rq_spec];
    wire [127:0] rq_word; wire rq_vout;
    genvar pp;
    generate for (pp=0; pp<P; pp=pp+1) begin: RRU
        wire signed [ACCW-1:0] ps_p = sel_bus[((rq_oc*P)+pp)*ACCW +: ACCW] + cur_bias;
        wire vo_p;
        rr_unit u_rr (
            .clk(clk),.rst_n(rst_n),
            .psum_in(ps_p),.rq_scale(cur_m0),.rq_shift(cur_sh),
            .apply_relu(apply_relu),.valid_in(rq_vin),
            .data_out(rq_word[pp*8 +: 8]),.valid_out(vo_p)
        );
    end endgenerate
    assign rq_vout = RRU[0].vo_p;

    // write oc-word index pipe matching rr_unit 5-cycle latency
    reg [MAW-1:0] oc_pipe [0:4];
    integer k;
    always @(posedge clk) begin
        oc_pipe[0] <= oc_abs;
        for (k=1;k<5;k=k+1) oc_pipe[k] <= oc_pipe[k-1];
    end
    always @(posedge clk) if (rq_vout) omem[oc_pipe[4]] <= rq_word;

    assign rd_o_data = omem[rd_o_addr];   // 128-bit word = 16 result bytes of oc=rd_o_addr

    // ── feed FSM ────────────────────────────────────────────────────────────────
    integer sp;
    always @(posedge clk) begin
        if (!rst_n) begin
            st<=S_IDLE; done<=0; acc_clr_r<=0; wt_load_r<=0; shuf_mode_r<=`SHUF_HOLD_MODE;
            mt_cnt<=0; cur_ob<=0; rq_vin<=0; rq_cnt<=0; drain_cnt<=0; rqdrain<=0;
        end else begin
            wt_load_r<=0; acc_clr_r<=0; rq_vin<=0; shuf_mode_r<=`SHUF_HOLD_MODE;
            case (st)
                S_IDLE: begin
                    done<=0;
                    if (start) begin cur_ob<=0; mt_cnt<=0; acc_clr_r<=1; st<=S_LDDS; end
                end
                S_LDDS: begin
                    ds_bus_r   <= amem[mt_cnt];
                    shuf_mode_r<= `SHUF_LOAD;
                    st<=S_WT;
                end
                S_WT: begin
                    // shared NZ indices from SPEC0 word
                    idx0_r<=cw[3:0]; idx1_r<=cw[7:4]; idx2_r<=cw[11:8]; idx3_r<=4'd0;
                    // per-SPEC weights (3 entries each)
                    for (sp=0; sp<NSPEC; sp=sp+1) begin
                        wv0[sp]<=cw[sp*128+19 -:8]; wa0[sp]<=cw[sp*128+27 -:8]; wb0[sp]<=cw[sp*128+35 -:8];
                        wv1[sp]<=cw[sp*128+43 -:8]; wa1[sp]<=cw[sp*128+51 -:8]; wb1[sp]<=cw[sp*128+59 -:8];
                        wv2[sp]<=cw[sp*128+67 -:8]; wa2[sp]<=cw[sp*128+75 -:8]; wb2[sp]<=cw[sp*128+83 -:8];
                        wv3[sp]<=cw[sp*128+91 -:8]; wa3[sp]<=cw[sp*128+99 -:8]; wb3[sp]<=cw[sp*128+107 -:8];
                    end
                    wt_load_r<=1; drain_cnt<=0; st<=S_DRAIN;
                end
                S_DRAIN: begin
                    drain_cnt<=drain_cnt+1;
                    if (drain_cnt >= n_cfg+3) st<=S_NEXTMT;
                end
                S_NEXTMT: begin
                    if (mt_cnt < m_tiles-1) begin mt_cnt<=mt_cnt+1; st<=S_LDDS; end
                    else begin rq_cnt<=0; rq_spec<=0; rq_oc<=0; st<=S_RQ; end
                end
                S_RQ: begin   // 16 (spec,oc) pairs, each requants 16 positions in parallel
                    rq_vin<=1;
                    rq_spec<=rq_cnt[3:2]; rq_oc<=rq_cnt[1:0];
                    if (rq_cnt==8'd15) begin rqdrain<=0; st<=S_RQWAIT; end
                    else rq_cnt<=rq_cnt+1;
                end
                S_RQWAIT: begin
                    rqdrain<=rqdrain+1;
                    if (rqdrain>=4'd7) st<=S_NEXTOB;
                end
                S_NEXTOB: begin
                    if (cur_ob < ob_blks-1) begin cur_ob<=cur_ob+1; mt_cnt<=0; acc_clr_r<=1; st<=S_LDDS; end
                    else st<=S_DONE;
                end
                S_DONE: begin done<=1; st<=S_IDLE; end
            endcase
        end
    end
endmodule
