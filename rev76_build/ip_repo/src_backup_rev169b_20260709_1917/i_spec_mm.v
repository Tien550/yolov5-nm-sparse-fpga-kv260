// i_spec_mm.v -- I-SPEC with FULL MM-tile on-chip COO (paper-faithful, n_cfg=N)
//
// Completes the paper's never-wired on-chip COO path:
//   - depth_shuffle_mm holds M×P=256 act (16 ic × 16 spatial)
//   - coo_decoder drains N NZ entries per M-tile (1 DMA load → N on-chip cycles)
//   - sel_addr (NZ index) selects act[ic] row → mul_group_array (spatial-parallel)
//   - psum_array accumulates over the N entries
//
// I-SPEC poc: all G=4 groups share the SAME sel_addr (same NZ positions); each
// group has its own weight value stream (different output channel). 4 coo_decoders
// share idx_in but get per-group val_in.
//
// One M-tile burst:  load=1 (latch idx+vals, n_cfg) → drain N cycles → done.
// Accumulate across M-tiles by NOT asserting acc_clr between them.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module i_spec_mm (
    input  wire        clk,
    input  wire        rst_n,

    input  wire [3:0]  n_cfg,          // NZ per M-tile (1..4 COO path)
    input  wire        acc_clr,        // clear psum (assert before first M-tile)

    // Activation MM-tile load (256 bytes = 16 ic × 16 spatial)
    input  wire [1:0]  shuf_mode,      // SHUF_LOAD to latch load_bus
    input  wire [`M*`P*`DATA_W-1:0] act_load_bus,

    // Weight stream: shared indices, per-group values; load pulse drains N.
    input  wire        wt_load,
    input  wire [3:0]  idx_0, input wire [3:0] idx_1, input wire [3:0] idx_2,  input wire [3:0] idx_3,
    input  wire [3:0]  idx_4, input wire [3:0] idx_5, input wire [3:0] idx_6,  input wire [3:0] idx_7,
    input  wire [3:0]  idx_8, input wire [3:0] idx_9, input wire [3:0] idx_10, input wire [3:0] idx_11,
    input  wire [3:0]  idx_12,input wire [3:0] idx_13,input wire [3:0] idx_14, input wire [3:0] idx_15,

    // Per-group weight values (g0..g3), first n_cfg used.  Flat 16 each.
    input  wire signed [`DATA_W-1:0] wv_g0_0, wv_g0_1, wv_g0_2, wv_g0_3,
    input  wire signed [`DATA_W-1:0] wv_g1_0, wv_g1_1, wv_g1_2, wv_g1_3,
    input  wire signed [`DATA_W-1:0] wv_g2_0, wv_g2_1, wv_g2_2, wv_g2_3,
    input  wire signed [`DATA_W-1:0] wv_g3_0, wv_g3_1, wv_g3_2, wv_g3_3,

    // PSUM outputs: G×P = 4×16 (flat). psum[g][p].
    output wire signed [`ACC_W-1:0]
        psum_g0_p0, psum_g0_p1, psum_g0_p2, psum_g0_p3, psum_g0_p4, psum_g0_p5, psum_g0_p6, psum_g0_p7,
        psum_g0_p8, psum_g0_p9, psum_g0_p10,psum_g0_p11,psum_g0_p12,psum_g0_p13,psum_g0_p14,psum_g0_p15,
        psum_g1_p0, psum_g1_p1, psum_g1_p2, psum_g1_p3, psum_g1_p4, psum_g1_p5, psum_g1_p6, psum_g1_p7,
        psum_g1_p8, psum_g1_p9, psum_g1_p10,psum_g1_p11,psum_g1_p12,psum_g1_p13,psum_g1_p14,psum_g1_p15,
        psum_g2_p0, psum_g2_p1, psum_g2_p2, psum_g2_p3, psum_g2_p4, psum_g2_p5, psum_g2_p6, psum_g2_p7,
        psum_g2_p8, psum_g2_p9, psum_g2_p10,psum_g2_p11,psum_g2_p12,psum_g2_p13,psum_g2_p14,psum_g2_p15,
        psum_g3_p0, psum_g3_p1, psum_g3_p2, psum_g3_p3, psum_g3_p4, psum_g3_p5, psum_g3_p6, psum_g3_p7,
        psum_g3_p8, psum_g3_p9, psum_g3_p10,psum_g3_p11,psum_g3_p12,psum_g3_p13,psum_g3_p14,psum_g3_p15,
    // Packed psum bus (B2): [(g*P+p)*ACC_W +: ACC_W] = psum_g{g}_p{p}.  G*P*ACC_W = 4*16*32 = 2048.
    output wire [`G*`P*`ACC_W-1:0] psum_bus,
    output wire        mac_valid
);

    // Tie unused COO val ports (entries 4..15) to 0 — COO path uses n_cfg<=4.
    wire signed [`DATA_W-1:0] z = {`DATA_W{1'b0}};

    // ── 4 COO decoders: shared idx, per-group values ─────────────────────────
    wire [3:0]               addr_g0, addr_g1, addr_g2, addr_g3;
    wire signed [`DATA_W-1:0] wt_g0, wt_g1, wt_g2, wt_g3;
    wire                     vld_g0, vld_g1, vld_g2, vld_g3;
    wire                     done_g0;

    coo_decoder u_coo0 (
        .clk(clk),.rst_n(rst_n),
        .idx_in_0(idx_0),.idx_in_1(idx_1),.idx_in_2(idx_2),.idx_in_3(idx_3),
        .idx_in_4(idx_4),.idx_in_5(idx_5),.idx_in_6(idx_6),.idx_in_7(idx_7),
        .idx_in_8(idx_8),.idx_in_9(idx_9),.idx_in_10(idx_10),.idx_in_11(idx_11),
        .idx_in_12(idx_12),.idx_in_13(idx_13),.idx_in_14(idx_14),.idx_in_15(idx_15),
        .val_in_0(wv_g0_0),.val_in_1(wv_g0_1),.val_in_2(wv_g0_2),.val_in_3(wv_g0_3),
        .val_in_4(z),.val_in_5(z),.val_in_6(z),.val_in_7(z),
        .val_in_8(z),.val_in_9(z),.val_in_10(z),.val_in_11(z),
        .val_in_12(z),.val_in_13(z),.val_in_14(z),.val_in_15(z),
        .n_cfg(n_cfg),.load(wt_load),
        .addr_out(addr_g0),.weight_out(wt_g0),.valid_out(vld_g0),.done(done_g0)
    );
    coo_decoder u_coo1 (
        .clk(clk),.rst_n(rst_n),
        .idx_in_0(idx_0),.idx_in_1(idx_1),.idx_in_2(idx_2),.idx_in_3(idx_3),
        .idx_in_4(idx_4),.idx_in_5(idx_5),.idx_in_6(idx_6),.idx_in_7(idx_7),
        .idx_in_8(idx_8),.idx_in_9(idx_9),.idx_in_10(idx_10),.idx_in_11(idx_11),
        .idx_in_12(idx_12),.idx_in_13(idx_13),.idx_in_14(idx_14),.idx_in_15(idx_15),
        .val_in_0(wv_g1_0),.val_in_1(wv_g1_1),.val_in_2(wv_g1_2),.val_in_3(wv_g1_3),
        .val_in_4(z),.val_in_5(z),.val_in_6(z),.val_in_7(z),
        .val_in_8(z),.val_in_9(z),.val_in_10(z),.val_in_11(z),
        .val_in_12(z),.val_in_13(z),.val_in_14(z),.val_in_15(z),
        .n_cfg(n_cfg),.load(wt_load),
        .addr_out(addr_g1),.weight_out(wt_g1),.valid_out(vld_g1),.done()
    );
    coo_decoder u_coo2 (
        .clk(clk),.rst_n(rst_n),
        .idx_in_0(idx_0),.idx_in_1(idx_1),.idx_in_2(idx_2),.idx_in_3(idx_3),
        .idx_in_4(idx_4),.idx_in_5(idx_5),.idx_in_6(idx_6),.idx_in_7(idx_7),
        .idx_in_8(idx_8),.idx_in_9(idx_9),.idx_in_10(idx_10),.idx_in_11(idx_11),
        .idx_in_12(idx_12),.idx_in_13(idx_13),.idx_in_14(idx_14),.idx_in_15(idx_15),
        .val_in_0(wv_g2_0),.val_in_1(wv_g2_1),.val_in_2(wv_g2_2),.val_in_3(wv_g2_3),
        .val_in_4(z),.val_in_5(z),.val_in_6(z),.val_in_7(z),
        .val_in_8(z),.val_in_9(z),.val_in_10(z),.val_in_11(z),
        .val_in_12(z),.val_in_13(z),.val_in_14(z),.val_in_15(z),
        .n_cfg(n_cfg),.load(wt_load),
        .addr_out(addr_g2),.weight_out(wt_g2),.valid_out(vld_g2),.done()
    );
    coo_decoder u_coo3 (
        .clk(clk),.rst_n(rst_n),
        .idx_in_0(idx_0),.idx_in_1(idx_1),.idx_in_2(idx_2),.idx_in_3(idx_3),
        .idx_in_4(idx_4),.idx_in_5(idx_5),.idx_in_6(idx_6),.idx_in_7(idx_7),
        .idx_in_8(idx_8),.idx_in_9(idx_9),.idx_in_10(idx_10),.idx_in_11(idx_11),
        .idx_in_12(idx_12),.idx_in_13(idx_13),.idx_in_14(idx_14),.idx_in_15(idx_15),
        .val_in_0(wv_g3_0),.val_in_1(wv_g3_1),.val_in_2(wv_g3_2),.val_in_3(wv_g3_3),
        .val_in_4(z),.val_in_5(z),.val_in_6(z),.val_in_7(z),
        .val_in_8(z),.val_in_9(z),.val_in_10(z),.val_in_11(z),
        .val_in_12(z),.val_in_13(z),.val_in_14(z),.val_in_15(z),
        .n_cfg(n_cfg),.load(wt_load),
        .addr_out(addr_g3),.weight_out(wt_g3),.valid_out(vld_g3),.done()
    );

    // ── MM-tile depth shuffle: sel by g0 addr (all groups share NZ index) ────
    wire signed [`DATA_W-1:0]
        a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15;

    depth_shuffle_mm u_ds (
        .clk(clk),.rst_n(rst_n),.shuffle_mode(shuf_mode),
        .sel_addr(addr_g0),
        .load_bus(act_load_bus),
        .act_out_0(a0),  .act_out_1(a1),  .act_out_2(a2),  .act_out_3(a3),
        .act_out_4(a4),  .act_out_5(a5),  .act_out_6(a6),  .act_out_7(a7),
        .act_out_8(a8),  .act_out_9(a9),  .act_out_10(a10),.act_out_11(a11),
        .act_out_12(a12),.act_out_13(a13),.act_out_14(a14),.act_out_15(a15)
    );

    // ── mul_group_array: same act to all 4 groups, per-group weight ──────────
    wire signed [`ACC_W-1:0]
        mg0p0,mg0p1,mg0p2,mg0p3,mg0p4,mg0p5,mg0p6,mg0p7,mg0p8,mg0p9,mg0p10,mg0p11,mg0p12,mg0p13,mg0p14,mg0p15,
        mg1p0,mg1p1,mg1p2,mg1p3,mg1p4,mg1p5,mg1p6,mg1p7,mg1p8,mg1p9,mg1p10,mg1p11,mg1p12,mg1p13,mg1p14,mg1p15,
        mg2p0,mg2p1,mg2p2,mg2p3,mg2p4,mg2p5,mg2p6,mg2p7,mg2p8,mg2p9,mg2p10,mg2p11,mg2p12,mg2p13,mg2p14,mg2p15,
        mg3p0,mg3p1,mg3p2,mg3p3,mg3p4,mg3p5,mg3p6,mg3p7,mg3p8,mg3p9,mg3p10,mg3p11,mg3p12,mg3p13,mg3p14,mg3p15;

    mul_group_array u_mga (
        .clk(clk),.rst_n(rst_n),.valid_in(vld_g0),
        .wt_g0(wt_g0),.wt_g1(wt_g1),.wt_g2(wt_g2),.wt_g3(wt_g3),
        .act_g0_p0(a0),.act_g0_p1(a1),.act_g0_p2(a2),.act_g0_p3(a3),.act_g0_p4(a4),.act_g0_p5(a5),.act_g0_p6(a6),.act_g0_p7(a7),
        .act_g0_p8(a8),.act_g0_p9(a9),.act_g0_p10(a10),.act_g0_p11(a11),.act_g0_p12(a12),.act_g0_p13(a13),.act_g0_p14(a14),.act_g0_p15(a15),
        .act_g1_p0(a0),.act_g1_p1(a1),.act_g1_p2(a2),.act_g1_p3(a3),.act_g1_p4(a4),.act_g1_p5(a5),.act_g1_p6(a6),.act_g1_p7(a7),
        .act_g1_p8(a8),.act_g1_p9(a9),.act_g1_p10(a10),.act_g1_p11(a11),.act_g1_p12(a12),.act_g1_p13(a13),.act_g1_p14(a14),.act_g1_p15(a15),
        .act_g2_p0(a0),.act_g2_p1(a1),.act_g2_p2(a2),.act_g2_p3(a3),.act_g2_p4(a4),.act_g2_p5(a5),.act_g2_p6(a6),.act_g2_p7(a7),
        .act_g2_p8(a8),.act_g2_p9(a9),.act_g2_p10(a10),.act_g2_p11(a11),.act_g2_p12(a12),.act_g2_p13(a13),.act_g2_p14(a14),.act_g2_p15(a15),
        .act_g3_p0(a0),.act_g3_p1(a1),.act_g3_p2(a2),.act_g3_p3(a3),.act_g3_p4(a4),.act_g3_p5(a5),.act_g3_p6(a6),.act_g3_p7(a7),
        .act_g3_p8(a8),.act_g3_p9(a9),.act_g3_p10(a10),.act_g3_p11(a11),.act_g3_p12(a12),.act_g3_p13(a13),.act_g3_p14(a14),.act_g3_p15(a15),
        .prod_g0_p0(mg0p0),.prod_g0_p1(mg0p1),.prod_g0_p2(mg0p2),.prod_g0_p3(mg0p3),.prod_g0_p4(mg0p4),.prod_g0_p5(mg0p5),.prod_g0_p6(mg0p6),.prod_g0_p7(mg0p7),
        .prod_g0_p8(mg0p8),.prod_g0_p9(mg0p9),.prod_g0_p10(mg0p10),.prod_g0_p11(mg0p11),.prod_g0_p12(mg0p12),.prod_g0_p13(mg0p13),.prod_g0_p14(mg0p14),.prod_g0_p15(mg0p15),
        .prod_g1_p0(mg1p0),.prod_g1_p1(mg1p1),.prod_g1_p2(mg1p2),.prod_g1_p3(mg1p3),.prod_g1_p4(mg1p4),.prod_g1_p5(mg1p5),.prod_g1_p6(mg1p6),.prod_g1_p7(mg1p7),
        .prod_g1_p8(mg1p8),.prod_g1_p9(mg1p9),.prod_g1_p10(mg1p10),.prod_g1_p11(mg1p11),.prod_g1_p12(mg1p12),.prod_g1_p13(mg1p13),.prod_g1_p14(mg1p14),.prod_g1_p15(mg1p15),
        .prod_g2_p0(mg2p0),.prod_g2_p1(mg2p1),.prod_g2_p2(mg2p2),.prod_g2_p3(mg2p3),.prod_g2_p4(mg2p4),.prod_g2_p5(mg2p5),.prod_g2_p6(mg2p6),.prod_g2_p7(mg2p7),
        .prod_g2_p8(mg2p8),.prod_g2_p9(mg2p9),.prod_g2_p10(mg2p10),.prod_g2_p11(mg2p11),.prod_g2_p12(mg2p12),.prod_g2_p13(mg2p13),.prod_g2_p14(mg2p14),.prod_g2_p15(mg2p15),
        .prod_g3_p0(mg3p0),.prod_g3_p1(mg3p1),.prod_g3_p2(mg3p2),.prod_g3_p3(mg3p3),.prod_g3_p4(mg3p4),.prod_g3_p5(mg3p5),.prod_g3_p6(mg3p6),.prod_g3_p7(mg3p7),
        .prod_g3_p8(mg3p8),.prod_g3_p9(mg3p9),.prod_g3_p10(mg3p10),.prod_g3_p11(mg3p11),.prod_g3_p12(mg3p12),.prod_g3_p13(mg3p13),.prod_g3_p14(mg3p14),.prod_g3_p15(mg3p15),
        .valid_out(mac_valid)
    );

    // ── psum_array: accumulate over the N entries (and across M-tiles) ───────
    psum_array u_psa (
        .clk(clk),.rst_n(rst_n),.en(mac_valid),.acc_clr(acc_clr),
        .prod_g0_p0(mg0p0),.prod_g0_p1(mg0p1),.prod_g0_p2(mg0p2),.prod_g0_p3(mg0p3),.prod_g0_p4(mg0p4),.prod_g0_p5(mg0p5),.prod_g0_p6(mg0p6),.prod_g0_p7(mg0p7),
        .prod_g0_p8(mg0p8),.prod_g0_p9(mg0p9),.prod_g0_p10(mg0p10),.prod_g0_p11(mg0p11),.prod_g0_p12(mg0p12),.prod_g0_p13(mg0p13),.prod_g0_p14(mg0p14),.prod_g0_p15(mg0p15),
        .prod_g1_p0(mg1p0),.prod_g1_p1(mg1p1),.prod_g1_p2(mg1p2),.prod_g1_p3(mg1p3),.prod_g1_p4(mg1p4),.prod_g1_p5(mg1p5),.prod_g1_p6(mg1p6),.prod_g1_p7(mg1p7),
        .prod_g1_p8(mg1p8),.prod_g1_p9(mg1p9),.prod_g1_p10(mg1p10),.prod_g1_p11(mg1p11),.prod_g1_p12(mg1p12),.prod_g1_p13(mg1p13),.prod_g1_p14(mg1p14),.prod_g1_p15(mg1p15),
        .prod_g2_p0(mg2p0),.prod_g2_p1(mg2p1),.prod_g2_p2(mg2p2),.prod_g2_p3(mg2p3),.prod_g2_p4(mg2p4),.prod_g2_p5(mg2p5),.prod_g2_p6(mg2p6),.prod_g2_p7(mg2p7),
        .prod_g2_p8(mg2p8),.prod_g2_p9(mg2p9),.prod_g2_p10(mg2p10),.prod_g2_p11(mg2p11),.prod_g2_p12(mg2p12),.prod_g2_p13(mg2p13),.prod_g2_p14(mg2p14),.prod_g2_p15(mg2p15),
        .prod_g3_p0(mg3p0),.prod_g3_p1(mg3p1),.prod_g3_p2(mg3p2),.prod_g3_p3(mg3p3),.prod_g3_p4(mg3p4),.prod_g3_p5(mg3p5),.prod_g3_p6(mg3p6),.prod_g3_p7(mg3p7),
        .prod_g3_p8(mg3p8),.prod_g3_p9(mg3p9),.prod_g3_p10(mg3p10),.prod_g3_p11(mg3p11),.prod_g3_p12(mg3p12),.prod_g3_p13(mg3p13),.prod_g3_p14(mg3p14),.prod_g3_p15(mg3p15),
        .psum_g0_p0(psum_g0_p0),.psum_g0_p1(psum_g0_p1),.psum_g0_p2(psum_g0_p2),.psum_g0_p3(psum_g0_p3),.psum_g0_p4(psum_g0_p4),.psum_g0_p5(psum_g0_p5),.psum_g0_p6(psum_g0_p6),.psum_g0_p7(psum_g0_p7),
        .psum_g0_p8(psum_g0_p8),.psum_g0_p9(psum_g0_p9),.psum_g0_p10(psum_g0_p10),.psum_g0_p11(psum_g0_p11),.psum_g0_p12(psum_g0_p12),.psum_g0_p13(psum_g0_p13),.psum_g0_p14(psum_g0_p14),.psum_g0_p15(psum_g0_p15),
        .psum_g1_p0(psum_g1_p0),.psum_g1_p1(psum_g1_p1),.psum_g1_p2(psum_g1_p2),.psum_g1_p3(psum_g1_p3),.psum_g1_p4(psum_g1_p4),.psum_g1_p5(psum_g1_p5),.psum_g1_p6(psum_g1_p6),.psum_g1_p7(psum_g1_p7),
        .psum_g1_p8(psum_g1_p8),.psum_g1_p9(psum_g1_p9),.psum_g1_p10(psum_g1_p10),.psum_g1_p11(psum_g1_p11),.psum_g1_p12(psum_g1_p12),.psum_g1_p13(psum_g1_p13),.psum_g1_p14(psum_g1_p14),.psum_g1_p15(psum_g1_p15),
        .psum_g2_p0(psum_g2_p0),.psum_g2_p1(psum_g2_p1),.psum_g2_p2(psum_g2_p2),.psum_g2_p3(psum_g2_p3),.psum_g2_p4(psum_g2_p4),.psum_g2_p5(psum_g2_p5),.psum_g2_p6(psum_g2_p6),.psum_g2_p7(psum_g2_p7),
        .psum_g2_p8(psum_g2_p8),.psum_g2_p9(psum_g2_p9),.psum_g2_p10(psum_g2_p10),.psum_g2_p11(psum_g2_p11),.psum_g2_p12(psum_g2_p12),.psum_g2_p13(psum_g2_p13),.psum_g2_p14(psum_g2_p14),.psum_g2_p15(psum_g2_p15),
        .psum_g3_p0(psum_g3_p0),.psum_g3_p1(psum_g3_p1),.psum_g3_p2(psum_g3_p2),.psum_g3_p3(psum_g3_p3),.psum_g3_p4(psum_g3_p4),.psum_g3_p5(psum_g3_p5),.psum_g3_p6(psum_g3_p6),.psum_g3_p7(psum_g3_p7),
        .psum_g3_p8(psum_g3_p8),.psum_g3_p9(psum_g3_p9),.psum_g3_p10(psum_g3_p10),.psum_g3_p11(psum_g3_p11),.psum_g3_p12(psum_g3_p12),.psum_g3_p13(psum_g3_p13),.psum_g3_p14(psum_g3_p14),.psum_g3_p15(psum_g3_p15)
    );

    // ── packed psum bus (B2 multi-SPEC collection) ───────────────────────────
    assign psum_bus = {
        psum_g3_p15,psum_g3_p14,psum_g3_p13,psum_g3_p12,psum_g3_p11,psum_g3_p10,psum_g3_p9,psum_g3_p8,
        psum_g3_p7, psum_g3_p6, psum_g3_p5, psum_g3_p4, psum_g3_p3, psum_g3_p2, psum_g3_p1,psum_g3_p0,
        psum_g2_p15,psum_g2_p14,psum_g2_p13,psum_g2_p12,psum_g2_p11,psum_g2_p10,psum_g2_p9,psum_g2_p8,
        psum_g2_p7, psum_g2_p6, psum_g2_p5, psum_g2_p4, psum_g2_p3, psum_g2_p2, psum_g2_p1,psum_g2_p0,
        psum_g1_p15,psum_g1_p14,psum_g1_p13,psum_g1_p12,psum_g1_p11,psum_g1_p10,psum_g1_p9,psum_g1_p8,
        psum_g1_p7, psum_g1_p6, psum_g1_p5, psum_g1_p4, psum_g1_p3, psum_g1_p2, psum_g1_p1,psum_g1_p0,
        psum_g0_p15,psum_g0_p14,psum_g0_p13,psum_g0_p12,psum_g0_p11,psum_g0_p10,psum_g0_p9,psum_g0_p8,
        psum_g0_p7, psum_g0_p6, psum_g0_p5, psum_g0_p4, psum_g0_p3, psum_g0_p2, psum_g0_p1,psum_g0_p0
    };

endmodule
