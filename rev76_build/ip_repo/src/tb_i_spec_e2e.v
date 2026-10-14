// tb_i_spec_e2e.v  --  End-to-end logic verification for i_spec
// Tests pch_gen + depth_shuffle + weight_decoder + mac + accumulator chain
// with exact expected PSUM values calculated analytically.
//
// Architecture (broadcast per group):
//   pch_gen: addr_g = poc_cnt*poc + g
//   act_sel_g = depth_shuffle[g][addr_g]   <- one value
//   psum[g][p] += wt_g * act_sel_g          <- same for all P positions
//
// Activation window: act[g][p] = g*16 + p
//   act[g0][0..15] = 0,1,...,15
//   act[g1][0..15] = 16,17,...,31
//   act[g2][0..15] = 32,33,...,47
//   act[g3][0..15] = 48,49,...,63
//
// TC_01: poc=0, N=2, idx=[3,5], wt_g0=[10,20], wt_g1..g3=0
//   poc=0: addr_g0 = idx = 3 then 5
//   act_sel_g0 at idx=3: act[g0][3]=3
//   act_sel_g0 at idx=5: act[g0][5]=5
//   psum_g0[all p] = 10*3 + 20*5 = 30+100 = 130
//   psum_g1..g3 = 0
//
// TC_02: poc=1 (poc_val=2), N=2, idx=[3,5]
//   wt_g0=[10,20], wt_g1=[-5,15], wt_g2=[1,-3], wt_g3=[7,2]
//   pch_gen with poc=1:
//     NZ idx=3: addr_g = 3*2+g = [6,7,8,9]
//       act_sel_g0=act[g0][6]=6,  act_sel_g1=act[g1][7]=23
//       act_sel_g2=act[g2][8]=40, act_sel_g3=act[g3][9]=57
//     NZ idx=5: addr_g = 5*2+g = [10,11,12,13]
//       act_sel_g0=act[g0][10]=10, act_sel_g1=act[g1][11]=27
//       act_sel_g2=act[g2][12]=44, act_sel_g3=act[g3][13]=61
//   psum_g0[all p] = 10*6  + 20*10  = 60+200   = 260
//   psum_g1[all p] = (-5)*23 + 15*27 = -115+405 = 290
//   psum_g2[all p] = 1*40 + (-3)*44  = 40-132   = -92
//   psum_g3[all p] = 7*57 + 2*61     = 399+122  = 521
//
// TC_03: acc_clr then reload TC_02 -> same values

`timescale 1ns/1ps
`include "cnn_defines.vh"

module tb_i_spec_e2e;

    localparam SHUF_LOAD = 2'b00;
    localparam SHUF_HOLD = 2'b10;

    reg clk, rst_n;
    reg [3:0]  n_cfg;
    reg [1:0]  poc;
    reg        apply_relu, wt_load, acc_clr;
    reg [1:0]  shuf_mode;

    reg [3:0] wi0,wi1,wi2,wi3,wi4,wi5,wi6,wi7,wi8,wi9,wi10,wi11,wi12,wi13,wi14,wi15;

    reg signed [7:0] wv_g0_0,wv_g0_1,wv_g0_2,wv_g0_3,wv_g0_4,wv_g0_5,wv_g0_6,wv_g0_7;
    reg signed [7:0] wv_g0_8,wv_g0_9,wv_g0_10,wv_g0_11,wv_g0_12,wv_g0_13,wv_g0_14,wv_g0_15;
    reg signed [7:0] wv_g1_0,wv_g1_1,wv_g1_2,wv_g1_3,wv_g1_4,wv_g1_5,wv_g1_6,wv_g1_7;
    reg signed [7:0] wv_g1_8,wv_g1_9,wv_g1_10,wv_g1_11,wv_g1_12,wv_g1_13,wv_g1_14,wv_g1_15;
    reg signed [7:0] wv_g2_0,wv_g2_1,wv_g2_2,wv_g2_3,wv_g2_4,wv_g2_5,wv_g2_6,wv_g2_7;
    reg signed [7:0] wv_g2_8,wv_g2_9,wv_g2_10,wv_g2_11,wv_g2_12,wv_g2_13,wv_g2_14,wv_g2_15;
    reg signed [7:0] wv_g3_0,wv_g3_1,wv_g3_2,wv_g3_3,wv_g3_4,wv_g3_5,wv_g3_6,wv_g3_7;
    reg signed [7:0] wv_g3_8,wv_g3_9,wv_g3_10,wv_g3_11,wv_g3_12,wv_g3_13,wv_g3_14,wv_g3_15;

    reg signed [7:0] ag0p0,ag0p1,ag0p2,ag0p3,ag0p4,ag0p5,ag0p6,ag0p7;
    reg signed [7:0] ag0p8,ag0p9,ag0p10,ag0p11,ag0p12,ag0p13,ag0p14,ag0p15;
    reg signed [7:0] ag1p0,ag1p1,ag1p2,ag1p3,ag1p4,ag1p5,ag1p6,ag1p7;
    reg signed [7:0] ag1p8,ag1p9,ag1p10,ag1p11,ag1p12,ag1p13,ag1p14,ag1p15;
    reg signed [7:0] ag2p0,ag2p1,ag2p2,ag2p3,ag2p4,ag2p5,ag2p6,ag2p7;
    reg signed [7:0] ag2p8,ag2p9,ag2p10,ag2p11,ag2p12,ag2p13,ag2p14,ag2p15;
    reg signed [7:0] ag3p0,ag3p1,ag3p2,ag3p3,ag3p4,ag3p5,ag3p6,ag3p7;
    reg signed [7:0] ag3p8,ag3p9,ag3p10,ag3p11,ag3p12,ag3p13,ag3p14,ag3p15;
    reg signed [7:0] nc0,nc1,nc2,nc3;

    // Monitor p0 and p15 to confirm broadcast
    wire signed [31:0] ps0p0,ps0p15;
    wire signed [31:0] ps1p0,ps1p15;
    wire signed [31:0] ps2p0,ps2p15;
    wire signed [31:0] ps3p0,ps3p15;
    wire mac_valid;

    // Unused outputs
    wire signed [31:0]
        _s0p1,_s0p2,_s0p3,_s0p4,_s0p5,_s0p6,_s0p7,
        _s0p8,_s0p9,_s0p10,_s0p11,_s0p12,_s0p13,_s0p14,
        _s1p1,_s1p2,_s1p3,_s1p4,_s1p5,_s1p6,_s1p7,
        _s1p8,_s1p9,_s1p10,_s1p11,_s1p12,_s1p13,_s1p14,
        _s2p1,_s2p2,_s2p3,_s2p4,_s2p5,_s2p6,_s2p7,
        _s2p8,_s2p9,_s2p10,_s2p11,_s2p12,_s2p13,_s2p14,
        _s3p1,_s3p2,_s3p3,_s3p4,_s3p5,_s3p6,_s3p7,
        _s3p8,_s3p9,_s3p10,_s3p11,_s3p12,_s3p13,_s3p14;

    reg [15:0] bm_word;
    reg signed [7:0] bv_g0_0,bv_g0_1,bv_g0_2,bv_g0_3,bv_g0_4,bv_g0_5,bv_g0_6,bv_g0_7;
    reg signed [7:0] bv_g0_8,bv_g0_9,bv_g0_10,bv_g0_11,bv_g0_12,bv_g0_13,bv_g0_14,bv_g0_15;
    reg signed [7:0] bv_g1_0,bv_g1_1,bv_g1_2,bv_g1_3,bv_g1_4,bv_g1_5,bv_g1_6,bv_g1_7;
    reg signed [7:0] bv_g1_8,bv_g1_9,bv_g1_10,bv_g1_11,bv_g1_12,bv_g1_13,bv_g1_14,bv_g1_15;
    reg signed [7:0] bv_g2_0,bv_g2_1,bv_g2_2,bv_g2_3,bv_g2_4,bv_g2_5,bv_g2_6,bv_g2_7;
    reg signed [7:0] bv_g2_8,bv_g2_9,bv_g2_10,bv_g2_11,bv_g2_12,bv_g2_13,bv_g2_14,bv_g2_15;
    reg signed [7:0] bv_g3_0,bv_g3_1,bv_g3_2,bv_g3_3,bv_g3_4,bv_g3_5,bv_g3_6,bv_g3_7;
    reg signed [7:0] bv_g3_8,bv_g3_9,bv_g3_10,bv_g3_11,bv_g3_12,bv_g3_13,bv_g3_14,bv_g3_15;

    i_spec dut (
        .clk(clk),.rst_n(rst_n),
        .n_cfg(n_cfg),.poc(poc),.apply_relu(apply_relu),
        .wt_idx_0(wi0),.wt_idx_1(wi1),.wt_idx_2(wi2),.wt_idx_3(wi3),
        .wt_idx_4(wi4),.wt_idx_5(wi5),.wt_idx_6(wi6),.wt_idx_7(wi7),
        .wt_idx_8(wi8),.wt_idx_9(wi9),.wt_idx_10(wi10),.wt_idx_11(wi11),
        .wt_idx_12(wi12),.wt_idx_13(wi13),.wt_idx_14(wi14),.wt_idx_15(wi15),
        .wt_val_g0_0(wv_g0_0),.wt_val_g0_1(wv_g0_1),.wt_val_g0_2(wv_g0_2),.wt_val_g0_3(wv_g0_3),
        .wt_val_g0_4(wv_g0_4),.wt_val_g0_5(wv_g0_5),.wt_val_g0_6(wv_g0_6),.wt_val_g0_7(wv_g0_7),
        .wt_val_g0_8(wv_g0_8),.wt_val_g0_9(wv_g0_9),.wt_val_g0_10(wv_g0_10),.wt_val_g0_11(wv_g0_11),
        .wt_val_g0_12(wv_g0_12),.wt_val_g0_13(wv_g0_13),.wt_val_g0_14(wv_g0_14),.wt_val_g0_15(wv_g0_15),
        .wt_val_g1_0(wv_g1_0),.wt_val_g1_1(wv_g1_1),.wt_val_g1_2(wv_g1_2),.wt_val_g1_3(wv_g1_3),
        .wt_val_g1_4(wv_g1_4),.wt_val_g1_5(wv_g1_5),.wt_val_g1_6(wv_g1_6),.wt_val_g1_7(wv_g1_7),
        .wt_val_g1_8(wv_g1_8),.wt_val_g1_9(wv_g1_9),.wt_val_g1_10(wv_g1_10),.wt_val_g1_11(wv_g1_11),
        .wt_val_g1_12(wv_g1_12),.wt_val_g1_13(wv_g1_13),.wt_val_g1_14(wv_g1_14),.wt_val_g1_15(wv_g1_15),
        .wt_val_g2_0(wv_g2_0),.wt_val_g2_1(wv_g2_1),.wt_val_g2_2(wv_g2_2),.wt_val_g2_3(wv_g2_3),
        .wt_val_g2_4(wv_g2_4),.wt_val_g2_5(wv_g2_5),.wt_val_g2_6(wv_g2_6),.wt_val_g2_7(wv_g2_7),
        .wt_val_g2_8(wv_g2_8),.wt_val_g2_9(wv_g2_9),.wt_val_g2_10(wv_g2_10),.wt_val_g2_11(wv_g2_11),
        .wt_val_g2_12(wv_g2_12),.wt_val_g2_13(wv_g2_13),.wt_val_g2_14(wv_g2_14),.wt_val_g2_15(wv_g2_15),
        .wt_val_g3_0(wv_g3_0),.wt_val_g3_1(wv_g3_1),.wt_val_g3_2(wv_g3_2),.wt_val_g3_3(wv_g3_3),
        .wt_val_g3_4(wv_g3_4),.wt_val_g3_5(wv_g3_5),.wt_val_g3_6(wv_g3_6),.wt_val_g3_7(wv_g3_7),
        .wt_val_g3_8(wv_g3_8),.wt_val_g3_9(wv_g3_9),.wt_val_g3_10(wv_g3_10),.wt_val_g3_11(wv_g3_11),
        .wt_val_g3_12(wv_g3_12),.wt_val_g3_13(wv_g3_13),.wt_val_g3_14(wv_g3_14),.wt_val_g3_15(wv_g3_15),
        .bm_word(bm_word),
        .bm_val_g0_0(bv_g0_0),.bm_val_g0_1(bv_g0_1),.bm_val_g0_2(bv_g0_2),.bm_val_g0_3(bv_g0_3),
        .bm_val_g0_4(bv_g0_4),.bm_val_g0_5(bv_g0_5),.bm_val_g0_6(bv_g0_6),.bm_val_g0_7(bv_g0_7),
        .bm_val_g0_8(bv_g0_8),.bm_val_g0_9(bv_g0_9),.bm_val_g0_10(bv_g0_10),.bm_val_g0_11(bv_g0_11),
        .bm_val_g0_12(bv_g0_12),.bm_val_g0_13(bv_g0_13),.bm_val_g0_14(bv_g0_14),.bm_val_g0_15(bv_g0_15),
        .bm_val_g1_0(bv_g1_0),.bm_val_g1_1(bv_g1_1),.bm_val_g1_2(bv_g1_2),.bm_val_g1_3(bv_g1_3),
        .bm_val_g1_4(bv_g1_4),.bm_val_g1_5(bv_g1_5),.bm_val_g1_6(bv_g1_6),.bm_val_g1_7(bv_g1_7),
        .bm_val_g1_8(bv_g1_8),.bm_val_g1_9(bv_g1_9),.bm_val_g1_10(bv_g1_10),.bm_val_g1_11(bv_g1_11),
        .bm_val_g1_12(bv_g1_12),.bm_val_g1_13(bv_g1_13),.bm_val_g1_14(bv_g1_14),.bm_val_g1_15(bv_g1_15),
        .bm_val_g2_0(bv_g2_0),.bm_val_g2_1(bv_g2_1),.bm_val_g2_2(bv_g2_2),.bm_val_g2_3(bv_g2_3),
        .bm_val_g2_4(bv_g2_4),.bm_val_g2_5(bv_g2_5),.bm_val_g2_6(bv_g2_6),.bm_val_g2_7(bv_g2_7),
        .bm_val_g2_8(bv_g2_8),.bm_val_g2_9(bv_g2_9),.bm_val_g2_10(bv_g2_10),.bm_val_g2_11(bv_g2_11),
        .bm_val_g2_12(bv_g2_12),.bm_val_g2_13(bv_g2_13),.bm_val_g2_14(bv_g2_14),.bm_val_g2_15(bv_g2_15),
        .bm_val_g3_0(bv_g3_0),.bm_val_g3_1(bv_g3_1),.bm_val_g3_2(bv_g3_2),.bm_val_g3_3(bv_g3_3),
        .bm_val_g3_4(bv_g3_4),.bm_val_g3_5(bv_g3_5),.bm_val_g3_6(bv_g3_6),.bm_val_g3_7(bv_g3_7),
        .bm_val_g3_8(bv_g3_8),.bm_val_g3_9(bv_g3_9),.bm_val_g3_10(bv_g3_10),.bm_val_g3_11(bv_g3_11),
        .bm_val_g3_12(bv_g3_12),.bm_val_g3_13(bv_g3_13),.bm_val_g3_14(bv_g3_14),.bm_val_g3_15(bv_g3_15),
        .wt_load(wt_load),.acc_clr(acc_clr),
        .shuf_mode(shuf_mode),
        .new_col_0(nc0),.new_col_1(nc1),.new_col_2(nc2),.new_col_3(nc3),
        .act_l_g0_p0(ag0p0),.act_l_g0_p1(ag0p1),.act_l_g0_p2(ag0p2),.act_l_g0_p3(ag0p3),
        .act_l_g0_p4(ag0p4),.act_l_g0_p5(ag0p5),.act_l_g0_p6(ag0p6),.act_l_g0_p7(ag0p7),
        .act_l_g0_p8(ag0p8),.act_l_g0_p9(ag0p9),.act_l_g0_p10(ag0p10),.act_l_g0_p11(ag0p11),
        .act_l_g0_p12(ag0p12),.act_l_g0_p13(ag0p13),.act_l_g0_p14(ag0p14),.act_l_g0_p15(ag0p15),
        .act_l_g1_p0(ag1p0),.act_l_g1_p1(ag1p1),.act_l_g1_p2(ag1p2),.act_l_g1_p3(ag1p3),
        .act_l_g1_p4(ag1p4),.act_l_g1_p5(ag1p5),.act_l_g1_p6(ag1p6),.act_l_g1_p7(ag1p7),
        .act_l_g1_p8(ag1p8),.act_l_g1_p9(ag1p9),.act_l_g1_p10(ag1p10),.act_l_g1_p11(ag1p11),
        .act_l_g1_p12(ag1p12),.act_l_g1_p13(ag1p13),.act_l_g1_p14(ag1p14),.act_l_g1_p15(ag1p15),
        .act_l_g2_p0(ag2p0),.act_l_g2_p1(ag2p1),.act_l_g2_p2(ag2p2),.act_l_g2_p3(ag2p3),
        .act_l_g2_p4(ag2p4),.act_l_g2_p5(ag2p5),.act_l_g2_p6(ag2p6),.act_l_g2_p7(ag2p7),
        .act_l_g2_p8(ag2p8),.act_l_g2_p9(ag2p9),.act_l_g2_p10(ag2p10),.act_l_g2_p11(ag2p11),
        .act_l_g2_p12(ag2p12),.act_l_g2_p13(ag2p13),.act_l_g2_p14(ag2p14),.act_l_g2_p15(ag2p15),
        .act_l_g3_p0(ag3p0),.act_l_g3_p1(ag3p1),.act_l_g3_p2(ag3p2),.act_l_g3_p3(ag3p3),
        .act_l_g3_p4(ag3p4),.act_l_g3_p5(ag3p5),.act_l_g3_p6(ag3p6),.act_l_g3_p7(ag3p7),
        .act_l_g3_p8(ag3p8),.act_l_g3_p9(ag3p9),.act_l_g3_p10(ag3p10),.act_l_g3_p11(ag3p11),
        .act_l_g3_p12(ag3p12),.act_l_g3_p13(ag3p13),.act_l_g3_p14(ag3p14),.act_l_g3_p15(ag3p15),
        .psum_g0_p0(ps0p0), .psum_g0_p1(_s0p1), .psum_g0_p2(_s0p2), .psum_g0_p3(_s0p3),
        .psum_g0_p4(_s0p4), .psum_g0_p5(_s0p5), .psum_g0_p6(_s0p6), .psum_g0_p7(_s0p7),
        .psum_g0_p8(_s0p8), .psum_g0_p9(_s0p9), .psum_g0_p10(_s0p10),.psum_g0_p11(_s0p11),
        .psum_g0_p12(_s0p12),.psum_g0_p13(_s0p13),.psum_g0_p14(_s0p14),.psum_g0_p15(ps0p15),
        .psum_g1_p0(ps1p0), .psum_g1_p1(_s1p1), .psum_g1_p2(_s1p2), .psum_g1_p3(_s1p3),
        .psum_g1_p4(_s1p4), .psum_g1_p5(_s1p5), .psum_g1_p6(_s1p6), .psum_g1_p7(_s1p7),
        .psum_g1_p8(_s1p8), .psum_g1_p9(_s1p9), .psum_g1_p10(_s1p10),.psum_g1_p11(_s1p11),
        .psum_g1_p12(_s1p12),.psum_g1_p13(_s1p13),.psum_g1_p14(_s1p14),.psum_g1_p15(ps1p15),
        .psum_g2_p0(ps2p0), .psum_g2_p1(_s2p1), .psum_g2_p2(_s2p2), .psum_g2_p3(_s2p3),
        .psum_g2_p4(_s2p4), .psum_g2_p5(_s2p5), .psum_g2_p6(_s2p6), .psum_g2_p7(_s2p7),
        .psum_g2_p8(_s2p8), .psum_g2_p9(_s2p9), .psum_g2_p10(_s2p10),.psum_g2_p11(_s2p11),
        .psum_g2_p12(_s2p12),.psum_g2_p13(_s2p13),.psum_g2_p14(_s2p14),.psum_g2_p15(ps2p15),
        .psum_g3_p0(ps3p0), .psum_g3_p1(_s3p1), .psum_g3_p2(_s3p2), .psum_g3_p3(_s3p3),
        .psum_g3_p4(_s3p4), .psum_g3_p5(_s3p5), .psum_g3_p6(_s3p6), .psum_g3_p7(_s3p7),
        .psum_g3_p8(_s3p8), .psum_g3_p9(_s3p9), .psum_g3_p10(_s3p10),.psum_g3_p11(_s3p11),
        .psum_g3_p12(_s3p12),.psum_g3_p13(_s3p13),.psum_g3_p14(_s3p14),.psum_g3_p15(ps3p15),
        .mac_valid(mac_valid)
    );

    initial clk=1'b0; always #5 clk=~clk;

    integer fail_count;

    task zero_all;
        begin
            wi0=0;wi1=0;wi2=0;wi3=0;wi4=0;wi5=0;wi6=0;wi7=0;
            wi8=0;wi9=0;wi10=0;wi11=0;wi12=0;wi13=0;wi14=0;wi15=0;
            wv_g0_0=0;wv_g0_1=0;wv_g0_2=0;wv_g0_3=0;wv_g0_4=0;wv_g0_5=0;wv_g0_6=0;wv_g0_7=0;
            wv_g0_8=0;wv_g0_9=0;wv_g0_10=0;wv_g0_11=0;wv_g0_12=0;wv_g0_13=0;wv_g0_14=0;wv_g0_15=0;
            wv_g1_0=0;wv_g1_1=0;wv_g1_2=0;wv_g1_3=0;wv_g1_4=0;wv_g1_5=0;wv_g1_6=0;wv_g1_7=0;
            wv_g1_8=0;wv_g1_9=0;wv_g1_10=0;wv_g1_11=0;wv_g1_12=0;wv_g1_13=0;wv_g1_14=0;wv_g1_15=0;
            wv_g2_0=0;wv_g2_1=0;wv_g2_2=0;wv_g2_3=0;wv_g2_4=0;wv_g2_5=0;wv_g2_6=0;wv_g2_7=0;
            wv_g2_8=0;wv_g2_9=0;wv_g2_10=0;wv_g2_11=0;wv_g2_12=0;wv_g2_13=0;wv_g2_14=0;wv_g2_15=0;
            wv_g3_0=0;wv_g3_1=0;wv_g3_2=0;wv_g3_3=0;wv_g3_4=0;wv_g3_5=0;wv_g3_6=0;wv_g3_7=0;
            wv_g3_8=0;wv_g3_9=0;wv_g3_10=0;wv_g3_11=0;wv_g3_12=0;wv_g3_13=0;wv_g3_14=0;wv_g3_15=0;
            bm_word=0;
            bv_g0_0=0;bv_g0_1=0;bv_g0_2=0;bv_g0_3=0;bv_g0_4=0;bv_g0_5=0;bv_g0_6=0;bv_g0_7=0;
            bv_g0_8=0;bv_g0_9=0;bv_g0_10=0;bv_g0_11=0;bv_g0_12=0;bv_g0_13=0;bv_g0_14=0;bv_g0_15=0;
            bv_g1_0=0;bv_g1_1=0;bv_g1_2=0;bv_g1_3=0;bv_g1_4=0;bv_g1_5=0;bv_g1_6=0;bv_g1_7=0;
            bv_g1_8=0;bv_g1_9=0;bv_g1_10=0;bv_g1_11=0;bv_g1_12=0;bv_g1_13=0;bv_g1_14=0;bv_g1_15=0;
            bv_g2_0=0;bv_g2_1=0;bv_g2_2=0;bv_g2_3=0;bv_g2_4=0;bv_g2_5=0;bv_g2_6=0;bv_g2_7=0;
            bv_g2_8=0;bv_g2_9=0;bv_g2_10=0;bv_g2_11=0;bv_g2_12=0;bv_g2_13=0;bv_g2_14=0;bv_g2_15=0;
            bv_g3_0=0;bv_g3_1=0;bv_g3_2=0;bv_g3_3=0;bv_g3_4=0;bv_g3_5=0;bv_g3_6=0;bv_g3_7=0;
            bv_g3_8=0;bv_g3_9=0;bv_g3_10=0;bv_g3_11=0;bv_g3_12=0;bv_g3_13=0;bv_g3_14=0;bv_g3_15=0;
            ag0p0=0;ag0p1=0;ag0p2=0;ag0p3=0;ag0p4=0;ag0p5=0;ag0p6=0;ag0p7=0;
            ag0p8=0;ag0p9=0;ag0p10=0;ag0p11=0;ag0p12=0;ag0p13=0;ag0p14=0;ag0p15=0;
            ag1p0=0;ag1p1=0;ag1p2=0;ag1p3=0;ag1p4=0;ag1p5=0;ag1p6=0;ag1p7=0;
            ag1p8=0;ag1p9=0;ag1p10=0;ag1p11=0;ag1p12=0;ag1p13=0;ag1p14=0;ag1p15=0;
            ag2p0=0;ag2p1=0;ag2p2=0;ag2p3=0;ag2p4=0;ag2p5=0;ag2p6=0;ag2p7=0;
            ag2p8=0;ag2p9=0;ag2p10=0;ag2p11=0;ag2p12=0;ag2p13=0;ag2p14=0;ag2p15=0;
            ag3p0=0;ag3p1=0;ag3p2=0;ag3p3=0;ag3p4=0;ag3p5=0;ag3p6=0;ag3p7=0;
            ag3p8=0;ag3p9=0;ag3p10=0;ag3p11=0;ag3p12=0;ag3p13=0;ag3p14=0;ag3p15=0;
            nc0=0;nc1=0;nc2=0;nc3=0;
        end
    endtask

    task do_load_act;
        begin
            @(negedge clk); shuf_mode=2'b00;
            @(posedge clk); @(negedge clk); shuf_mode=2'b10;
        end
    endtask

    task do_clr;
        begin
            @(negedge clk); acc_clr=1'b1;
            @(posedge clk); @(negedge clk); acc_clr=1'b0;
        end
    endtask

    task do_wt_load;
        begin
            @(negedge clk); wt_load=1'b1;
            @(posedge clk); @(negedge clk); wt_load=1'b0;
        end
    endtask

    task wait_psum;
        begin repeat(6) @(posedge clk); @(negedge clk); end
    endtask

    initial begin
        fail_count=0;
        rst_n=1'b0; wt_load=0; acc_clr=0;
        shuf_mode=2'b10; apply_relu=0; poc=0; n_cfg=0;
        zero_all;
        repeat(4) @(posedge clk); rst_n=1'b1; @(posedge clk);

        // Load activation window: act[g][p] = g*16+p
        ag0p0=0; ag0p1=1; ag0p2=2; ag0p3=3; ag0p4=4; ag0p5=5; ag0p6=6; ag0p7=7;
        ag0p8=8; ag0p9=9; ag0p10=10;ag0p11=11;ag0p12=12;ag0p13=13;ag0p14=14;ag0p15=15;
        ag1p0=16;ag1p1=17;ag1p2=18;ag1p3=19;ag1p4=20;ag1p5=21;ag1p6=22;ag1p7=23;
        ag1p8=24;ag1p9=25;ag1p10=26;ag1p11=27;ag1p12=28;ag1p13=29;ag1p14=30;ag1p15=31;
        ag2p0=32;ag2p1=33;ag2p2=34;ag2p3=35;ag2p4=36;ag2p5=37;ag2p6=38;ag2p7=39;
        ag2p8=40;ag2p9=41;ag2p10=42;ag2p11=43;ag2p12=44;ag2p13=45;ag2p14=46;ag2p15=47;
        ag3p0=48;ag3p1=49;ag3p2=50;ag3p3=51;ag3p4=52;ag3p5=53;ag3p6=54;ag3p7=55;
        ag3p8=56;ag3p9=57;ag3p10=58;ag3p11=59;ag3p12=60;ag3p13=61;ag3p14=62;ag3p15=63;
        do_load_act;
        do_clr;

        // ══════════════════════════════════════════════════════════════════
        // TC_01: poc=0, N=2, idx=[3,5], wt_g0=[10,20]
        //   act_sel_g0 at idx=3: act[g0][3]=3
        //   act_sel_g0 at idx=5: act[g0][5]=5
        //   psum_g0[all p] = 10*3 + 20*5 = 130
        //   psum_g1..g3 = 0
        // ══════════════════════════════════════════════════════════════════
        n_cfg=4'd2; poc=2'd0;
        wi0=4'd3; wi1=4'd5;
        wv_g0_0=8'sd10; wv_g0_1=8'sd20;
        do_wt_load; wait_psum;

        if (ps0p0!==32'sd130 || ps0p15!==32'sd130) begin
            $display("FAIL TC_01a: psum_g0 exp=130 got p0=%0d p15=%0d",ps0p0,ps0p15);
            fail_count=fail_count+1;
        end else $display("PASS TC_01a: psum_g0=130 broadcast confirmed");

        if (ps1p0!==32'sd0) begin
            $display("FAIL TC_01b: psum_g1 exp=0 got=%0d",ps1p0); fail_count=fail_count+1;
        end else $display("PASS TC_01b: psum_g1..g3=0");

        // ══════════════════════════════════════════════════════════════════
        // TC_02: poc=1, N=2, idx=[3,5], 4 groups with different weights
        //   psum_g0=260, psum_g1=290, psum_g2=-92, psum_g3=521
        // ══════════════════════════════════════════════════════════════════
        do_clr;
        n_cfg=4'd2; poc=2'd1;
        wi0=4'd3; wi1=4'd5;
        wv_g0_0=8'sd10;  wv_g0_1=8'sd20;
        wv_g1_0=-8'sd5;  wv_g1_1=8'sd15;
        wv_g2_0=8'sd1;   wv_g2_1=-8'sd3;
        wv_g3_0=8'sd7;   wv_g3_1=8'sd2;
        do_wt_load; wait_psum;

        if (ps0p0!==32'sd260) begin
            $display("FAIL TC_02a: psum_g0 exp=260 got=%0d",ps0p0); fail_count=fail_count+1;
        end else $display("PASS TC_02a: psum_g0=%0d (expected 260)", ps0p0);

        if (ps1p0!==32'sd290) begin
            $display("FAIL TC_02b: psum_g1 exp=290 got=%0d",ps1p0); fail_count=fail_count+1;
        end else $display("PASS TC_02b: psum_g1=%0d (expected 290)", ps1p0);

        if (ps2p0!==-32'sd92) begin
            $display("FAIL TC_02c: psum_g2 exp=-92 got=%0d",ps2p0); fail_count=fail_count+1;
        end else $display("PASS TC_02c: psum_g2=%0d (expected -92)", ps2p0);

        if (ps3p0!==32'sd521) begin
            $display("FAIL TC_02d: psum_g3 exp=521 got=%0d",ps3p0); fail_count=fail_count+1;
        end else $display("PASS TC_02d: psum_g3=%0d (expected 521)", ps3p0);

        // broadcast check p0==p15
        if (ps0p0!==ps0p15 || ps1p0!==ps1p15 || ps2p0!==ps2p15 || ps3p0!==ps3p15) begin
            $display("FAIL TC_02e: broadcast broken"); fail_count=fail_count+1;
        end else $display("PASS TC_02e: broadcast confirmed p0==p15");

        // ══════════════════════════════════════════════════════════════════
        // TC_03: acc_clr then reload -> same values as TC_02
        // ══════════════════════════════════════════════════════════════════
        do_clr;
        if (ps0p0!==32'sd0) begin
            $display("FAIL TC_03a: after clr exp=0 got=%0d",ps0p0); fail_count=fail_count+1;
        end else $display("PASS TC_03a: acc_clr OK");

        do_wt_load; wait_psum;
        if (ps0p0!==32'sd260 || ps1p0!==32'sd290 || ps2p0!==-32'sd92 || ps3p0!==32'sd521) begin
            $display("FAIL TC_03b: reload mismatch g0=%0d g1=%0d g2=%0d g3=%0d",
                     ps0p0,ps1p0,ps2p0,ps3p0); fail_count=fail_count+1;
        end else $display("PASS TC_03b: reload after clr correct");

        repeat(4) @(posedge clk);
        if (fail_count==0)
            $display("\n[PASS] tb_i_spec_e2e: all test cases passed");
        else
            $display("\n[FAIL] tb_i_spec_e2e: %0d failure(s)", fail_count);
        $finish;
    end

    initial begin #500000; $display("[FAIL] tb_i_spec_e2e: TIMEOUT"); $finish; end

endmodule
