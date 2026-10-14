// mul_group_array.v  --  G=4 groups, each with P=16 MAC units
// Paper: "Mul_Group 0..3" Fig. 3  (CLAUDE.md §8.10)
// DSP usage: G*P = 4*16 = 64 DSP48E2 per SPEC instance
// Key: all P MACs within one group share the SAME weight (broadcast)
//      pch_gen selects which activation register feeds each group
//
// Latency: 2 clock cycles (mac_unit pipeline)

`timescale 1ns/1ps
`include "cnn_defines.vh"

(* keep_hierarchy = "yes" *)
module mul_group_array (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        valid_in,

    // Weight inputs: one weight broadcast per group (G=4)
    input  wire signed [`DATA_W-1:0] wt_g0,
    input  wire signed [`DATA_W-1:0] wt_g1,
    input  wire signed [`DATA_W-1:0] wt_g2,
    input  wire signed [`DATA_W-1:0] wt_g3,

    // Activation inputs: G*P = 64 values (from depth_shuffle)
    input  wire signed [`DATA_W-1:0] act_g0_p0,  input  wire signed [`DATA_W-1:0] act_g0_p1,
    input  wire signed [`DATA_W-1:0] act_g0_p2,  input  wire signed [`DATA_W-1:0] act_g0_p3,
    input  wire signed [`DATA_W-1:0] act_g0_p4,  input  wire signed [`DATA_W-1:0] act_g0_p5,
    input  wire signed [`DATA_W-1:0] act_g0_p6,  input  wire signed [`DATA_W-1:0] act_g0_p7,
    input  wire signed [`DATA_W-1:0] act_g0_p8,  input  wire signed [`DATA_W-1:0] act_g0_p9,
    input  wire signed [`DATA_W-1:0] act_g0_p10, input  wire signed [`DATA_W-1:0] act_g0_p11,
    input  wire signed [`DATA_W-1:0] act_g0_p12, input  wire signed [`DATA_W-1:0] act_g0_p13,
    input  wire signed [`DATA_W-1:0] act_g0_p14, input  wire signed [`DATA_W-1:0] act_g0_p15,

    input  wire signed [`DATA_W-1:0] act_g1_p0,  input  wire signed [`DATA_W-1:0] act_g1_p1,
    input  wire signed [`DATA_W-1:0] act_g1_p2,  input  wire signed [`DATA_W-1:0] act_g1_p3,
    input  wire signed [`DATA_W-1:0] act_g1_p4,  input  wire signed [`DATA_W-1:0] act_g1_p5,
    input  wire signed [`DATA_W-1:0] act_g1_p6,  input  wire signed [`DATA_W-1:0] act_g1_p7,
    input  wire signed [`DATA_W-1:0] act_g1_p8,  input  wire signed [`DATA_W-1:0] act_g1_p9,
    input  wire signed [`DATA_W-1:0] act_g1_p10, input  wire signed [`DATA_W-1:0] act_g1_p11,
    input  wire signed [`DATA_W-1:0] act_g1_p12, input  wire signed [`DATA_W-1:0] act_g1_p13,
    input  wire signed [`DATA_W-1:0] act_g1_p14, input  wire signed [`DATA_W-1:0] act_g1_p15,

    input  wire signed [`DATA_W-1:0] act_g2_p0,  input  wire signed [`DATA_W-1:0] act_g2_p1,
    input  wire signed [`DATA_W-1:0] act_g2_p2,  input  wire signed [`DATA_W-1:0] act_g2_p3,
    input  wire signed [`DATA_W-1:0] act_g2_p4,  input  wire signed [`DATA_W-1:0] act_g2_p5,
    input  wire signed [`DATA_W-1:0] act_g2_p6,  input  wire signed [`DATA_W-1:0] act_g2_p7,
    input  wire signed [`DATA_W-1:0] act_g2_p8,  input  wire signed [`DATA_W-1:0] act_g2_p9,
    input  wire signed [`DATA_W-1:0] act_g2_p10, input  wire signed [`DATA_W-1:0] act_g2_p11,
    input  wire signed [`DATA_W-1:0] act_g2_p12, input  wire signed [`DATA_W-1:0] act_g2_p13,
    input  wire signed [`DATA_W-1:0] act_g2_p14, input  wire signed [`DATA_W-1:0] act_g2_p15,

    input  wire signed [`DATA_W-1:0] act_g3_p0,  input  wire signed [`DATA_W-1:0] act_g3_p1,
    input  wire signed [`DATA_W-1:0] act_g3_p2,  input  wire signed [`DATA_W-1:0] act_g3_p3,
    input  wire signed [`DATA_W-1:0] act_g3_p4,  input  wire signed [`DATA_W-1:0] act_g3_p5,
    input  wire signed [`DATA_W-1:0] act_g3_p6,  input  wire signed [`DATA_W-1:0] act_g3_p7,
    input  wire signed [`DATA_W-1:0] act_g3_p8,  input  wire signed [`DATA_W-1:0] act_g3_p9,
    input  wire signed [`DATA_W-1:0] act_g3_p10, input  wire signed [`DATA_W-1:0] act_g3_p11,
    input  wire signed [`DATA_W-1:0] act_g3_p12, input  wire signed [`DATA_W-1:0] act_g3_p13,
    input  wire signed [`DATA_W-1:0] act_g3_p14, input  wire signed [`DATA_W-1:0] act_g3_p15,

    // Product outputs: G*P = 64 INT32 products (to psum_array)
    output wire signed [`ACC_W-1:0] prod_g0_p0,  output wire signed [`ACC_W-1:0] prod_g0_p1,
    output wire signed [`ACC_W-1:0] prod_g0_p2,  output wire signed [`ACC_W-1:0] prod_g0_p3,
    output wire signed [`ACC_W-1:0] prod_g0_p4,  output wire signed [`ACC_W-1:0] prod_g0_p5,
    output wire signed [`ACC_W-1:0] prod_g0_p6,  output wire signed [`ACC_W-1:0] prod_g0_p7,
    output wire signed [`ACC_W-1:0] prod_g0_p8,  output wire signed [`ACC_W-1:0] prod_g0_p9,
    output wire signed [`ACC_W-1:0] prod_g0_p10, output wire signed [`ACC_W-1:0] prod_g0_p11,
    output wire signed [`ACC_W-1:0] prod_g0_p12, output wire signed [`ACC_W-1:0] prod_g0_p13,
    output wire signed [`ACC_W-1:0] prod_g0_p14, output wire signed [`ACC_W-1:0] prod_g0_p15,

    output wire signed [`ACC_W-1:0] prod_g1_p0,  output wire signed [`ACC_W-1:0] prod_g1_p1,
    output wire signed [`ACC_W-1:0] prod_g1_p2,  output wire signed [`ACC_W-1:0] prod_g1_p3,
    output wire signed [`ACC_W-1:0] prod_g1_p4,  output wire signed [`ACC_W-1:0] prod_g1_p5,
    output wire signed [`ACC_W-1:0] prod_g1_p6,  output wire signed [`ACC_W-1:0] prod_g1_p7,
    output wire signed [`ACC_W-1:0] prod_g1_p8,  output wire signed [`ACC_W-1:0] prod_g1_p9,
    output wire signed [`ACC_W-1:0] prod_g1_p10, output wire signed [`ACC_W-1:0] prod_g1_p11,
    output wire signed [`ACC_W-1:0] prod_g1_p12, output wire signed [`ACC_W-1:0] prod_g1_p13,
    output wire signed [`ACC_W-1:0] prod_g1_p14, output wire signed [`ACC_W-1:0] prod_g1_p15,

    output wire signed [`ACC_W-1:0] prod_g2_p0,  output wire signed [`ACC_W-1:0] prod_g2_p1,
    output wire signed [`ACC_W-1:0] prod_g2_p2,  output wire signed [`ACC_W-1:0] prod_g2_p3,
    output wire signed [`ACC_W-1:0] prod_g2_p4,  output wire signed [`ACC_W-1:0] prod_g2_p5,
    output wire signed [`ACC_W-1:0] prod_g2_p6,  output wire signed [`ACC_W-1:0] prod_g2_p7,
    output wire signed [`ACC_W-1:0] prod_g2_p8,  output wire signed [`ACC_W-1:0] prod_g2_p9,
    output wire signed [`ACC_W-1:0] prod_g2_p10, output wire signed [`ACC_W-1:0] prod_g2_p11,
    output wire signed [`ACC_W-1:0] prod_g2_p12, output wire signed [`ACC_W-1:0] prod_g2_p13,
    output wire signed [`ACC_W-1:0] prod_g2_p14, output wire signed [`ACC_W-1:0] prod_g2_p15,

    output wire signed [`ACC_W-1:0] prod_g3_p0,  output wire signed [`ACC_W-1:0] prod_g3_p1,
    output wire signed [`ACC_W-1:0] prod_g3_p2,  output wire signed [`ACC_W-1:0] prod_g3_p3,
    output wire signed [`ACC_W-1:0] prod_g3_p4,  output wire signed [`ACC_W-1:0] prod_g3_p5,
    output wire signed [`ACC_W-1:0] prod_g3_p6,  output wire signed [`ACC_W-1:0] prod_g3_p7,
    output wire signed [`ACC_W-1:0] prod_g3_p8,  output wire signed [`ACC_W-1:0] prod_g3_p9,
    output wire signed [`ACC_W-1:0] prod_g3_p10, output wire signed [`ACC_W-1:0] prod_g3_p11,
    output wire signed [`ACC_W-1:0] prod_g3_p12, output wire signed [`ACC_W-1:0] prod_g3_p13,
    output wire signed [`ACC_W-1:0] prod_g3_p14, output wire signed [`ACC_W-1:0] prod_g3_p15,

    output wire        valid_out  // valid_in delayed 2 cycles
);

    // Instantiate G*P = 64 mac_units
    // Group 0 (weight = wt_g0 broadcast)
    mac_unit u_g0_p0  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p0), .valid_in(valid_in),.p_out(prod_g0_p0), .valid_out(valid_out));
    mac_unit u_g0_p1  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p1), .valid_in(valid_in),.p_out(prod_g0_p1), .valid_out());
    mac_unit u_g0_p2  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p2), .valid_in(valid_in),.p_out(prod_g0_p2), .valid_out());
    mac_unit u_g0_p3  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p3), .valid_in(valid_in),.p_out(prod_g0_p3), .valid_out());
    mac_unit u_g0_p4  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p4), .valid_in(valid_in),.p_out(prod_g0_p4), .valid_out());
    mac_unit u_g0_p5  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p5), .valid_in(valid_in),.p_out(prod_g0_p5), .valid_out());
    mac_unit u_g0_p6  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p6), .valid_in(valid_in),.p_out(prod_g0_p6), .valid_out());
    mac_unit u_g0_p7  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p7), .valid_in(valid_in),.p_out(prod_g0_p7), .valid_out());
    mac_unit u_g0_p8  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p8), .valid_in(valid_in),.p_out(prod_g0_p8), .valid_out());
    mac_unit u_g0_p9  (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p9), .valid_in(valid_in),.p_out(prod_g0_p9), .valid_out());
    mac_unit u_g0_p10 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p10),.valid_in(valid_in),.p_out(prod_g0_p10),.valid_out());
    mac_unit u_g0_p11 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p11),.valid_in(valid_in),.p_out(prod_g0_p11),.valid_out());
    mac_unit u_g0_p12 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p12),.valid_in(valid_in),.p_out(prod_g0_p12),.valid_out());
    mac_unit u_g0_p13 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p13),.valid_in(valid_in),.p_out(prod_g0_p13),.valid_out());
    mac_unit u_g0_p14 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p14),.valid_in(valid_in),.p_out(prod_g0_p14),.valid_out());
    mac_unit u_g0_p15 (.clk(clk),.rst_n(rst_n),.a_in(wt_g0),.b_in(act_g0_p15),.valid_in(valid_in),.p_out(prod_g0_p15),.valid_out());

    // Group 1
    mac_unit u_g1_p0  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p0), .valid_in(valid_in),.p_out(prod_g1_p0), .valid_out());
    mac_unit u_g1_p1  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p1), .valid_in(valid_in),.p_out(prod_g1_p1), .valid_out());
    mac_unit u_g1_p2  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p2), .valid_in(valid_in),.p_out(prod_g1_p2), .valid_out());
    mac_unit u_g1_p3  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p3), .valid_in(valid_in),.p_out(prod_g1_p3), .valid_out());
    mac_unit u_g1_p4  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p4), .valid_in(valid_in),.p_out(prod_g1_p4), .valid_out());
    mac_unit u_g1_p5  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p5), .valid_in(valid_in),.p_out(prod_g1_p5), .valid_out());
    mac_unit u_g1_p6  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p6), .valid_in(valid_in),.p_out(prod_g1_p6), .valid_out());
    mac_unit u_g1_p7  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p7), .valid_in(valid_in),.p_out(prod_g1_p7), .valid_out());
    mac_unit u_g1_p8  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p8), .valid_in(valid_in),.p_out(prod_g1_p8), .valid_out());
    mac_unit u_g1_p9  (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p9), .valid_in(valid_in),.p_out(prod_g1_p9), .valid_out());
    mac_unit u_g1_p10 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p10),.valid_in(valid_in),.p_out(prod_g1_p10),.valid_out());
    mac_unit u_g1_p11 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p11),.valid_in(valid_in),.p_out(prod_g1_p11),.valid_out());
    mac_unit u_g1_p12 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p12),.valid_in(valid_in),.p_out(prod_g1_p12),.valid_out());
    mac_unit u_g1_p13 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p13),.valid_in(valid_in),.p_out(prod_g1_p13),.valid_out());
    mac_unit u_g1_p14 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p14),.valid_in(valid_in),.p_out(prod_g1_p14),.valid_out());
    mac_unit u_g1_p15 (.clk(clk),.rst_n(rst_n),.a_in(wt_g1),.b_in(act_g1_p15),.valid_in(valid_in),.p_out(prod_g1_p15),.valid_out());

    // Group 2
    mac_unit u_g2_p0  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p0), .valid_in(valid_in),.p_out(prod_g2_p0), .valid_out());
    mac_unit u_g2_p1  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p1), .valid_in(valid_in),.p_out(prod_g2_p1), .valid_out());
    mac_unit u_g2_p2  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p2), .valid_in(valid_in),.p_out(prod_g2_p2), .valid_out());
    mac_unit u_g2_p3  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p3), .valid_in(valid_in),.p_out(prod_g2_p3), .valid_out());
    mac_unit u_g2_p4  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p4), .valid_in(valid_in),.p_out(prod_g2_p4), .valid_out());
    mac_unit u_g2_p5  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p5), .valid_in(valid_in),.p_out(prod_g2_p5), .valid_out());
    mac_unit u_g2_p6  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p6), .valid_in(valid_in),.p_out(prod_g2_p6), .valid_out());
    mac_unit u_g2_p7  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p7), .valid_in(valid_in),.p_out(prod_g2_p7), .valid_out());
    mac_unit u_g2_p8  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p8), .valid_in(valid_in),.p_out(prod_g2_p8), .valid_out());
    mac_unit u_g2_p9  (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p9), .valid_in(valid_in),.p_out(prod_g2_p9), .valid_out());
    mac_unit u_g2_p10 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p10),.valid_in(valid_in),.p_out(prod_g2_p10),.valid_out());
    mac_unit u_g2_p11 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p11),.valid_in(valid_in),.p_out(prod_g2_p11),.valid_out());
    mac_unit u_g2_p12 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p12),.valid_in(valid_in),.p_out(prod_g2_p12),.valid_out());
    mac_unit u_g2_p13 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p13),.valid_in(valid_in),.p_out(prod_g2_p13),.valid_out());
    mac_unit u_g2_p14 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p14),.valid_in(valid_in),.p_out(prod_g2_p14),.valid_out());
    mac_unit u_g2_p15 (.clk(clk),.rst_n(rst_n),.a_in(wt_g2),.b_in(act_g2_p15),.valid_in(valid_in),.p_out(prod_g2_p15),.valid_out());

    // Group 3
    mac_unit u_g3_p0  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p0), .valid_in(valid_in),.p_out(prod_g3_p0), .valid_out());
    mac_unit u_g3_p1  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p1), .valid_in(valid_in),.p_out(prod_g3_p1), .valid_out());
    mac_unit u_g3_p2  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p2), .valid_in(valid_in),.p_out(prod_g3_p2), .valid_out());
    mac_unit u_g3_p3  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p3), .valid_in(valid_in),.p_out(prod_g3_p3), .valid_out());
    mac_unit u_g3_p4  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p4), .valid_in(valid_in),.p_out(prod_g3_p4), .valid_out());
    mac_unit u_g3_p5  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p5), .valid_in(valid_in),.p_out(prod_g3_p5), .valid_out());
    mac_unit u_g3_p6  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p6), .valid_in(valid_in),.p_out(prod_g3_p6), .valid_out());
    mac_unit u_g3_p7  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p7), .valid_in(valid_in),.p_out(prod_g3_p7), .valid_out());
    mac_unit u_g3_p8  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p8), .valid_in(valid_in),.p_out(prod_g3_p8), .valid_out());
    mac_unit u_g3_p9  (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p9), .valid_in(valid_in),.p_out(prod_g3_p9), .valid_out());
    mac_unit u_g3_p10 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p10),.valid_in(valid_in),.p_out(prod_g3_p10),.valid_out());
    mac_unit u_g3_p11 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p11),.valid_in(valid_in),.p_out(prod_g3_p11),.valid_out());
    mac_unit u_g3_p12 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p12),.valid_in(valid_in),.p_out(prod_g3_p12),.valid_out());
    mac_unit u_g3_p13 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p13),.valid_in(valid_in),.p_out(prod_g3_p13),.valid_out());
    mac_unit u_g3_p14 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p14),.valid_in(valid_in),.p_out(prod_g3_p14),.valid_out());
    mac_unit u_g3_p15 (.clk(clk),.rst_n(rst_n),.a_in(wt_g3),.b_in(act_g3_p15),.valid_in(valid_in),.p_out(prod_g3_p15),.valid_out());

endmodule
