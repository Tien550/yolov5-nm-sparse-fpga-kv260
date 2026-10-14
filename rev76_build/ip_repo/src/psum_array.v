// psum_array.v  --  G*P = 64 accumulators, 32-bit each
// Paper: "Accumulators (depth=M)" Fig. 3a  (CLAUDE.md §8.11)
// Resource: G*P*ACC_W = 4*16*32 = 2048 FF, 0 DSP, 0 BRAM
// acc_clr resets all 64 registers synchronously (MM-tile boundary)

`timescale 1ns/1ps
`include "cnn_defines.vh"

module psum_array (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        en,      // accumulate when high (valid from mul_group_array)
    input  wire        acc_clr, // synchronous clear all PSUMs (MM-tile boundary)

    // Product inputs from mul_group_array (G*P = 64)
    input  wire signed [`ACC_W-1:0] prod_g0_p0,  input  wire signed [`ACC_W-1:0] prod_g0_p1,
    input  wire signed [`ACC_W-1:0] prod_g0_p2,  input  wire signed [`ACC_W-1:0] prod_g0_p3,
    input  wire signed [`ACC_W-1:0] prod_g0_p4,  input  wire signed [`ACC_W-1:0] prod_g0_p5,
    input  wire signed [`ACC_W-1:0] prod_g0_p6,  input  wire signed [`ACC_W-1:0] prod_g0_p7,
    input  wire signed [`ACC_W-1:0] prod_g0_p8,  input  wire signed [`ACC_W-1:0] prod_g0_p9,
    input  wire signed [`ACC_W-1:0] prod_g0_p10, input  wire signed [`ACC_W-1:0] prod_g0_p11,
    input  wire signed [`ACC_W-1:0] prod_g0_p12, input  wire signed [`ACC_W-1:0] prod_g0_p13,
    input  wire signed [`ACC_W-1:0] prod_g0_p14, input  wire signed [`ACC_W-1:0] prod_g0_p15,

    input  wire signed [`ACC_W-1:0] prod_g1_p0,  input  wire signed [`ACC_W-1:0] prod_g1_p1,
    input  wire signed [`ACC_W-1:0] prod_g1_p2,  input  wire signed [`ACC_W-1:0] prod_g1_p3,
    input  wire signed [`ACC_W-1:0] prod_g1_p4,  input  wire signed [`ACC_W-1:0] prod_g1_p5,
    input  wire signed [`ACC_W-1:0] prod_g1_p6,  input  wire signed [`ACC_W-1:0] prod_g1_p7,
    input  wire signed [`ACC_W-1:0] prod_g1_p8,  input  wire signed [`ACC_W-1:0] prod_g1_p9,
    input  wire signed [`ACC_W-1:0] prod_g1_p10, input  wire signed [`ACC_W-1:0] prod_g1_p11,
    input  wire signed [`ACC_W-1:0] prod_g1_p12, input  wire signed [`ACC_W-1:0] prod_g1_p13,
    input  wire signed [`ACC_W-1:0] prod_g1_p14, input  wire signed [`ACC_W-1:0] prod_g1_p15,

    input  wire signed [`ACC_W-1:0] prod_g2_p0,  input  wire signed [`ACC_W-1:0] prod_g2_p1,
    input  wire signed [`ACC_W-1:0] prod_g2_p2,  input  wire signed [`ACC_W-1:0] prod_g2_p3,
    input  wire signed [`ACC_W-1:0] prod_g2_p4,  input  wire signed [`ACC_W-1:0] prod_g2_p5,
    input  wire signed [`ACC_W-1:0] prod_g2_p6,  input  wire signed [`ACC_W-1:0] prod_g2_p7,
    input  wire signed [`ACC_W-1:0] prod_g2_p8,  input  wire signed [`ACC_W-1:0] prod_g2_p9,
    input  wire signed [`ACC_W-1:0] prod_g2_p10, input  wire signed [`ACC_W-1:0] prod_g2_p11,
    input  wire signed [`ACC_W-1:0] prod_g2_p12, input  wire signed [`ACC_W-1:0] prod_g2_p13,
    input  wire signed [`ACC_W-1:0] prod_g2_p14, input  wire signed [`ACC_W-1:0] prod_g2_p15,

    input  wire signed [`ACC_W-1:0] prod_g3_p0,  input  wire signed [`ACC_W-1:0] prod_g3_p1,
    input  wire signed [`ACC_W-1:0] prod_g3_p2,  input  wire signed [`ACC_W-1:0] prod_g3_p3,
    input  wire signed [`ACC_W-1:0] prod_g3_p4,  input  wire signed [`ACC_W-1:0] prod_g3_p5,
    input  wire signed [`ACC_W-1:0] prod_g3_p6,  input  wire signed [`ACC_W-1:0] prod_g3_p7,
    input  wire signed [`ACC_W-1:0] prod_g3_p8,  input  wire signed [`ACC_W-1:0] prod_g3_p9,
    input  wire signed [`ACC_W-1:0] prod_g3_p10, input  wire signed [`ACC_W-1:0] prod_g3_p11,
    input  wire signed [`ACC_W-1:0] prod_g3_p12, input  wire signed [`ACC_W-1:0] prod_g3_p13,
    input  wire signed [`ACC_W-1:0] prod_g3_p14, input  wire signed [`ACC_W-1:0] prod_g3_p15,

    // PSUM outputs (G*P = 64)
    output wire signed [`ACC_W-1:0] psum_g0_p0,  output wire signed [`ACC_W-1:0] psum_g0_p1,
    output wire signed [`ACC_W-1:0] psum_g0_p2,  output wire signed [`ACC_W-1:0] psum_g0_p3,
    output wire signed [`ACC_W-1:0] psum_g0_p4,  output wire signed [`ACC_W-1:0] psum_g0_p5,
    output wire signed [`ACC_W-1:0] psum_g0_p6,  output wire signed [`ACC_W-1:0] psum_g0_p7,
    output wire signed [`ACC_W-1:0] psum_g0_p8,  output wire signed [`ACC_W-1:0] psum_g0_p9,
    output wire signed [`ACC_W-1:0] psum_g0_p10, output wire signed [`ACC_W-1:0] psum_g0_p11,
    output wire signed [`ACC_W-1:0] psum_g0_p12, output wire signed [`ACC_W-1:0] psum_g0_p13,
    output wire signed [`ACC_W-1:0] psum_g0_p14, output wire signed [`ACC_W-1:0] psum_g0_p15,

    output wire signed [`ACC_W-1:0] psum_g1_p0,  output wire signed [`ACC_W-1:0] psum_g1_p1,
    output wire signed [`ACC_W-1:0] psum_g1_p2,  output wire signed [`ACC_W-1:0] psum_g1_p3,
    output wire signed [`ACC_W-1:0] psum_g1_p4,  output wire signed [`ACC_W-1:0] psum_g1_p5,
    output wire signed [`ACC_W-1:0] psum_g1_p6,  output wire signed [`ACC_W-1:0] psum_g1_p7,
    output wire signed [`ACC_W-1:0] psum_g1_p8,  output wire signed [`ACC_W-1:0] psum_g1_p9,
    output wire signed [`ACC_W-1:0] psum_g1_p10, output wire signed [`ACC_W-1:0] psum_g1_p11,
    output wire signed [`ACC_W-1:0] psum_g1_p12, output wire signed [`ACC_W-1:0] psum_g1_p13,
    output wire signed [`ACC_W-1:0] psum_g1_p14, output wire signed [`ACC_W-1:0] psum_g1_p15,

    output wire signed [`ACC_W-1:0] psum_g2_p0,  output wire signed [`ACC_W-1:0] psum_g2_p1,
    output wire signed [`ACC_W-1:0] psum_g2_p2,  output wire signed [`ACC_W-1:0] psum_g2_p3,
    output wire signed [`ACC_W-1:0] psum_g2_p4,  output wire signed [`ACC_W-1:0] psum_g2_p5,
    output wire signed [`ACC_W-1:0] psum_g2_p6,  output wire signed [`ACC_W-1:0] psum_g2_p7,
    output wire signed [`ACC_W-1:0] psum_g2_p8,  output wire signed [`ACC_W-1:0] psum_g2_p9,
    output wire signed [`ACC_W-1:0] psum_g2_p10, output wire signed [`ACC_W-1:0] psum_g2_p11,
    output wire signed [`ACC_W-1:0] psum_g2_p12, output wire signed [`ACC_W-1:0] psum_g2_p13,
    output wire signed [`ACC_W-1:0] psum_g2_p14, output wire signed [`ACC_W-1:0] psum_g2_p15,

    output wire signed [`ACC_W-1:0] psum_g3_p0,  output wire signed [`ACC_W-1:0] psum_g3_p1,
    output wire signed [`ACC_W-1:0] psum_g3_p2,  output wire signed [`ACC_W-1:0] psum_g3_p3,
    output wire signed [`ACC_W-1:0] psum_g3_p4,  output wire signed [`ACC_W-1:0] psum_g3_p5,
    output wire signed [`ACC_W-1:0] psum_g3_p6,  output wire signed [`ACC_W-1:0] psum_g3_p7,
    output wire signed [`ACC_W-1:0] psum_g3_p8,  output wire signed [`ACC_W-1:0] psum_g3_p9,
    output wire signed [`ACC_W-1:0] psum_g3_p10, output wire signed [`ACC_W-1:0] psum_g3_p11,
    output wire signed [`ACC_W-1:0] psum_g3_p12, output wire signed [`ACC_W-1:0] psum_g3_p13,
    output wire signed [`ACC_W-1:0] psum_g3_p14, output wire signed [`ACC_W-1:0] psum_g3_p15
);

    // Instantiate G*P = 64 accumulator instances
    // Group 0
    accumulator u_g0_p0  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p0), .en(en),.clr(acc_clr),.psum(psum_g0_p0));
    accumulator u_g0_p1  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p1), .en(en),.clr(acc_clr),.psum(psum_g0_p1));
    accumulator u_g0_p2  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p2), .en(en),.clr(acc_clr),.psum(psum_g0_p2));
    accumulator u_g0_p3  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p3), .en(en),.clr(acc_clr),.psum(psum_g0_p3));
    accumulator u_g0_p4  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p4), .en(en),.clr(acc_clr),.psum(psum_g0_p4));
    accumulator u_g0_p5  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p5), .en(en),.clr(acc_clr),.psum(psum_g0_p5));
    accumulator u_g0_p6  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p6), .en(en),.clr(acc_clr),.psum(psum_g0_p6));
    accumulator u_g0_p7  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p7), .en(en),.clr(acc_clr),.psum(psum_g0_p7));
    accumulator u_g0_p8  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p8), .en(en),.clr(acc_clr),.psum(psum_g0_p8));
    accumulator u_g0_p9  (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p9), .en(en),.clr(acc_clr),.psum(psum_g0_p9));
    accumulator u_g0_p10 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p10),.en(en),.clr(acc_clr),.psum(psum_g0_p10));
    accumulator u_g0_p11 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p11),.en(en),.clr(acc_clr),.psum(psum_g0_p11));
    accumulator u_g0_p12 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p12),.en(en),.clr(acc_clr),.psum(psum_g0_p12));
    accumulator u_g0_p13 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p13),.en(en),.clr(acc_clr),.psum(psum_g0_p13));
    accumulator u_g0_p14 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p14),.en(en),.clr(acc_clr),.psum(psum_g0_p14));
    accumulator u_g0_p15 (.clk(clk),.rst_n(rst_n),.data_in(prod_g0_p15),.en(en),.clr(acc_clr),.psum(psum_g0_p15));

    // Group 1
    accumulator u_g1_p0  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p0), .en(en),.clr(acc_clr),.psum(psum_g1_p0));
    accumulator u_g1_p1  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p1), .en(en),.clr(acc_clr),.psum(psum_g1_p1));
    accumulator u_g1_p2  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p2), .en(en),.clr(acc_clr),.psum(psum_g1_p2));
    accumulator u_g1_p3  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p3), .en(en),.clr(acc_clr),.psum(psum_g1_p3));
    accumulator u_g1_p4  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p4), .en(en),.clr(acc_clr),.psum(psum_g1_p4));
    accumulator u_g1_p5  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p5), .en(en),.clr(acc_clr),.psum(psum_g1_p5));
    accumulator u_g1_p6  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p6), .en(en),.clr(acc_clr),.psum(psum_g1_p6));
    accumulator u_g1_p7  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p7), .en(en),.clr(acc_clr),.psum(psum_g1_p7));
    accumulator u_g1_p8  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p8), .en(en),.clr(acc_clr),.psum(psum_g1_p8));
    accumulator u_g1_p9  (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p9), .en(en),.clr(acc_clr),.psum(psum_g1_p9));
    accumulator u_g1_p10 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p10),.en(en),.clr(acc_clr),.psum(psum_g1_p10));
    accumulator u_g1_p11 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p11),.en(en),.clr(acc_clr),.psum(psum_g1_p11));
    accumulator u_g1_p12 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p12),.en(en),.clr(acc_clr),.psum(psum_g1_p12));
    accumulator u_g1_p13 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p13),.en(en),.clr(acc_clr),.psum(psum_g1_p13));
    accumulator u_g1_p14 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p14),.en(en),.clr(acc_clr),.psum(psum_g1_p14));
    accumulator u_g1_p15 (.clk(clk),.rst_n(rst_n),.data_in(prod_g1_p15),.en(en),.clr(acc_clr),.psum(psum_g1_p15));

    // Group 2
    accumulator u_g2_p0  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p0), .en(en),.clr(acc_clr),.psum(psum_g2_p0));
    accumulator u_g2_p1  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p1), .en(en),.clr(acc_clr),.psum(psum_g2_p1));
    accumulator u_g2_p2  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p2), .en(en),.clr(acc_clr),.psum(psum_g2_p2));
    accumulator u_g2_p3  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p3), .en(en),.clr(acc_clr),.psum(psum_g2_p3));
    accumulator u_g2_p4  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p4), .en(en),.clr(acc_clr),.psum(psum_g2_p4));
    accumulator u_g2_p5  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p5), .en(en),.clr(acc_clr),.psum(psum_g2_p5));
    accumulator u_g2_p6  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p6), .en(en),.clr(acc_clr),.psum(psum_g2_p6));
    accumulator u_g2_p7  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p7), .en(en),.clr(acc_clr),.psum(psum_g2_p7));
    accumulator u_g2_p8  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p8), .en(en),.clr(acc_clr),.psum(psum_g2_p8));
    accumulator u_g2_p9  (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p9), .en(en),.clr(acc_clr),.psum(psum_g2_p9));
    accumulator u_g2_p10 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p10),.en(en),.clr(acc_clr),.psum(psum_g2_p10));
    accumulator u_g2_p11 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p11),.en(en),.clr(acc_clr),.psum(psum_g2_p11));
    accumulator u_g2_p12 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p12),.en(en),.clr(acc_clr),.psum(psum_g2_p12));
    accumulator u_g2_p13 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p13),.en(en),.clr(acc_clr),.psum(psum_g2_p13));
    accumulator u_g2_p14 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p14),.en(en),.clr(acc_clr),.psum(psum_g2_p14));
    accumulator u_g2_p15 (.clk(clk),.rst_n(rst_n),.data_in(prod_g2_p15),.en(en),.clr(acc_clr),.psum(psum_g2_p15));

    // Group 3
    accumulator u_g3_p0  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p0), .en(en),.clr(acc_clr),.psum(psum_g3_p0));
    accumulator u_g3_p1  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p1), .en(en),.clr(acc_clr),.psum(psum_g3_p1));
    accumulator u_g3_p2  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p2), .en(en),.clr(acc_clr),.psum(psum_g3_p2));
    accumulator u_g3_p3  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p3), .en(en),.clr(acc_clr),.psum(psum_g3_p3));
    accumulator u_g3_p4  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p4), .en(en),.clr(acc_clr),.psum(psum_g3_p4));
    accumulator u_g3_p5  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p5), .en(en),.clr(acc_clr),.psum(psum_g3_p5));
    accumulator u_g3_p6  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p6), .en(en),.clr(acc_clr),.psum(psum_g3_p6));
    accumulator u_g3_p7  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p7), .en(en),.clr(acc_clr),.psum(psum_g3_p7));
    accumulator u_g3_p8  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p8), .en(en),.clr(acc_clr),.psum(psum_g3_p8));
    accumulator u_g3_p9  (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p9), .en(en),.clr(acc_clr),.psum(psum_g3_p9));
    accumulator u_g3_p10 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p10),.en(en),.clr(acc_clr),.psum(psum_g3_p10));
    accumulator u_g3_p11 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p11),.en(en),.clr(acc_clr),.psum(psum_g3_p11));
    accumulator u_g3_p12 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p12),.en(en),.clr(acc_clr),.psum(psum_g3_p12));
    accumulator u_g3_p13 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p13),.en(en),.clr(acc_clr),.psum(psum_g3_p13));
    accumulator u_g3_p14 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p14),.en(en),.clr(acc_clr),.psum(psum_g3_p14));
    accumulator u_g3_p15 (.clk(clk),.rst_n(rst_n),.data_in(prod_g3_p15),.en(en),.clr(acc_clr),.psum(psum_g3_p15));

endmodule
