// accu_top.v  --  Cross-SPEC PSUM accumulation  (CLAUDE.md §8.15)
// Paper: "ACCU module" Section 4.3
//
// Modes (accu_mode):
//   2'b00 = 1-in: pass SPEC0 directly to RR
//   2'b01 = 2-in: SPEC0+SPEC1 → RR
//   2'b10 = 4-in: SPEC0+SPEC1+SPEC2+SPEC3 → RR (adder tree)
//
// All logic combinational. G*P=64 output positions computed in parallel.
// For KV260 (N_SPEC=4) only. Ports cover all 4 SPECs.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module accu_top (
    input  wire [1:0] accu_mode,   // 00=1-in 01=2-in 10=4-in

    // SPEC 0 (64 PSUMs)
    input  wire signed [`ACC_W-1:0] s0_g0p0,s0_g0p1,s0_g0p2,s0_g0p3,s0_g0p4,s0_g0p5,s0_g0p6,s0_g0p7,
    input  wire signed [`ACC_W-1:0] s0_g0p8,s0_g0p9,s0_g0p10,s0_g0p11,s0_g0p12,s0_g0p13,s0_g0p14,s0_g0p15,
    input  wire signed [`ACC_W-1:0] s0_g1p0,s0_g1p1,s0_g1p2,s0_g1p3,s0_g1p4,s0_g1p5,s0_g1p6,s0_g1p7,
    input  wire signed [`ACC_W-1:0] s0_g1p8,s0_g1p9,s0_g1p10,s0_g1p11,s0_g1p12,s0_g1p13,s0_g1p14,s0_g1p15,
    input  wire signed [`ACC_W-1:0] s0_g2p0,s0_g2p1,s0_g2p2,s0_g2p3,s0_g2p4,s0_g2p5,s0_g2p6,s0_g2p7,
    input  wire signed [`ACC_W-1:0] s0_g2p8,s0_g2p9,s0_g2p10,s0_g2p11,s0_g2p12,s0_g2p13,s0_g2p14,s0_g2p15,
    input  wire signed [`ACC_W-1:0] s0_g3p0,s0_g3p1,s0_g3p2,s0_g3p3,s0_g3p4,s0_g3p5,s0_g3p6,s0_g3p7,
    input  wire signed [`ACC_W-1:0] s0_g3p8,s0_g3p9,s0_g3p10,s0_g3p11,s0_g3p12,s0_g3p13,s0_g3p14,s0_g3p15,

    // SPEC 1
    input  wire signed [`ACC_W-1:0] s1_g0p0,s1_g0p1,s1_g0p2,s1_g0p3,s1_g0p4,s1_g0p5,s1_g0p6,s1_g0p7,
    input  wire signed [`ACC_W-1:0] s1_g0p8,s1_g0p9,s1_g0p10,s1_g0p11,s1_g0p12,s1_g0p13,s1_g0p14,s1_g0p15,
    input  wire signed [`ACC_W-1:0] s1_g1p0,s1_g1p1,s1_g1p2,s1_g1p3,s1_g1p4,s1_g1p5,s1_g1p6,s1_g1p7,
    input  wire signed [`ACC_W-1:0] s1_g1p8,s1_g1p9,s1_g1p10,s1_g1p11,s1_g1p12,s1_g1p13,s1_g1p14,s1_g1p15,
    input  wire signed [`ACC_W-1:0] s1_g2p0,s1_g2p1,s1_g2p2,s1_g2p3,s1_g2p4,s1_g2p5,s1_g2p6,s1_g2p7,
    input  wire signed [`ACC_W-1:0] s1_g2p8,s1_g2p9,s1_g2p10,s1_g2p11,s1_g2p12,s1_g2p13,s1_g2p14,s1_g2p15,
    input  wire signed [`ACC_W-1:0] s1_g3p0,s1_g3p1,s1_g3p2,s1_g3p3,s1_g3p4,s1_g3p5,s1_g3p6,s1_g3p7,
    input  wire signed [`ACC_W-1:0] s1_g3p8,s1_g3p9,s1_g3p10,s1_g3p11,s1_g3p12,s1_g3p13,s1_g3p14,s1_g3p15,

    // SPEC 2
    input  wire signed [`ACC_W-1:0] s2_g0p0,s2_g0p1,s2_g0p2,s2_g0p3,s2_g0p4,s2_g0p5,s2_g0p6,s2_g0p7,
    input  wire signed [`ACC_W-1:0] s2_g0p8,s2_g0p9,s2_g0p10,s2_g0p11,s2_g0p12,s2_g0p13,s2_g0p14,s2_g0p15,
    input  wire signed [`ACC_W-1:0] s2_g1p0,s2_g1p1,s2_g1p2,s2_g1p3,s2_g1p4,s2_g1p5,s2_g1p6,s2_g1p7,
    input  wire signed [`ACC_W-1:0] s2_g1p8,s2_g1p9,s2_g1p10,s2_g1p11,s2_g1p12,s2_g1p13,s2_g1p14,s2_g1p15,
    input  wire signed [`ACC_W-1:0] s2_g2p0,s2_g2p1,s2_g2p2,s2_g2p3,s2_g2p4,s2_g2p5,s2_g2p6,s2_g2p7,
    input  wire signed [`ACC_W-1:0] s2_g2p8,s2_g2p9,s2_g2p10,s2_g2p11,s2_g2p12,s2_g2p13,s2_g2p14,s2_g2p15,
    input  wire signed [`ACC_W-1:0] s2_g3p0,s2_g3p1,s2_g3p2,s2_g3p3,s2_g3p4,s2_g3p5,s2_g3p6,s2_g3p7,
    input  wire signed [`ACC_W-1:0] s2_g3p8,s2_g3p9,s2_g3p10,s2_g3p11,s2_g3p12,s2_g3p13,s2_g3p14,s2_g3p15,

    // SPEC 3
    input  wire signed [`ACC_W-1:0] s3_g0p0,s3_g0p1,s3_g0p2,s3_g0p3,s3_g0p4,s3_g0p5,s3_g0p6,s3_g0p7,
    input  wire signed [`ACC_W-1:0] s3_g0p8,s3_g0p9,s3_g0p10,s3_g0p11,s3_g0p12,s3_g0p13,s3_g0p14,s3_g0p15,
    input  wire signed [`ACC_W-1:0] s3_g1p0,s3_g1p1,s3_g1p2,s3_g1p3,s3_g1p4,s3_g1p5,s3_g1p6,s3_g1p7,
    input  wire signed [`ACC_W-1:0] s3_g1p8,s3_g1p9,s3_g1p10,s3_g1p11,s3_g1p12,s3_g1p13,s3_g1p14,s3_g1p15,
    input  wire signed [`ACC_W-1:0] s3_g2p0,s3_g2p1,s3_g2p2,s3_g2p3,s3_g2p4,s3_g2p5,s3_g2p6,s3_g2p7,
    input  wire signed [`ACC_W-1:0] s3_g2p8,s3_g2p9,s3_g2p10,s3_g2p11,s3_g2p12,s3_g2p13,s3_g2p14,s3_g2p15,
    input  wire signed [`ACC_W-1:0] s3_g3p0,s3_g3p1,s3_g3p2,s3_g3p3,s3_g3p4,s3_g3p5,s3_g3p6,s3_g3p7,
    input  wire signed [`ACC_W-1:0] s3_g3p8,s3_g3p9,s3_g3p10,s3_g3p11,s3_g3p12,s3_g3p13,s3_g3p14,s3_g3p15,

    // Accumulated outputs (64 PSUMs, combinational)
    output wire signed [`ACC_W-1:0] out_g0p0,out_g0p1,out_g0p2,out_g0p3,out_g0p4,out_g0p5,out_g0p6,out_g0p7,
    output wire signed [`ACC_W-1:0] out_g0p8,out_g0p9,out_g0p10,out_g0p11,out_g0p12,out_g0p13,out_g0p14,out_g0p15,
    output wire signed [`ACC_W-1:0] out_g1p0,out_g1p1,out_g1p2,out_g1p3,out_g1p4,out_g1p5,out_g1p6,out_g1p7,
    output wire signed [`ACC_W-1:0] out_g1p8,out_g1p9,out_g1p10,out_g1p11,out_g1p12,out_g1p13,out_g1p14,out_g1p15,
    output wire signed [`ACC_W-1:0] out_g2p0,out_g2p1,out_g2p2,out_g2p3,out_g2p4,out_g2p5,out_g2p6,out_g2p7,
    output wire signed [`ACC_W-1:0] out_g2p8,out_g2p9,out_g2p10,out_g2p11,out_g2p12,out_g2p13,out_g2p14,out_g2p15,
    output wire signed [`ACC_W-1:0] out_g3p0,out_g3p1,out_g3p2,out_g3p3,out_g3p4,out_g3p5,out_g3p6,out_g3p7,
    output wire signed [`ACC_W-1:0] out_g3p8,out_g3p9,out_g3p10,out_g3p11,out_g3p12,out_g3p13,out_g3p14,out_g3p15
);

    localparam MODE_1IN = 2'b00;
    localparam MODE_2IN = 2'b01;
    localparam MODE_4IN = 2'b10;

    // Internal function: select accumulation value based on mode
    // Using a function avoids repeating the ternary 64 times
    function signed [`ACC_W-1:0] accu_sel;
        input signed [`ACC_W-1:0] a0, a1, a2, a3;
        begin
            case (accu_mode)
                MODE_4IN: accu_sel = (a0 + a1) + (a2 + a3);
                MODE_2IN: accu_sel = a0 + a1;
                default:  accu_sel = a0;
            endcase
        end
    endfunction

    // Group 0
    assign out_g0p0  = accu_sel(s0_g0p0,  s1_g0p0,  s2_g0p0,  s3_g0p0);
    assign out_g0p1  = accu_sel(s0_g0p1,  s1_g0p1,  s2_g0p1,  s3_g0p1);
    assign out_g0p2  = accu_sel(s0_g0p2,  s1_g0p2,  s2_g0p2,  s3_g0p2);
    assign out_g0p3  = accu_sel(s0_g0p3,  s1_g0p3,  s2_g0p3,  s3_g0p3);
    assign out_g0p4  = accu_sel(s0_g0p4,  s1_g0p4,  s2_g0p4,  s3_g0p4);
    assign out_g0p5  = accu_sel(s0_g0p5,  s1_g0p5,  s2_g0p5,  s3_g0p5);
    assign out_g0p6  = accu_sel(s0_g0p6,  s1_g0p6,  s2_g0p6,  s3_g0p6);
    assign out_g0p7  = accu_sel(s0_g0p7,  s1_g0p7,  s2_g0p7,  s3_g0p7);
    assign out_g0p8  = accu_sel(s0_g0p8,  s1_g0p8,  s2_g0p8,  s3_g0p8);
    assign out_g0p9  = accu_sel(s0_g0p9,  s1_g0p9,  s2_g0p9,  s3_g0p9);
    assign out_g0p10 = accu_sel(s0_g0p10, s1_g0p10, s2_g0p10, s3_g0p10);
    assign out_g0p11 = accu_sel(s0_g0p11, s1_g0p11, s2_g0p11, s3_g0p11);
    assign out_g0p12 = accu_sel(s0_g0p12, s1_g0p12, s2_g0p12, s3_g0p12);
    assign out_g0p13 = accu_sel(s0_g0p13, s1_g0p13, s2_g0p13, s3_g0p13);
    assign out_g0p14 = accu_sel(s0_g0p14, s1_g0p14, s2_g0p14, s3_g0p14);
    assign out_g0p15 = accu_sel(s0_g0p15, s1_g0p15, s2_g0p15, s3_g0p15);

    // Group 1
    assign out_g1p0  = accu_sel(s0_g1p0,  s1_g1p0,  s2_g1p0,  s3_g1p0);
    assign out_g1p1  = accu_sel(s0_g1p1,  s1_g1p1,  s2_g1p1,  s3_g1p1);
    assign out_g1p2  = accu_sel(s0_g1p2,  s1_g1p2,  s2_g1p2,  s3_g1p2);
    assign out_g1p3  = accu_sel(s0_g1p3,  s1_g1p3,  s2_g1p3,  s3_g1p3);
    assign out_g1p4  = accu_sel(s0_g1p4,  s1_g1p4,  s2_g1p4,  s3_g1p4);
    assign out_g1p5  = accu_sel(s0_g1p5,  s1_g1p5,  s2_g1p5,  s3_g1p5);
    assign out_g1p6  = accu_sel(s0_g1p6,  s1_g1p6,  s2_g1p6,  s3_g1p6);
    assign out_g1p7  = accu_sel(s0_g1p7,  s1_g1p7,  s2_g1p7,  s3_g1p7);
    assign out_g1p8  = accu_sel(s0_g1p8,  s1_g1p8,  s2_g1p8,  s3_g1p8);
    assign out_g1p9  = accu_sel(s0_g1p9,  s1_g1p9,  s2_g1p9,  s3_g1p9);
    assign out_g1p10 = accu_sel(s0_g1p10, s1_g1p10, s2_g1p10, s3_g1p10);
    assign out_g1p11 = accu_sel(s0_g1p11, s1_g1p11, s2_g1p11, s3_g1p11);
    assign out_g1p12 = accu_sel(s0_g1p12, s1_g1p12, s2_g1p12, s3_g1p12);
    assign out_g1p13 = accu_sel(s0_g1p13, s1_g1p13, s2_g1p13, s3_g1p13);
    assign out_g1p14 = accu_sel(s0_g1p14, s1_g1p14, s2_g1p14, s3_g1p14);
    assign out_g1p15 = accu_sel(s0_g1p15, s1_g1p15, s2_g1p15, s3_g1p15);

    // Group 2
    assign out_g2p0  = accu_sel(s0_g2p0,  s1_g2p0,  s2_g2p0,  s3_g2p0);
    assign out_g2p1  = accu_sel(s0_g2p1,  s1_g2p1,  s2_g2p1,  s3_g2p1);
    assign out_g2p2  = accu_sel(s0_g2p2,  s1_g2p2,  s2_g2p2,  s3_g2p2);
    assign out_g2p3  = accu_sel(s0_g2p3,  s1_g2p3,  s2_g2p3,  s3_g2p3);
    assign out_g2p4  = accu_sel(s0_g2p4,  s1_g2p4,  s2_g2p4,  s3_g2p4);
    assign out_g2p5  = accu_sel(s0_g2p5,  s1_g2p5,  s2_g2p5,  s3_g2p5);
    assign out_g2p6  = accu_sel(s0_g2p6,  s1_g2p6,  s2_g2p6,  s3_g2p6);
    assign out_g2p7  = accu_sel(s0_g2p7,  s1_g2p7,  s2_g2p7,  s3_g2p7);
    assign out_g2p8  = accu_sel(s0_g2p8,  s1_g2p8,  s2_g2p8,  s3_g2p8);
    assign out_g2p9  = accu_sel(s0_g2p9,  s1_g2p9,  s2_g2p9,  s3_g2p9);
    assign out_g2p10 = accu_sel(s0_g2p10, s1_g2p10, s2_g2p10, s3_g2p10);
    assign out_g2p11 = accu_sel(s0_g2p11, s1_g2p11, s2_g2p11, s3_g2p11);
    assign out_g2p12 = accu_sel(s0_g2p12, s1_g2p12, s2_g2p12, s3_g2p12);
    assign out_g2p13 = accu_sel(s0_g2p13, s1_g2p13, s2_g2p13, s3_g2p13);
    assign out_g2p14 = accu_sel(s0_g2p14, s1_g2p14, s2_g2p14, s3_g2p14);
    assign out_g2p15 = accu_sel(s0_g2p15, s1_g2p15, s2_g2p15, s3_g2p15);

    // Group 3
    assign out_g3p0  = accu_sel(s0_g3p0,  s1_g3p0,  s2_g3p0,  s3_g3p0);
    assign out_g3p1  = accu_sel(s0_g3p1,  s1_g3p1,  s2_g3p1,  s3_g3p1);
    assign out_g3p2  = accu_sel(s0_g3p2,  s1_g3p2,  s2_g3p2,  s3_g3p2);
    assign out_g3p3  = accu_sel(s0_g3p3,  s1_g3p3,  s2_g3p3,  s3_g3p3);
    assign out_g3p4  = accu_sel(s0_g3p4,  s1_g3p4,  s2_g3p4,  s3_g3p4);
    assign out_g3p5  = accu_sel(s0_g3p5,  s1_g3p5,  s2_g3p5,  s3_g3p5);
    assign out_g3p6  = accu_sel(s0_g3p6,  s1_g3p6,  s2_g3p6,  s3_g3p6);
    assign out_g3p7  = accu_sel(s0_g3p7,  s1_g3p7,  s2_g3p7,  s3_g3p7);
    assign out_g3p8  = accu_sel(s0_g3p8,  s1_g3p8,  s2_g3p8,  s3_g3p8);
    assign out_g3p9  = accu_sel(s0_g3p9,  s1_g3p9,  s2_g3p9,  s3_g3p9);
    assign out_g3p10 = accu_sel(s0_g3p10, s1_g3p10, s2_g3p10, s3_g3p10);
    assign out_g3p11 = accu_sel(s0_g3p11, s1_g3p11, s2_g3p11, s3_g3p11);
    assign out_g3p12 = accu_sel(s0_g3p12, s1_g3p12, s2_g3p12, s3_g3p12);
    assign out_g3p13 = accu_sel(s0_g3p13, s1_g3p13, s2_g3p13, s3_g3p13);
    assign out_g3p14 = accu_sel(s0_g3p14, s1_g3p14, s2_g3p14, s3_g3p14);
    assign out_g3p15 = accu_sel(s0_g3p15, s1_g3p15, s2_g3p15, s3_g3p15);

endmodule
