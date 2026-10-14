// pch_gen.v  --  P-CH Generation: compute input register address per group
// Paper: "P-CH Generation" Fig. 7a  (CLAUDE.md §8.8)
// Purely combinational -- 0 FF, ~8 LUT
//
// For each group g in [0, G-1]:
//   in_reg_addr[g] = poc_cnt * poc + g
// poc is runtime (1, 2, or 4). Use case statement to avoid inferred multiplier.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module pch_gen (
    input  wire [7:0]  poc_cnt,   // output-channel tile counter
    input  wire [1:0]  poc,       // output channel parallelism: 0=1, 1=2, 2=4

    // G=4 output addresses (one per multiplier group)
    output wire [7:0]  in_reg_addr_0,
    output wire [7:0]  in_reg_addr_1,
    output wire [7:0]  in_reg_addr_2,
    output wire [7:0]  in_reg_addr_3
);

    wire [7:0] base;  // poc_cnt * poc_val (computed below)

    // Use shift instead of multiply to stay LUT-only (no DSP)
    // poc=0 -> poc_val=1, poc=1 -> poc_val=2, poc=2 -> poc_val=4
    assign base = (poc == 2'd2) ? {poc_cnt[5:0], 2'b00} :  // *4
                  (poc == 2'd1) ? {poc_cnt[6:0], 1'b0}  :  // *2
                                   poc_cnt;                  // *1

    assign in_reg_addr_0 = base;
    assign in_reg_addr_1 = base + 8'd1;
    assign in_reg_addr_2 = base + 8'd2;
    assign in_reg_addr_3 = base + 8'd3;

endmodule
