// accumulator.v  --  32-bit signed PSUM accumulator with synchronous clear
// Paper: "Accumulators (depth=M)" (CLAUDE.md §8.11)
// Resource: pure FF, 0 DSP, 0 BRAM

`timescale 1ns/1ps
`include "cnn_defines.vh"

module accumulator (
    input  wire                      clk,
    input  wire                      rst_n,   // synchronous active-low

    input  wire signed [`ACC_W-1:0]  data_in,
    input  wire                      en,      // accumulate when high
    input  wire                      clr,     // synchronous clear (MM-tile boundary)

    output reg  signed [`ACC_W-1:0]  psum
);

    // rst_n and clr both clear to zero; clr has priority over en.
    // Separated to help Vivado infer FF reset correctly (avoids LUT-reset).
    always @(posedge clk) begin
        if (!rst_n)
            psum <= {`ACC_W{1'b0}};
        else if (clr)
            psum <= {`ACC_W{1'b0}};
        else if (en)
            psum <= psum + data_in;
    end

endmodule
