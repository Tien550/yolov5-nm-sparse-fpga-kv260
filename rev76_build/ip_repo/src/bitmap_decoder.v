// bitmap_decoder.v  --  Scan 16-bit bitmap, emit one (addr,weight) pair/cycle
// Paper: "bm2onehot -> Onehot2Binary" Fig. 7b  (CLAUDE.md §8.5)
// Resource: ~40 LUT, ~20 FF.  Latency: N cycles for N non-zero bits.
//
// Algorithm:
//   isolate lowest bit : bm_onehot = bm_reg & (~bm_reg + 1)
//   priority encode    : OR-tree -> 4-bit binary position
//   clear bit          : bm_reg  <= bm_reg & ~bm_onehot

`timescale 1ns/1ps
`include "cnn_defines.vh"

module bitmap_decoder (
    input  wire                         clk,
    input  wire                         rst_n,       // synchronous active-low

    // Load interface
    input  wire [`M-1:0]                bm_in,       // 16-bit bitmap
    // weight values for all M positions (caller provides full tile)
    input  wire signed [`DATA_W-1:0]    vals_in_0,  input  wire signed [`DATA_W-1:0]    vals_in_1,
    input  wire signed [`DATA_W-1:0]    vals_in_2,  input  wire signed [`DATA_W-1:0]    vals_in_3,
    input  wire signed [`DATA_W-1:0]    vals_in_4,  input  wire signed [`DATA_W-1:0]    vals_in_5,
    input  wire signed [`DATA_W-1:0]    vals_in_6,  input  wire signed [`DATA_W-1:0]    vals_in_7,
    input  wire signed [`DATA_W-1:0]    vals_in_8,  input  wire signed [`DATA_W-1:0]    vals_in_9,
    input  wire signed [`DATA_W-1:0]    vals_in_10, input  wire signed [`DATA_W-1:0]    vals_in_11,
    input  wire signed [`DATA_W-1:0]    vals_in_12, input  wire signed [`DATA_W-1:0]    vals_in_13,
    input  wire signed [`DATA_W-1:0]    vals_in_14, input  wire signed [`DATA_W-1:0]    vals_in_15,
    input  wire                         load,        // pulse to load new tile

    // Output stream
    output wire [3:0]                   addr_out,    // binary position of current NZ
    output wire signed [`DATA_W-1:0]    weight_out,  // weight at that position
    output wire                         valid_out,   // one NZ emitted this cycle
    output wire                         done         // all NZ emitted (bm_reg==0)
);

    // ── Working registers ───────────────────────────────────────────────────
    reg [`M-1:0]             bm_reg;
    reg signed [`DATA_W-1:0] vals_reg [0:`M-1];

    // ── Isolate lowest set bit: bm_onehot = bm_reg & (-bm_reg) ────────────
    // Use explicit M-bit constant to avoid width-mismatch warning on +1
    wire [`M-1:0] bm_onehot;
    assign bm_onehot = bm_reg & (~bm_reg + 16'b1);

    // ── OR-tree priority encoder: 4-bit binary from one-hot ────────────────
    wire [3:0] bm_binary;
    assign bm_binary[3] = |bm_onehot[15:8];
    assign bm_binary[2] = |(bm_onehot[15:12] | bm_onehot[7:4]);
    assign bm_binary[1] = |(bm_onehot[15:14] | bm_onehot[11:10] |
                             bm_onehot[7:6]   | bm_onehot[3:2]);
    // bit[0]: set when onehot position is odd (bits 1,3,5,7,9,11,13,15)
    assign bm_binary[0] = bm_onehot[1]  | bm_onehot[3]  | bm_onehot[5]  | bm_onehot[7]  |
                          bm_onehot[9]  | bm_onehot[11] | bm_onehot[13] | bm_onehot[15];

    // ── Sequential logic ───────────────────────────────────────────────────
    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            bm_reg <= {`M{1'b0}};
            for (i = 0; i < `M; i = i + 1)
                vals_reg[i] <= {`DATA_W{1'b0}};
        end else if (load) begin
            bm_reg      <= bm_in;
            vals_reg[0]  <= vals_in_0;  vals_reg[1]  <= vals_in_1;
            vals_reg[2]  <= vals_in_2;  vals_reg[3]  <= vals_in_3;
            vals_reg[4]  <= vals_in_4;  vals_reg[5]  <= vals_in_5;
            vals_reg[6]  <= vals_in_6;  vals_reg[7]  <= vals_in_7;
            vals_reg[8]  <= vals_in_8;  vals_reg[9]  <= vals_in_9;
            vals_reg[10] <= vals_in_10; vals_reg[11] <= vals_in_11;
            vals_reg[12] <= vals_in_12; vals_reg[13] <= vals_in_13;
            vals_reg[14] <= vals_in_14; vals_reg[15] <= vals_in_15;
        end else if (bm_reg != {`M{1'b0}}) begin
            bm_reg <= bm_reg & ~bm_onehot;  // clear lowest set bit
        end
    end

    // ── Outputs ────────────────────────────────────────────────────────────
    assign valid_out  = (bm_reg != {`M{1'b0}}) && !load;
    assign done       = (bm_reg == {`M{1'b0}}) && !load;
    assign addr_out   = bm_binary;
    assign weight_out = vals_reg[bm_binary];

endmodule
