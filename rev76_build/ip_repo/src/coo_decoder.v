// coo_decoder.v  --  Drain N packed (4-bit index, 8-bit value) pairs, 1/cycle
// Paper: "COO encoding" Section 4.2  (CLAUDE.md §8.6)
// Resource: ~20 LUT + N*12 FF.  n_cfg is runtime (1..4 for COO path).

`timescale 1ns/1ps
`include "cnn_defines.vh"

module coo_decoder (
    input  wire                         clk,
    input  wire                         rst_n,   // synchronous active-low

    // Load interface -- caller provides all N entries at once
    input  wire [3:0]                   idx_in_0,  input  wire [3:0]  idx_in_1,
    input  wire [3:0]                   idx_in_2,  input  wire [3:0]  idx_in_3,
    input  wire [3:0]                   idx_in_4,  input  wire [3:0]  idx_in_5,
    input  wire [3:0]                   idx_in_6,  input  wire [3:0]  idx_in_7,
    input  wire [3:0]                   idx_in_8,  input  wire [3:0]  idx_in_9,
    input  wire [3:0]                   idx_in_10, input  wire [3:0]  idx_in_11,
    input  wire [3:0]                   idx_in_12, input  wire [3:0]  idx_in_13,
    input  wire [3:0]                   idx_in_14, input  wire [3:0]  idx_in_15,

    input  wire signed [`DATA_W-1:0]    val_in_0,  input  wire signed [`DATA_W-1:0] val_in_1,
    input  wire signed [`DATA_W-1:0]    val_in_2,  input  wire signed [`DATA_W-1:0] val_in_3,
    input  wire signed [`DATA_W-1:0]    val_in_4,  input  wire signed [`DATA_W-1:0] val_in_5,
    input  wire signed [`DATA_W-1:0]    val_in_6,  input  wire signed [`DATA_W-1:0] val_in_7,
    input  wire signed [`DATA_W-1:0]    val_in_8,  input  wire signed [`DATA_W-1:0] val_in_9,
    input  wire signed [`DATA_W-1:0]    val_in_10, input  wire signed [`DATA_W-1:0] val_in_11,
    input  wire signed [`DATA_W-1:0]    val_in_12, input  wire signed [`DATA_W-1:0] val_in_13,
    input  wire signed [`DATA_W-1:0]    val_in_14, input  wire signed [`DATA_W-1:0] val_in_15,

    input  wire [3:0]                   n_cfg,   // number of NZ to drain (1..M)
    input  wire                         load,    // pulse to load one M-tile

    // Output stream
    output wire [3:0]                   addr_out,    // index of current NZ
    output wire signed [`DATA_W-1:0]    weight_out,  // value of current NZ
    output wire                         valid_out,
    output wire                         done
);

    reg [3:0]              idx_reg [0:`M-1];
    reg signed [`DATA_W-1:0] val_reg [0:`M-1];
    reg [3:0]              n_reg;   // latched n_cfg
    reg [3:0]              cnt;     // drain counter

    integer i;
    always @(posedge clk) begin
        if (!rst_n) begin
            cnt   <= 4'd0;
            n_reg <= 4'd0;
            for (i = 0; i < `M; i = i + 1) begin
                idx_reg[i] <= 4'd0;
                val_reg[i] <= {`DATA_W{1'b0}};
            end
        end else if (load) begin
            n_reg  <= n_cfg;
            cnt    <= 4'd0;
            idx_reg[0]  <= idx_in_0;  idx_reg[1]  <= idx_in_1;
            idx_reg[2]  <= idx_in_2;  idx_reg[3]  <= idx_in_3;
            idx_reg[4]  <= idx_in_4;  idx_reg[5]  <= idx_in_5;
            idx_reg[6]  <= idx_in_6;  idx_reg[7]  <= idx_in_7;
            idx_reg[8]  <= idx_in_8;  idx_reg[9]  <= idx_in_9;
            idx_reg[10] <= idx_in_10; idx_reg[11] <= idx_in_11;
            idx_reg[12] <= idx_in_12; idx_reg[13] <= idx_in_13;
            idx_reg[14] <= idx_in_14; idx_reg[15] <= idx_in_15;
            val_reg[0]  <= val_in_0;  val_reg[1]  <= val_in_1;
            val_reg[2]  <= val_in_2;  val_reg[3]  <= val_in_3;
            val_reg[4]  <= val_in_4;  val_reg[5]  <= val_in_5;
            val_reg[6]  <= val_in_6;  val_reg[7]  <= val_in_7;
            val_reg[8]  <= val_in_8;  val_reg[9]  <= val_in_9;
            val_reg[10] <= val_in_10; val_reg[11] <= val_in_11;
            val_reg[12] <= val_in_12; val_reg[13] <= val_in_13;
            val_reg[14] <= val_in_14; val_reg[15] <= val_in_15;
        end else if (cnt < n_reg) begin
            cnt <= cnt + 4'd1;
        end
    end

    assign valid_out  = (cnt < n_reg) && !load;
    assign done       = (cnt >= n_reg) && !load;
    assign addr_out   = idx_reg[cnt];
    assign weight_out = val_reg[cnt];

endmodule
