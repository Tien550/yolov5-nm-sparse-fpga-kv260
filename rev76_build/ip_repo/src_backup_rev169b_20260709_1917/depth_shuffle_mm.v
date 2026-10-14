// depth_shuffle_mm.v -- MM-tile Depth Shuffle (paper Fig.3 / §4.1, FULL on-chip COO)
// Paper Micromachines 2023 §4.1: "M×P inputs ... allocated in Depth Shuffle".
//
// Difference vs depth_shuffle.v (the G×P=64 simplified, n_cfg=1 host-fed version):
//   This holds M×P = 16 input-channels × 16 spatial positions = 256 regs per SPEC.
//   The NZ index (addr from weight_decoder/coo_decoder) SELECTS which input
//   channel's 16 spatial activations are output → enables n_cfg=N on-chip COO
//   drain (one DMA load per M-tile, decode N entries on-chip) = real DMA savings.
//
// For I-SPEC (input-channel sparse): all G groups share the SAME addr (same NZ
// positions), so one addr selects act[addr][0..15] broadcast to all G groups'
// P=16 mults. Each group multiplies by its own (per-output-channel) weight.
//
// Modes:
//   SHUF_LOAD (2'b00): latch full M×P window from load bus (one DMA M-tile)
//   SHUF_SEL  (2'b01): hold; output = act_mem[sel_addr][p]  (per-NZ select)
//   SHUF_HOLD (2'b10): hold; output unchanged
//
// Flat ports (Verilog-2001, no packed arrays) to match i_spec style.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module depth_shuffle_mm (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  shuffle_mode,        // 00=LOAD 01=SEL 10=HOLD

    // NZ input-channel index from decoder (selects which of M ic to output)
    input  wire [3:0]  sel_addr,

    // Full M×P load bus: load_ic{ic}_p{p}  (ic=0..15, p=0..15) = 256 ports.
    // Packed as a flat 256*8-bit bus to keep the port list sane.
    input  wire [`M*`P*`DATA_W-1:0] load_bus,   // [ic*P+p] byte = act[ic][p]

    // Output: 16 spatial activations for the selected input channel
    output wire signed [`DATA_W-1:0] act_out_0,  output wire signed [`DATA_W-1:0] act_out_1,
    output wire signed [`DATA_W-1:0] act_out_2,  output wire signed [`DATA_W-1:0] act_out_3,
    output wire signed [`DATA_W-1:0] act_out_4,  output wire signed [`DATA_W-1:0] act_out_5,
    output wire signed [`DATA_W-1:0] act_out_6,  output wire signed [`DATA_W-1:0] act_out_7,
    output wire signed [`DATA_W-1:0] act_out_8,  output wire signed [`DATA_W-1:0] act_out_9,
    output wire signed [`DATA_W-1:0] act_out_10, output wire signed [`DATA_W-1:0] act_out_11,
    output wire signed [`DATA_W-1:0] act_out_12, output wire signed [`DATA_W-1:0] act_out_13,
    output wire signed [`DATA_W-1:0] act_out_14, output wire signed [`DATA_W-1:0] act_out_15
);

    // ── MM-tile storage: act_mem[ic][p], ic=0..M-1, p=0..P-1 ──────────────────
    reg signed [`DATA_W-1:0] act_mem [0:`M-1][0:`P-1];

    integer ic, p;
    always @(posedge clk) begin
        if (!rst_n) begin
            for (ic = 0; ic < `M; ic = ic + 1)
                for (p = 0; p < `P; p = p + 1)
                    act_mem[ic][p] <= {`DATA_W{1'b0}};
        end else if (shuffle_mode == `SHUF_LOAD) begin
            for (ic = 0; ic < `M; ic = ic + 1)
                for (p = 0; p < `P; p = p + 1)
                    act_mem[ic][p] <= $signed(load_bus[(ic*`P + p)*`DATA_W +: `DATA_W]);
        end
        // SEL / HOLD: storage unchanged (combinational read below)
    end

    // ── Combinational read: select row sel_addr, expose 16 spatial values ─────
    assign act_out_0  = act_mem[sel_addr][0];   assign act_out_1  = act_mem[sel_addr][1];
    assign act_out_2  = act_mem[sel_addr][2];   assign act_out_3  = act_mem[sel_addr][3];
    assign act_out_4  = act_mem[sel_addr][4];   assign act_out_5  = act_mem[sel_addr][5];
    assign act_out_6  = act_mem[sel_addr][6];   assign act_out_7  = act_mem[sel_addr][7];
    assign act_out_8  = act_mem[sel_addr][8];   assign act_out_9  = act_mem[sel_addr][9];
    assign act_out_10 = act_mem[sel_addr][10];  assign act_out_11 = act_mem[sel_addr][11];
    assign act_out_12 = act_mem[sel_addr][12];  assign act_out_13 = act_mem[sel_addr][13];
    assign act_out_14 = act_mem[sel_addr][14];  assign act_out_15 = act_mem[sel_addr][15];

endmodule
