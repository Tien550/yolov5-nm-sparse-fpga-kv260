// weight_decoder.v  --  MUX: select COO or Bitmap decoder based on n_cfg
// Paper: "Weight Decoder" Fig. 7a  (CLAUDE.md §8.7)
// use_bitmap = (n_cfg >= 5): purely combinational select, latched on load

`timescale 1ns/1ps
`include "cnn_defines.vh"

module weight_decoder (
    input  wire                         clk,
    input  wire                         rst_n,

    // Runtime config
    input  wire [3:0]                   n_cfg,       // NZ count per M-block (1..16)

    // COO path inputs
    input  wire [3:0]  coo_idx_0,  input  wire [3:0]  coo_idx_1,
    input  wire [3:0]  coo_idx_2,  input  wire [3:0]  coo_idx_3,
    input  wire [3:0]  coo_idx_4,  input  wire [3:0]  coo_idx_5,
    input  wire [3:0]  coo_idx_6,  input  wire [3:0]  coo_idx_7,
    input  wire [3:0]  coo_idx_8,  input  wire [3:0]  coo_idx_9,
    input  wire [3:0]  coo_idx_10, input  wire [3:0]  coo_idx_11,
    input  wire [3:0]  coo_idx_12, input  wire [3:0]  coo_idx_13,
    input  wire [3:0]  coo_idx_14, input  wire [3:0]  coo_idx_15,

    input  wire signed [`DATA_W-1:0] coo_val_0,  input  wire signed [`DATA_W-1:0] coo_val_1,
    input  wire signed [`DATA_W-1:0] coo_val_2,  input  wire signed [`DATA_W-1:0] coo_val_3,
    input  wire signed [`DATA_W-1:0] coo_val_4,  input  wire signed [`DATA_W-1:0] coo_val_5,
    input  wire signed [`DATA_W-1:0] coo_val_6,  input  wire signed [`DATA_W-1:0] coo_val_7,
    input  wire signed [`DATA_W-1:0] coo_val_8,  input  wire signed [`DATA_W-1:0] coo_val_9,
    input  wire signed [`DATA_W-1:0] coo_val_10, input  wire signed [`DATA_W-1:0] coo_val_11,
    input  wire signed [`DATA_W-1:0] coo_val_12, input  wire signed [`DATA_W-1:0] coo_val_13,
    input  wire signed [`DATA_W-1:0] coo_val_14, input  wire signed [`DATA_W-1:0] coo_val_15,

    // Bitmap path inputs
    input  wire [`M-1:0]              bm_word,     // 16-bit bitmap

    input  wire signed [`DATA_W-1:0] bm_val_0,  input  wire signed [`DATA_W-1:0] bm_val_1,
    input  wire signed [`DATA_W-1:0] bm_val_2,  input  wire signed [`DATA_W-1:0] bm_val_3,
    input  wire signed [`DATA_W-1:0] bm_val_4,  input  wire signed [`DATA_W-1:0] bm_val_5,
    input  wire signed [`DATA_W-1:0] bm_val_6,  input  wire signed [`DATA_W-1:0] bm_val_7,
    input  wire signed [`DATA_W-1:0] bm_val_8,  input  wire signed [`DATA_W-1:0] bm_val_9,
    input  wire signed [`DATA_W-1:0] bm_val_10, input  wire signed [`DATA_W-1:0] bm_val_11,
    input  wire signed [`DATA_W-1:0] bm_val_12, input  wire signed [`DATA_W-1:0] bm_val_13,
    input  wire signed [`DATA_W-1:0] bm_val_14, input  wire signed [`DATA_W-1:0] bm_val_15,

    // Common
    input  wire                      load,        // pulse to load one M-tile

    // Output stream (same interface regardless of encoding path)
    output wire [3:0]                addr_out,    // position within M-block
    output wire signed [`DATA_W-1:0] weight_out,  // INT8 weight value
    output wire                      valid_out,
    output wire                      done
);

    // ── Path select: COO if n_cfg<=4, Bitmap if n_cfg>=5 ───────────────────
    wire use_bitmap;
    assign use_bitmap = (n_cfg >= 4'd5);

    // Latch path at load time so MUX stays stable during drain.
    // Rev152 200MHz: max_fanout — use_bm_r drives addr_out/weight_out mux → fanout 368 tới DSP
    // (net routing 3.3ns = critical @200). Replicate (functionally-neutral) → mỗi copy ít load.
    (* max_fanout = 32 *) reg use_bm_r;
    always @(posedge clk) begin
        if (!rst_n)    use_bm_r <= 1'b0;
        else if (load) use_bm_r <= use_bitmap;
    end

    // ── Sub-decoder load enables ────────────────────────────────────────────
    wire coo_load = load && !use_bitmap;
    wire bm_load  = load &&  use_bitmap;

    // ── COO decoder ────────────────────────────────────────────────────────
    wire [3:0]              coo_addr;
    wire signed [`DATA_W-1:0] coo_wt;
    wire                    coo_vld, coo_done;

    coo_decoder u_coo (
        .clk        (clk),        .rst_n      (rst_n),
        .idx_in_0   (coo_idx_0),  .idx_in_1   (coo_idx_1),
        .idx_in_2   (coo_idx_2),  .idx_in_3   (coo_idx_3),
        .idx_in_4   (coo_idx_4),  .idx_in_5   (coo_idx_5),
        .idx_in_6   (coo_idx_6),  .idx_in_7   (coo_idx_7),
        .idx_in_8   (coo_idx_8),  .idx_in_9   (coo_idx_9),
        .idx_in_10  (coo_idx_10), .idx_in_11  (coo_idx_11),
        .idx_in_12  (coo_idx_12), .idx_in_13  (coo_idx_13),
        .idx_in_14  (coo_idx_14), .idx_in_15  (coo_idx_15),
        .val_in_0   (coo_val_0),  .val_in_1   (coo_val_1),
        .val_in_2   (coo_val_2),  .val_in_3   (coo_val_3),
        .val_in_4   (coo_val_4),  .val_in_5   (coo_val_5),
        .val_in_6   (coo_val_6),  .val_in_7   (coo_val_7),
        .val_in_8   (coo_val_8),  .val_in_9   (coo_val_9),
        .val_in_10  (coo_val_10), .val_in_11  (coo_val_11),
        .val_in_12  (coo_val_12), .val_in_13  (coo_val_13),
        .val_in_14  (coo_val_14), .val_in_15  (coo_val_15),
        .n_cfg      (n_cfg),      .load       (coo_load),
        .addr_out   (coo_addr),   .weight_out (coo_wt),
        .valid_out  (coo_vld),    .done       (coo_done)
    );

    // ── Bitmap decoder ──────────────────────────────────────────────────────
    wire [3:0]              bm_addr;
    wire signed [`DATA_W-1:0] bm_wt;
    wire                    bm_vld, bm_done;

    bitmap_decoder u_bm (
        .clk        (clk),       .rst_n      (rst_n),
        .bm_in      (bm_word),
        .vals_in_0  (bm_val_0),  .vals_in_1  (bm_val_1),
        .vals_in_2  (bm_val_2),  .vals_in_3  (bm_val_3),
        .vals_in_4  (bm_val_4),  .vals_in_5  (bm_val_5),
        .vals_in_6  (bm_val_6),  .vals_in_7  (bm_val_7),
        .vals_in_8  (bm_val_8),  .vals_in_9  (bm_val_9),
        .vals_in_10 (bm_val_10), .vals_in_11 (bm_val_11),
        .vals_in_12 (bm_val_12), .vals_in_13 (bm_val_13),
        .vals_in_14 (bm_val_14), .vals_in_15 (bm_val_15),
        .load       (bm_load),
        .addr_out   (bm_addr),   .weight_out (bm_wt),
        .valid_out  (bm_vld),    .done       (bm_done)
    );

    // ── Output MUX ─────────────────────────────────────────────────────────
    assign addr_out   = use_bm_r ? bm_addr  : coo_addr;
    assign weight_out = use_bm_r ? bm_wt    : coo_wt;
    assign valid_out  = use_bm_r ? bm_vld   : coo_vld;
    assign done       = use_bm_r ? bm_done  : coo_done;

endmodule
