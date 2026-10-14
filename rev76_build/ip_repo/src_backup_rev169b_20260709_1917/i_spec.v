// i_spec.v  --  I-SPEC: Input-channel Sparse Processing Element Cluster
// Paper: "I-SPEC" Fig. 3a  (CLAUDE.md §8.12)
//
// Datapath per MM-tile cycle:
//   G=4 parallel weight_decoders (one per group) -> each emits (addr_g, wt_g, valid_g)
//   All G decoders decode THE SAME weight tile (same packed word), synchronized
//   addr_g selects act[g][addr_g] from depth_shuffle for group g
//   psum[g][p] += wt_g * act[g][addr_g]  for all p (broadcast within group)
//
// Why G identical decoders on same data:
//   In input-channel-sparse (poc) mode, the G groups share the SAME non-zero
//   weight positions (same addr), but each group's weight VALUE may differ
//   (output-channel parallelism: group g handles output channels poc_cnt*poc+g).
//   One weight_decoder drives addr; G separate weight value registers hold wt_g.
//
// Interface kept flat (Verilog 2001 style, no packed arrays).
// DMA/AXI wiring added in Phase 3 top-level.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module i_spec (
    input  wire        clk,
    input  wire        rst_n,

    // ── Config (held stable during computation) ──────────────────────────────
    input  wire [3:0]  n_cfg,       // NZ per M-block (1..16)
    input  wire [1:0]  poc,         // output channel parallelism (0=1,1=2,2=4)
    input  wire        apply_relu,  // unused here; passed through for top-level

    // ── Weight stream (one M-tile burst): G groups share same packed word ────
    // COO indices (same for all groups; values differ per group)
    input  wire [3:0]  wt_idx_0,  input  wire [3:0]  wt_idx_1,
    input  wire [3:0]  wt_idx_2,  input  wire [3:0]  wt_idx_3,
    input  wire [3:0]  wt_idx_4,  input  wire [3:0]  wt_idx_5,
    input  wire [3:0]  wt_idx_6,  input  wire [3:0]  wt_idx_7,
    input  wire [3:0]  wt_idx_8,  input  wire [3:0]  wt_idx_9,
    input  wire [3:0]  wt_idx_10, input  wire [3:0]  wt_idx_11,
    input  wire [3:0]  wt_idx_12, input  wire [3:0]  wt_idx_13,
    input  wire [3:0]  wt_idx_14, input  wire [3:0]  wt_idx_15,

    // Weight values for group 0
    input  wire signed [`DATA_W-1:0] wt_val_g0_0,  input  wire signed [`DATA_W-1:0] wt_val_g0_1,
    input  wire signed [`DATA_W-1:0] wt_val_g0_2,  input  wire signed [`DATA_W-1:0] wt_val_g0_3,
    input  wire signed [`DATA_W-1:0] wt_val_g0_4,  input  wire signed [`DATA_W-1:0] wt_val_g0_5,
    input  wire signed [`DATA_W-1:0] wt_val_g0_6,  input  wire signed [`DATA_W-1:0] wt_val_g0_7,
    input  wire signed [`DATA_W-1:0] wt_val_g0_8,  input  wire signed [`DATA_W-1:0] wt_val_g0_9,
    input  wire signed [`DATA_W-1:0] wt_val_g0_10, input  wire signed [`DATA_W-1:0] wt_val_g0_11,
    input  wire signed [`DATA_W-1:0] wt_val_g0_12, input  wire signed [`DATA_W-1:0] wt_val_g0_13,
    input  wire signed [`DATA_W-1:0] wt_val_g0_14, input  wire signed [`DATA_W-1:0] wt_val_g0_15,

    // Weight values for group 1
    input  wire signed [`DATA_W-1:0] wt_val_g1_0,  input  wire signed [`DATA_W-1:0] wt_val_g1_1,
    input  wire signed [`DATA_W-1:0] wt_val_g1_2,  input  wire signed [`DATA_W-1:0] wt_val_g1_3,
    input  wire signed [`DATA_W-1:0] wt_val_g1_4,  input  wire signed [`DATA_W-1:0] wt_val_g1_5,
    input  wire signed [`DATA_W-1:0] wt_val_g1_6,  input  wire signed [`DATA_W-1:0] wt_val_g1_7,
    input  wire signed [`DATA_W-1:0] wt_val_g1_8,  input  wire signed [`DATA_W-1:0] wt_val_g1_9,
    input  wire signed [`DATA_W-1:0] wt_val_g1_10, input  wire signed [`DATA_W-1:0] wt_val_g1_11,
    input  wire signed [`DATA_W-1:0] wt_val_g1_12, input  wire signed [`DATA_W-1:0] wt_val_g1_13,
    input  wire signed [`DATA_W-1:0] wt_val_g1_14, input  wire signed [`DATA_W-1:0] wt_val_g1_15,

    // Weight values for group 2
    input  wire signed [`DATA_W-1:0] wt_val_g2_0,  input  wire signed [`DATA_W-1:0] wt_val_g2_1,
    input  wire signed [`DATA_W-1:0] wt_val_g2_2,  input  wire signed [`DATA_W-1:0] wt_val_g2_3,
    input  wire signed [`DATA_W-1:0] wt_val_g2_4,  input  wire signed [`DATA_W-1:0] wt_val_g2_5,
    input  wire signed [`DATA_W-1:0] wt_val_g2_6,  input  wire signed [`DATA_W-1:0] wt_val_g2_7,
    input  wire signed [`DATA_W-1:0] wt_val_g2_8,  input  wire signed [`DATA_W-1:0] wt_val_g2_9,
    input  wire signed [`DATA_W-1:0] wt_val_g2_10, input  wire signed [`DATA_W-1:0] wt_val_g2_11,
    input  wire signed [`DATA_W-1:0] wt_val_g2_12, input  wire signed [`DATA_W-1:0] wt_val_g2_13,
    input  wire signed [`DATA_W-1:0] wt_val_g2_14, input  wire signed [`DATA_W-1:0] wt_val_g2_15,

    // Weight values for group 3
    input  wire signed [`DATA_W-1:0] wt_val_g3_0,  input  wire signed [`DATA_W-1:0] wt_val_g3_1,
    input  wire signed [`DATA_W-1:0] wt_val_g3_2,  input  wire signed [`DATA_W-1:0] wt_val_g3_3,
    input  wire signed [`DATA_W-1:0] wt_val_g3_4,  input  wire signed [`DATA_W-1:0] wt_val_g3_5,
    input  wire signed [`DATA_W-1:0] wt_val_g3_6,  input  wire signed [`DATA_W-1:0] wt_val_g3_7,
    input  wire signed [`DATA_W-1:0] wt_val_g3_8,  input  wire signed [`DATA_W-1:0] wt_val_g3_9,
    input  wire signed [`DATA_W-1:0] wt_val_g3_10, input  wire signed [`DATA_W-1:0] wt_val_g3_11,
    input  wire signed [`DATA_W-1:0] wt_val_g3_12, input  wire signed [`DATA_W-1:0] wt_val_g3_13,
    input  wire signed [`DATA_W-1:0] wt_val_g3_14, input  wire signed [`DATA_W-1:0] wt_val_g3_15,

    // Bitmap path (all groups share same bitmap; values per group)
    input  wire [`M-1:0]              bm_word,

    input  wire signed [`DATA_W-1:0] bm_val_g0_0,  input  wire signed [`DATA_W-1:0] bm_val_g0_1,
    input  wire signed [`DATA_W-1:0] bm_val_g0_2,  input  wire signed [`DATA_W-1:0] bm_val_g0_3,
    input  wire signed [`DATA_W-1:0] bm_val_g0_4,  input  wire signed [`DATA_W-1:0] bm_val_g0_5,
    input  wire signed [`DATA_W-1:0] bm_val_g0_6,  input  wire signed [`DATA_W-1:0] bm_val_g0_7,
    input  wire signed [`DATA_W-1:0] bm_val_g0_8,  input  wire signed [`DATA_W-1:0] bm_val_g0_9,
    input  wire signed [`DATA_W-1:0] bm_val_g0_10, input  wire signed [`DATA_W-1:0] bm_val_g0_11,
    input  wire signed [`DATA_W-1:0] bm_val_g0_12, input  wire signed [`DATA_W-1:0] bm_val_g0_13,
    input  wire signed [`DATA_W-1:0] bm_val_g0_14, input  wire signed [`DATA_W-1:0] bm_val_g0_15,

    input  wire signed [`DATA_W-1:0] bm_val_g1_0,  input  wire signed [`DATA_W-1:0] bm_val_g1_1,
    input  wire signed [`DATA_W-1:0] bm_val_g1_2,  input  wire signed [`DATA_W-1:0] bm_val_g1_3,
    input  wire signed [`DATA_W-1:0] bm_val_g1_4,  input  wire signed [`DATA_W-1:0] bm_val_g1_5,
    input  wire signed [`DATA_W-1:0] bm_val_g1_6,  input  wire signed [`DATA_W-1:0] bm_val_g1_7,
    input  wire signed [`DATA_W-1:0] bm_val_g1_8,  input  wire signed [`DATA_W-1:0] bm_val_g1_9,
    input  wire signed [`DATA_W-1:0] bm_val_g1_10, input  wire signed [`DATA_W-1:0] bm_val_g1_11,
    input  wire signed [`DATA_W-1:0] bm_val_g1_12, input  wire signed [`DATA_W-1:0] bm_val_g1_13,
    input  wire signed [`DATA_W-1:0] bm_val_g1_14, input  wire signed [`DATA_W-1:0] bm_val_g1_15,

    input  wire signed [`DATA_W-1:0] bm_val_g2_0,  input  wire signed [`DATA_W-1:0] bm_val_g2_1,
    input  wire signed [`DATA_W-1:0] bm_val_g2_2,  input  wire signed [`DATA_W-1:0] bm_val_g2_3,
    input  wire signed [`DATA_W-1:0] bm_val_g2_4,  input  wire signed [`DATA_W-1:0] bm_val_g2_5,
    input  wire signed [`DATA_W-1:0] bm_val_g2_6,  input  wire signed [`DATA_W-1:0] bm_val_g2_7,
    input  wire signed [`DATA_W-1:0] bm_val_g2_8,  input  wire signed [`DATA_W-1:0] bm_val_g2_9,
    input  wire signed [`DATA_W-1:0] bm_val_g2_10, input  wire signed [`DATA_W-1:0] bm_val_g2_11,
    input  wire signed [`DATA_W-1:0] bm_val_g2_12, input  wire signed [`DATA_W-1:0] bm_val_g2_13,
    input  wire signed [`DATA_W-1:0] bm_val_g2_14, input  wire signed [`DATA_W-1:0] bm_val_g2_15,

    input  wire signed [`DATA_W-1:0] bm_val_g3_0,  input  wire signed [`DATA_W-1:0] bm_val_g3_1,
    input  wire signed [`DATA_W-1:0] bm_val_g3_2,  input  wire signed [`DATA_W-1:0] bm_val_g3_3,
    input  wire signed [`DATA_W-1:0] bm_val_g3_4,  input  wire signed [`DATA_W-1:0] bm_val_g3_5,
    input  wire signed [`DATA_W-1:0] bm_val_g3_6,  input  wire signed [`DATA_W-1:0] bm_val_g3_7,
    input  wire signed [`DATA_W-1:0] bm_val_g3_8,  input  wire signed [`DATA_W-1:0] bm_val_g3_9,
    input  wire signed [`DATA_W-1:0] bm_val_g3_10, input  wire signed [`DATA_W-1:0] bm_val_g3_11,
    input  wire signed [`DATA_W-1:0] bm_val_g3_12, input  wire signed [`DATA_W-1:0] bm_val_g3_13,
    input  wire signed [`DATA_W-1:0] bm_val_g3_14, input  wire signed [`DATA_W-1:0] bm_val_g3_15,

    input  wire        wt_load,     // pulse: load one M-tile into all G decoders
    input  wire        acc_clr,     // pulse: clear all PSUMs (MM-tile boundary)

    // ── Activation window (depth_shuffle) ───────────────────────────────────
    input  wire [1:0]  shuf_mode,   // 00=LOAD 01=SHIFT 10=HOLD
    input  wire signed [`DATA_W-1:0] new_col_0,
    input  wire signed [`DATA_W-1:0] new_col_1,
    input  wire signed [`DATA_W-1:0] new_col_2,
    input  wire signed [`DATA_W-1:0] new_col_3,

    // Full window load (G*P = 64 values)
    input  wire signed [`DATA_W-1:0] act_l_g0_p0,  input  wire signed [`DATA_W-1:0] act_l_g0_p1,
    input  wire signed [`DATA_W-1:0] act_l_g0_p2,  input  wire signed [`DATA_W-1:0] act_l_g0_p3,
    input  wire signed [`DATA_W-1:0] act_l_g0_p4,  input  wire signed [`DATA_W-1:0] act_l_g0_p5,
    input  wire signed [`DATA_W-1:0] act_l_g0_p6,  input  wire signed [`DATA_W-1:0] act_l_g0_p7,
    input  wire signed [`DATA_W-1:0] act_l_g0_p8,  input  wire signed [`DATA_W-1:0] act_l_g0_p9,
    input  wire signed [`DATA_W-1:0] act_l_g0_p10, input  wire signed [`DATA_W-1:0] act_l_g0_p11,
    input  wire signed [`DATA_W-1:0] act_l_g0_p12, input  wire signed [`DATA_W-1:0] act_l_g0_p13,
    input  wire signed [`DATA_W-1:0] act_l_g0_p14, input  wire signed [`DATA_W-1:0] act_l_g0_p15,
    input  wire signed [`DATA_W-1:0] act_l_g1_p0,  input  wire signed [`DATA_W-1:0] act_l_g1_p1,
    input  wire signed [`DATA_W-1:0] act_l_g1_p2,  input  wire signed [`DATA_W-1:0] act_l_g1_p3,
    input  wire signed [`DATA_W-1:0] act_l_g1_p4,  input  wire signed [`DATA_W-1:0] act_l_g1_p5,
    input  wire signed [`DATA_W-1:0] act_l_g1_p6,  input  wire signed [`DATA_W-1:0] act_l_g1_p7,
    input  wire signed [`DATA_W-1:0] act_l_g1_p8,  input  wire signed [`DATA_W-1:0] act_l_g1_p9,
    input  wire signed [`DATA_W-1:0] act_l_g1_p10, input  wire signed [`DATA_W-1:0] act_l_g1_p11,
    input  wire signed [`DATA_W-1:0] act_l_g1_p12, input  wire signed [`DATA_W-1:0] act_l_g1_p13,
    input  wire signed [`DATA_W-1:0] act_l_g1_p14, input  wire signed [`DATA_W-1:0] act_l_g1_p15,
    input  wire signed [`DATA_W-1:0] act_l_g2_p0,  input  wire signed [`DATA_W-1:0] act_l_g2_p1,
    input  wire signed [`DATA_W-1:0] act_l_g2_p2,  input  wire signed [`DATA_W-1:0] act_l_g2_p3,
    input  wire signed [`DATA_W-1:0] act_l_g2_p4,  input  wire signed [`DATA_W-1:0] act_l_g2_p5,
    input  wire signed [`DATA_W-1:0] act_l_g2_p6,  input  wire signed [`DATA_W-1:0] act_l_g2_p7,
    input  wire signed [`DATA_W-1:0] act_l_g2_p8,  input  wire signed [`DATA_W-1:0] act_l_g2_p9,
    input  wire signed [`DATA_W-1:0] act_l_g2_p10, input  wire signed [`DATA_W-1:0] act_l_g2_p11,
    input  wire signed [`DATA_W-1:0] act_l_g2_p12, input  wire signed [`DATA_W-1:0] act_l_g2_p13,
    input  wire signed [`DATA_W-1:0] act_l_g2_p14, input  wire signed [`DATA_W-1:0] act_l_g2_p15,
    input  wire signed [`DATA_W-1:0] act_l_g3_p0,  input  wire signed [`DATA_W-1:0] act_l_g3_p1,
    input  wire signed [`DATA_W-1:0] act_l_g3_p2,  input  wire signed [`DATA_W-1:0] act_l_g3_p3,
    input  wire signed [`DATA_W-1:0] act_l_g3_p4,  input  wire signed [`DATA_W-1:0] act_l_g3_p5,
    input  wire signed [`DATA_W-1:0] act_l_g3_p6,  input  wire signed [`DATA_W-1:0] act_l_g3_p7,
    input  wire signed [`DATA_W-1:0] act_l_g3_p8,  input  wire signed [`DATA_W-1:0] act_l_g3_p9,
    input  wire signed [`DATA_W-1:0] act_l_g3_p10, input  wire signed [`DATA_W-1:0] act_l_g3_p11,
    input  wire signed [`DATA_W-1:0] act_l_g3_p12, input  wire signed [`DATA_W-1:0] act_l_g3_p13,
    input  wire signed [`DATA_W-1:0] act_l_g3_p14, input  wire signed [`DATA_W-1:0] act_l_g3_p15,

    // ── PSUM outputs (G*P = 64) ───────────────────────────────────────────────
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
    output wire signed [`ACC_W-1:0] psum_g3_p14, output wire signed [`ACC_W-1:0] psum_g3_p15,

    output wire        mac_valid    // valid from mul_group_array (2-cycle latency)
);

    // ── weight_decoder outputs per group ─────────────────────────────────────
    // Group 0 decoder: addr_g0 (index in M-block), wt_g0 (weight value)
    wire [3:0]               wd_addr_g0;
    wire signed [`DATA_W-1:0] wd_wt_g0;
    wire                     wd_valid_g0, wd_done_g0;

    // Group 1 decoder
    wire [3:0]               wd_addr_g1;
    wire signed [`DATA_W-1:0] wd_wt_g1;
    wire                     wd_valid_g1, wd_done_g1;

    // Group 2 decoder
    wire [3:0]               wd_addr_g2;
    wire signed [`DATA_W-1:0] wd_wt_g2;
    wire                     wd_valid_g2, wd_done_g2;

    // Group 3 decoder
    wire [3:0]               wd_addr_g3;
    wire signed [`DATA_W-1:0] wd_wt_g3;
    wire                     wd_valid_g3, wd_done_g3;

    // All four decoders share the same indices, loaded simultaneously.
    // mac_valid uses group0 valid (all groups synchronize because same n_cfg).
    wire wd_valid;
    assign wd_valid = wd_valid_g0;

    // ── depth_shuffle outputs (G*P = 64) ─────────────────────────────────────
    wire signed [`DATA_W-1:0]
        ds_g0p0,  ds_g0p1,  ds_g0p2,  ds_g0p3,  ds_g0p4,  ds_g0p5,  ds_g0p6,  ds_g0p7,
        ds_g0p8,  ds_g0p9,  ds_g0p10, ds_g0p11, ds_g0p12, ds_g0p13, ds_g0p14, ds_g0p15,
        ds_g1p0,  ds_g1p1,  ds_g1p2,  ds_g1p3,  ds_g1p4,  ds_g1p5,  ds_g1p6,  ds_g1p7,
        ds_g1p8,  ds_g1p9,  ds_g1p10, ds_g1p11, ds_g1p12, ds_g1p13, ds_g1p14, ds_g1p15,
        ds_g2p0,  ds_g2p1,  ds_g2p2,  ds_g2p3,  ds_g2p4,  ds_g2p5,  ds_g2p6,  ds_g2p7,
        ds_g2p8,  ds_g2p9,  ds_g2p10, ds_g2p11, ds_g2p12, ds_g2p13, ds_g2p14, ds_g2p15,
        ds_g3p0,  ds_g3p1,  ds_g3p2,  ds_g3p3,  ds_g3p4,  ds_g3p5,  ds_g3p6,  ds_g3p7,
        ds_g3p8,  ds_g3p9,  ds_g3p10, ds_g3p11, ds_g3p12, ds_g3p13, ds_g3p14, ds_g3p15;

    // ── pch_gen: compute per-group activation address ─────────────────────────
    // addr_g = poc_cnt * poc + g  (poc_cnt = NZ index from weight_decoder)
    // When poc=0: all groups use same addr (wd_addr_g0 directly)
    wire [7:0] pch_addr_0, pch_addr_1, pch_addr_2, pch_addr_3;

    pch_gen u_pch (
        .poc_cnt ({4'b0, wd_addr_g0}),
        .poc     (poc),
        .in_reg_addr_0 (pch_addr_0),
        .in_reg_addr_1 (pch_addr_1),
        .in_reg_addr_2 (pch_addr_2),
        .in_reg_addr_3 (pch_addr_3)
    );

    // ── Activation MUX: each group selects one activation from depth_shuffle ──
    // pch_addr_g selects which position p to read for group g.
    // The selected value is broadcast to all P MACs within that group.
    reg signed [`DATA_W-1:0] ds_arr_g0 [0:`P-1];
    reg signed [`DATA_W-1:0] ds_arr_g1 [0:`P-1];
    reg signed [`DATA_W-1:0] ds_arr_g2 [0:`P-1];
    reg signed [`DATA_W-1:0] ds_arr_g3 [0:`P-1];

    reg signed [`DATA_W-1:0] act_sel_g0, act_sel_g1, act_sel_g2, act_sel_g3;

    always @(*) begin
        ds_arr_g0[0]=ds_g0p0;   ds_arr_g0[1]=ds_g0p1;   ds_arr_g0[2]=ds_g0p2;   ds_arr_g0[3]=ds_g0p3;
        ds_arr_g0[4]=ds_g0p4;   ds_arr_g0[5]=ds_g0p5;   ds_arr_g0[6]=ds_g0p6;   ds_arr_g0[7]=ds_g0p7;
        ds_arr_g0[8]=ds_g0p8;   ds_arr_g0[9]=ds_g0p9;   ds_arr_g0[10]=ds_g0p10; ds_arr_g0[11]=ds_g0p11;
        ds_arr_g0[12]=ds_g0p12; ds_arr_g0[13]=ds_g0p13; ds_arr_g0[14]=ds_g0p14; ds_arr_g0[15]=ds_g0p15;

        ds_arr_g1[0]=ds_g1p0;   ds_arr_g1[1]=ds_g1p1;   ds_arr_g1[2]=ds_g1p2;   ds_arr_g1[3]=ds_g1p3;
        ds_arr_g1[4]=ds_g1p4;   ds_arr_g1[5]=ds_g1p5;   ds_arr_g1[6]=ds_g1p6;   ds_arr_g1[7]=ds_g1p7;
        ds_arr_g1[8]=ds_g1p8;   ds_arr_g1[9]=ds_g1p9;   ds_arr_g1[10]=ds_g1p10; ds_arr_g1[11]=ds_g1p11;
        ds_arr_g1[12]=ds_g1p12; ds_arr_g1[13]=ds_g1p13; ds_arr_g1[14]=ds_g1p14; ds_arr_g1[15]=ds_g1p15;

        ds_arr_g2[0]=ds_g2p0;   ds_arr_g2[1]=ds_g2p1;   ds_arr_g2[2]=ds_g2p2;   ds_arr_g2[3]=ds_g2p3;
        ds_arr_g2[4]=ds_g2p4;   ds_arr_g2[5]=ds_g2p5;   ds_arr_g2[6]=ds_g2p6;   ds_arr_g2[7]=ds_g2p7;
        ds_arr_g2[8]=ds_g2p8;   ds_arr_g2[9]=ds_g2p9;   ds_arr_g2[10]=ds_g2p10; ds_arr_g2[11]=ds_g2p11;
        ds_arr_g2[12]=ds_g2p12; ds_arr_g2[13]=ds_g2p13; ds_arr_g2[14]=ds_g2p14; ds_arr_g2[15]=ds_g2p15;

        ds_arr_g3[0]=ds_g3p0;   ds_arr_g3[1]=ds_g3p1;   ds_arr_g3[2]=ds_g3p2;   ds_arr_g3[3]=ds_g3p3;
        ds_arr_g3[4]=ds_g3p4;   ds_arr_g3[5]=ds_g3p5;   ds_arr_g3[6]=ds_g3p6;   ds_arr_g3[7]=ds_g3p7;
        ds_arr_g3[8]=ds_g3p8;   ds_arr_g3[9]=ds_g3p9;   ds_arr_g3[10]=ds_g3p10; ds_arr_g3[11]=ds_g3p11;
        ds_arr_g3[12]=ds_g3p12; ds_arr_g3[13]=ds_g3p13; ds_arr_g3[14]=ds_g3p14; ds_arr_g3[15]=ds_g3p15;

        if (poc == 2'd0) begin
            // poc=0: all groups use same NZ index directly
            act_sel_g0 = ds_arr_g0[wd_addr_g0];
            act_sel_g1 = ds_arr_g1[wd_addr_g0];
            act_sel_g2 = ds_arr_g2[wd_addr_g0];
            act_sel_g3 = ds_arr_g3[wd_addr_g0];
        end else begin
            // poc>0: each group uses its own pch_gen address
            act_sel_g0 = ds_arr_g0[pch_addr_0[3:0]];
            act_sel_g1 = ds_arr_g1[pch_addr_1[3:0]];
            act_sel_g2 = ds_arr_g2[pch_addr_2[3:0]];
            act_sel_g3 = ds_arr_g3[pch_addr_3[3:0]];
        end
    end

    // ── mul_group_array product wires (G*P = 64) ──────────────────────────────
    wire signed [`ACC_W-1:0]
        mg0p0,  mg0p1,  mg0p2,  mg0p3,  mg0p4,  mg0p5,  mg0p6,  mg0p7,
        mg0p8,  mg0p9,  mg0p10, mg0p11, mg0p12, mg0p13, mg0p14, mg0p15,
        mg1p0,  mg1p1,  mg1p2,  mg1p3,  mg1p4,  mg1p5,  mg1p6,  mg1p7,
        mg1p8,  mg1p9,  mg1p10, mg1p11, mg1p12, mg1p13, mg1p14, mg1p15,
        mg2p0,  mg2p1,  mg2p2,  mg2p3,  mg2p4,  mg2p5,  mg2p6,  mg2p7,
        mg2p8,  mg2p9,  mg2p10, mg2p11, mg2p12, mg2p13, mg2p14, mg2p15,
        mg3p0,  mg3p1,  mg3p2,  mg3p3,  mg3p4,  mg3p5,  mg3p6,  mg3p7,
        mg3p8,  mg3p9,  mg3p10, mg3p11, mg3p12, mg3p13, mg3p14, mg3p15;

    // ── Instantiations ────────────────────────────────────────────────────────
    // NOTE: act_sel_g* (pch_gen selection) is unused in mul_group_array wiring.
    // Each MAC p uses ds_g{g}p{p} directly: all P spatial positions computed in
    // parallel, each with the same weight but its own activation from depth_shuffle.
    // This provides spatial parallelism: psum[g][p] accumulates act[spatial_p][IC].

    // Group 0 weight decoder
    weight_decoder u_wd_g0 (
        .clk(clk), .rst_n(rst_n), .n_cfg(n_cfg),
        .coo_idx_0(wt_idx_0),   .coo_idx_1(wt_idx_1),   .coo_idx_2(wt_idx_2),   .coo_idx_3(wt_idx_3),
        .coo_idx_4(wt_idx_4),   .coo_idx_5(wt_idx_5),   .coo_idx_6(wt_idx_6),   .coo_idx_7(wt_idx_7),
        .coo_idx_8(wt_idx_8),   .coo_idx_9(wt_idx_9),   .coo_idx_10(wt_idx_10), .coo_idx_11(wt_idx_11),
        .coo_idx_12(wt_idx_12), .coo_idx_13(wt_idx_13), .coo_idx_14(wt_idx_14), .coo_idx_15(wt_idx_15),
        .coo_val_0(wt_val_g0_0),   .coo_val_1(wt_val_g0_1),   .coo_val_2(wt_val_g0_2),   .coo_val_3(wt_val_g0_3),
        .coo_val_4(wt_val_g0_4),   .coo_val_5(wt_val_g0_5),   .coo_val_6(wt_val_g0_6),   .coo_val_7(wt_val_g0_7),
        .coo_val_8(wt_val_g0_8),   .coo_val_9(wt_val_g0_9),   .coo_val_10(wt_val_g0_10), .coo_val_11(wt_val_g0_11),
        .coo_val_12(wt_val_g0_12), .coo_val_13(wt_val_g0_13), .coo_val_14(wt_val_g0_14), .coo_val_15(wt_val_g0_15),
        .bm_word(bm_word),
        .bm_val_0(bm_val_g0_0),   .bm_val_1(bm_val_g0_1),   .bm_val_2(bm_val_g0_2),   .bm_val_3(bm_val_g0_3),
        .bm_val_4(bm_val_g0_4),   .bm_val_5(bm_val_g0_5),   .bm_val_6(bm_val_g0_6),   .bm_val_7(bm_val_g0_7),
        .bm_val_8(bm_val_g0_8),   .bm_val_9(bm_val_g0_9),   .bm_val_10(bm_val_g0_10), .bm_val_11(bm_val_g0_11),
        .bm_val_12(bm_val_g0_12), .bm_val_13(bm_val_g0_13), .bm_val_14(bm_val_g0_14), .bm_val_15(bm_val_g0_15),
        .load(wt_load),
        .addr_out(wd_addr_g0), .weight_out(wd_wt_g0),
        .valid_out(wd_valid_g0), .done(wd_done_g0)
    );

    // Group 1 weight decoder (same indices, different values)
    weight_decoder u_wd_g1 (
        .clk(clk), .rst_n(rst_n), .n_cfg(n_cfg),
        .coo_idx_0(wt_idx_0),   .coo_idx_1(wt_idx_1),   .coo_idx_2(wt_idx_2),   .coo_idx_3(wt_idx_3),
        .coo_idx_4(wt_idx_4),   .coo_idx_5(wt_idx_5),   .coo_idx_6(wt_idx_6),   .coo_idx_7(wt_idx_7),
        .coo_idx_8(wt_idx_8),   .coo_idx_9(wt_idx_9),   .coo_idx_10(wt_idx_10), .coo_idx_11(wt_idx_11),
        .coo_idx_12(wt_idx_12), .coo_idx_13(wt_idx_13), .coo_idx_14(wt_idx_14), .coo_idx_15(wt_idx_15),
        .coo_val_0(wt_val_g1_0),   .coo_val_1(wt_val_g1_1),   .coo_val_2(wt_val_g1_2),   .coo_val_3(wt_val_g1_3),
        .coo_val_4(wt_val_g1_4),   .coo_val_5(wt_val_g1_5),   .coo_val_6(wt_val_g1_6),   .coo_val_7(wt_val_g1_7),
        .coo_val_8(wt_val_g1_8),   .coo_val_9(wt_val_g1_9),   .coo_val_10(wt_val_g1_10), .coo_val_11(wt_val_g1_11),
        .coo_val_12(wt_val_g1_12), .coo_val_13(wt_val_g1_13), .coo_val_14(wt_val_g1_14), .coo_val_15(wt_val_g1_15),
        .bm_word(bm_word),
        .bm_val_0(bm_val_g1_0),   .bm_val_1(bm_val_g1_1),   .bm_val_2(bm_val_g1_2),   .bm_val_3(bm_val_g1_3),
        .bm_val_4(bm_val_g1_4),   .bm_val_5(bm_val_g1_5),   .bm_val_6(bm_val_g1_6),   .bm_val_7(bm_val_g1_7),
        .bm_val_8(bm_val_g1_8),   .bm_val_9(bm_val_g1_9),   .bm_val_10(bm_val_g1_10), .bm_val_11(bm_val_g1_11),
        .bm_val_12(bm_val_g1_12), .bm_val_13(bm_val_g1_13), .bm_val_14(bm_val_g1_14), .bm_val_15(bm_val_g1_15),
        .load(wt_load),
        .addr_out(wd_addr_g1), .weight_out(wd_wt_g1),
        .valid_out(wd_valid_g1), .done(wd_done_g1)
    );

    // Group 2 weight decoder
    weight_decoder u_wd_g2 (
        .clk(clk), .rst_n(rst_n), .n_cfg(n_cfg),
        .coo_idx_0(wt_idx_0),   .coo_idx_1(wt_idx_1),   .coo_idx_2(wt_idx_2),   .coo_idx_3(wt_idx_3),
        .coo_idx_4(wt_idx_4),   .coo_idx_5(wt_idx_5),   .coo_idx_6(wt_idx_6),   .coo_idx_7(wt_idx_7),
        .coo_idx_8(wt_idx_8),   .coo_idx_9(wt_idx_9),   .coo_idx_10(wt_idx_10), .coo_idx_11(wt_idx_11),
        .coo_idx_12(wt_idx_12), .coo_idx_13(wt_idx_13), .coo_idx_14(wt_idx_14), .coo_idx_15(wt_idx_15),
        .coo_val_0(wt_val_g2_0),   .coo_val_1(wt_val_g2_1),   .coo_val_2(wt_val_g2_2),   .coo_val_3(wt_val_g2_3),
        .coo_val_4(wt_val_g2_4),   .coo_val_5(wt_val_g2_5),   .coo_val_6(wt_val_g2_6),   .coo_val_7(wt_val_g2_7),
        .coo_val_8(wt_val_g2_8),   .coo_val_9(wt_val_g2_9),   .coo_val_10(wt_val_g2_10), .coo_val_11(wt_val_g2_11),
        .coo_val_12(wt_val_g2_12), .coo_val_13(wt_val_g2_13), .coo_val_14(wt_val_g2_14), .coo_val_15(wt_val_g2_15),
        .bm_word(bm_word),
        .bm_val_0(bm_val_g2_0),   .bm_val_1(bm_val_g2_1),   .bm_val_2(bm_val_g2_2),   .bm_val_3(bm_val_g2_3),
        .bm_val_4(bm_val_g2_4),   .bm_val_5(bm_val_g2_5),   .bm_val_6(bm_val_g2_6),   .bm_val_7(bm_val_g2_7),
        .bm_val_8(bm_val_g2_8),   .bm_val_9(bm_val_g2_9),   .bm_val_10(bm_val_g2_10), .bm_val_11(bm_val_g2_11),
        .bm_val_12(bm_val_g2_12), .bm_val_13(bm_val_g2_13), .bm_val_14(bm_val_g2_14), .bm_val_15(bm_val_g2_15),
        .load(wt_load),
        .addr_out(wd_addr_g2), .weight_out(wd_wt_g2),
        .valid_out(wd_valid_g2), .done(wd_done_g2)
    );

    // Group 3 weight decoder
    weight_decoder u_wd_g3 (
        .clk(clk), .rst_n(rst_n), .n_cfg(n_cfg),
        .coo_idx_0(wt_idx_0),   .coo_idx_1(wt_idx_1),   .coo_idx_2(wt_idx_2),   .coo_idx_3(wt_idx_3),
        .coo_idx_4(wt_idx_4),   .coo_idx_5(wt_idx_5),   .coo_idx_6(wt_idx_6),   .coo_idx_7(wt_idx_7),
        .coo_idx_8(wt_idx_8),   .coo_idx_9(wt_idx_9),   .coo_idx_10(wt_idx_10), .coo_idx_11(wt_idx_11),
        .coo_idx_12(wt_idx_12), .coo_idx_13(wt_idx_13), .coo_idx_14(wt_idx_14), .coo_idx_15(wt_idx_15),
        .coo_val_0(wt_val_g3_0),   .coo_val_1(wt_val_g3_1),   .coo_val_2(wt_val_g3_2),   .coo_val_3(wt_val_g3_3),
        .coo_val_4(wt_val_g3_4),   .coo_val_5(wt_val_g3_5),   .coo_val_6(wt_val_g3_6),   .coo_val_7(wt_val_g3_7),
        .coo_val_8(wt_val_g3_8),   .coo_val_9(wt_val_g3_9),   .coo_val_10(wt_val_g3_10), .coo_val_11(wt_val_g3_11),
        .coo_val_12(wt_val_g3_12), .coo_val_13(wt_val_g3_13), .coo_val_14(wt_val_g3_14), .coo_val_15(wt_val_g3_15),
        .bm_word(bm_word),
        .bm_val_0(bm_val_g3_0),   .bm_val_1(bm_val_g3_1),   .bm_val_2(bm_val_g3_2),   .bm_val_3(bm_val_g3_3),
        .bm_val_4(bm_val_g3_4),   .bm_val_5(bm_val_g3_5),   .bm_val_6(bm_val_g3_6),   .bm_val_7(bm_val_g3_7),
        .bm_val_8(bm_val_g3_8),   .bm_val_9(bm_val_g3_9),   .bm_val_10(bm_val_g3_10), .bm_val_11(bm_val_g3_11),
        .bm_val_12(bm_val_g3_12), .bm_val_13(bm_val_g3_13), .bm_val_14(bm_val_g3_14), .bm_val_15(bm_val_g3_15),
        .load(wt_load),
        .addr_out(wd_addr_g3), .weight_out(wd_wt_g3),
        .valid_out(wd_valid_g3), .done(wd_done_g3)
    );

    // depth_shuffle: G*P activation window
    depth_shuffle u_ds (
        .clk(clk), .rst_n(rst_n), .shuffle_mode(shuf_mode),
        .new_col_0(new_col_0), .new_col_1(new_col_1),
        .new_col_2(new_col_2), .new_col_3(new_col_3),
        .load_g0_p0(act_l_g0_p0),   .load_g0_p1(act_l_g0_p1),   .load_g0_p2(act_l_g0_p2),   .load_g0_p3(act_l_g0_p3),
        .load_g0_p4(act_l_g0_p4),   .load_g0_p5(act_l_g0_p5),   .load_g0_p6(act_l_g0_p6),   .load_g0_p7(act_l_g0_p7),
        .load_g0_p8(act_l_g0_p8),   .load_g0_p9(act_l_g0_p9),   .load_g0_p10(act_l_g0_p10), .load_g0_p11(act_l_g0_p11),
        .load_g0_p12(act_l_g0_p12), .load_g0_p13(act_l_g0_p13), .load_g0_p14(act_l_g0_p14), .load_g0_p15(act_l_g0_p15),
        .load_g1_p0(act_l_g1_p0),   .load_g1_p1(act_l_g1_p1),   .load_g1_p2(act_l_g1_p2),   .load_g1_p3(act_l_g1_p3),
        .load_g1_p4(act_l_g1_p4),   .load_g1_p5(act_l_g1_p5),   .load_g1_p6(act_l_g1_p6),   .load_g1_p7(act_l_g1_p7),
        .load_g1_p8(act_l_g1_p8),   .load_g1_p9(act_l_g1_p9),   .load_g1_p10(act_l_g1_p10), .load_g1_p11(act_l_g1_p11),
        .load_g1_p12(act_l_g1_p12), .load_g1_p13(act_l_g1_p13), .load_g1_p14(act_l_g1_p14), .load_g1_p15(act_l_g1_p15),
        .load_g2_p0(act_l_g2_p0),   .load_g2_p1(act_l_g2_p1),   .load_g2_p2(act_l_g2_p2),   .load_g2_p3(act_l_g2_p3),
        .load_g2_p4(act_l_g2_p4),   .load_g2_p5(act_l_g2_p5),   .load_g2_p6(act_l_g2_p6),   .load_g2_p7(act_l_g2_p7),
        .load_g2_p8(act_l_g2_p8),   .load_g2_p9(act_l_g2_p9),   .load_g2_p10(act_l_g2_p10), .load_g2_p11(act_l_g2_p11),
        .load_g2_p12(act_l_g2_p12), .load_g2_p13(act_l_g2_p13), .load_g2_p14(act_l_g2_p14), .load_g2_p15(act_l_g2_p15),
        .load_g3_p0(act_l_g3_p0),   .load_g3_p1(act_l_g3_p1),   .load_g3_p2(act_l_g3_p2),   .load_g3_p3(act_l_g3_p3),
        .load_g3_p4(act_l_g3_p4),   .load_g3_p5(act_l_g3_p5),   .load_g3_p6(act_l_g3_p6),   .load_g3_p7(act_l_g3_p7),
        .load_g3_p8(act_l_g3_p8),   .load_g3_p9(act_l_g3_p9),   .load_g3_p10(act_l_g3_p10), .load_g3_p11(act_l_g3_p11),
        .load_g3_p12(act_l_g3_p12), .load_g3_p13(act_l_g3_p13), .load_g3_p14(act_l_g3_p14), .load_g3_p15(act_l_g3_p15),
        .act_g0_p0(ds_g0p0),   .act_g0_p1(ds_g0p1),   .act_g0_p2(ds_g0p2),   .act_g0_p3(ds_g0p3),
        .act_g0_p4(ds_g0p4),   .act_g0_p5(ds_g0p5),   .act_g0_p6(ds_g0p6),   .act_g0_p7(ds_g0p7),
        .act_g0_p8(ds_g0p8),   .act_g0_p9(ds_g0p9),   .act_g0_p10(ds_g0p10), .act_g0_p11(ds_g0p11),
        .act_g0_p12(ds_g0p12), .act_g0_p13(ds_g0p13), .act_g0_p14(ds_g0p14), .act_g0_p15(ds_g0p15),
        .act_g1_p0(ds_g1p0),   .act_g1_p1(ds_g1p1),   .act_g1_p2(ds_g1p2),   .act_g1_p3(ds_g1p3),
        .act_g1_p4(ds_g1p4),   .act_g1_p5(ds_g1p5),   .act_g1_p6(ds_g1p6),   .act_g1_p7(ds_g1p7),
        .act_g1_p8(ds_g1p8),   .act_g1_p9(ds_g1p9),   .act_g1_p10(ds_g1p10), .act_g1_p11(ds_g1p11),
        .act_g1_p12(ds_g1p12), .act_g1_p13(ds_g1p13), .act_g1_p14(ds_g1p14), .act_g1_p15(ds_g1p15),
        .act_g2_p0(ds_g2p0),   .act_g2_p1(ds_g2p1),   .act_g2_p2(ds_g2p2),   .act_g2_p3(ds_g2p3),
        .act_g2_p4(ds_g2p4),   .act_g2_p5(ds_g2p5),   .act_g2_p6(ds_g2p6),   .act_g2_p7(ds_g2p7),
        .act_g2_p8(ds_g2p8),   .act_g2_p9(ds_g2p9),   .act_g2_p10(ds_g2p10), .act_g2_p11(ds_g2p11),
        .act_g2_p12(ds_g2p12), .act_g2_p13(ds_g2p13), .act_g2_p14(ds_g2p14), .act_g2_p15(ds_g2p15),
        .act_g3_p0(ds_g3p0),   .act_g3_p1(ds_g3p1),   .act_g3_p2(ds_g3p2),   .act_g3_p3(ds_g3p3),
        .act_g3_p4(ds_g3p4),   .act_g3_p5(ds_g3p5),   .act_g3_p6(ds_g3p6),   .act_g3_p7(ds_g3p7),
        .act_g3_p8(ds_g3p8),   .act_g3_p9(ds_g3p9),   .act_g3_p10(ds_g3p10), .act_g3_p11(ds_g3p11),
        .act_g3_p12(ds_g3p12), .act_g3_p13(ds_g3p13), .act_g3_p14(ds_g3p14), .act_g3_p15(ds_g3p15)
    );

    // mul_group_array: G=4 groups, each with P=16 MACs.
    // All P MACs in group g share same weight (wt_g) but each uses its OWN
    // activation ds_g{g}p{p} from depth_shuffle (spatial parallelism).
    // depth_shuffle[g][p] = act[spatial_position_p][current_IC_block_g].
    // psum[g][p] += wt_g * ds_g{g}p{p}  for each p independently.
    mul_group_array u_mga (
        .clk(clk), .rst_n(rst_n), .valid_in(wd_valid),
        .wt_g0(wd_wt_g0), .wt_g1(wd_wt_g1), .wt_g2(wd_wt_g2), .wt_g3(wd_wt_g3),
        .act_g0_p0(ds_g0p0),   .act_g0_p1(ds_g0p1),   .act_g0_p2(ds_g0p2),   .act_g0_p3(ds_g0p3),
        .act_g0_p4(ds_g0p4),   .act_g0_p5(ds_g0p5),   .act_g0_p6(ds_g0p6),   .act_g0_p7(ds_g0p7),
        .act_g0_p8(ds_g0p8),   .act_g0_p9(ds_g0p9),   .act_g0_p10(ds_g0p10), .act_g0_p11(ds_g0p11),
        .act_g0_p12(ds_g0p12), .act_g0_p13(ds_g0p13), .act_g0_p14(ds_g0p14), .act_g0_p15(ds_g0p15),
        .act_g1_p0(ds_g1p0),   .act_g1_p1(ds_g1p1),   .act_g1_p2(ds_g1p2),   .act_g1_p3(ds_g1p3),
        .act_g1_p4(ds_g1p4),   .act_g1_p5(ds_g1p5),   .act_g1_p6(ds_g1p6),   .act_g1_p7(ds_g1p7),
        .act_g1_p8(ds_g1p8),   .act_g1_p9(ds_g1p9),   .act_g1_p10(ds_g1p10), .act_g1_p11(ds_g1p11),
        .act_g1_p12(ds_g1p12), .act_g1_p13(ds_g1p13), .act_g1_p14(ds_g1p14), .act_g1_p15(ds_g1p15),
        .act_g2_p0(ds_g2p0),   .act_g2_p1(ds_g2p1),   .act_g2_p2(ds_g2p2),   .act_g2_p3(ds_g2p3),
        .act_g2_p4(ds_g2p4),   .act_g2_p5(ds_g2p5),   .act_g2_p6(ds_g2p6),   .act_g2_p7(ds_g2p7),
        .act_g2_p8(ds_g2p8),   .act_g2_p9(ds_g2p9),   .act_g2_p10(ds_g2p10), .act_g2_p11(ds_g2p11),
        .act_g2_p12(ds_g2p12), .act_g2_p13(ds_g2p13), .act_g2_p14(ds_g2p14), .act_g2_p15(ds_g2p15),
        .act_g3_p0(ds_g3p0),   .act_g3_p1(ds_g3p1),   .act_g3_p2(ds_g3p2),   .act_g3_p3(ds_g3p3),
        .act_g3_p4(ds_g3p4),   .act_g3_p5(ds_g3p5),   .act_g3_p6(ds_g3p6),   .act_g3_p7(ds_g3p7),
        .act_g3_p8(ds_g3p8),   .act_g3_p9(ds_g3p9),   .act_g3_p10(ds_g3p10), .act_g3_p11(ds_g3p11),
        .act_g3_p12(ds_g3p12), .act_g3_p13(ds_g3p13), .act_g3_p14(ds_g3p14), .act_g3_p15(ds_g3p15),
        .prod_g0_p0(mg0p0),   .prod_g0_p1(mg0p1),   .prod_g0_p2(mg0p2),   .prod_g0_p3(mg0p3),
        .prod_g0_p4(mg0p4),   .prod_g0_p5(mg0p5),   .prod_g0_p6(mg0p6),   .prod_g0_p7(mg0p7),
        .prod_g0_p8(mg0p8),   .prod_g0_p9(mg0p9),   .prod_g0_p10(mg0p10), .prod_g0_p11(mg0p11),
        .prod_g0_p12(mg0p12), .prod_g0_p13(mg0p13), .prod_g0_p14(mg0p14), .prod_g0_p15(mg0p15),
        .prod_g1_p0(mg1p0),   .prod_g1_p1(mg1p1),   .prod_g1_p2(mg1p2),   .prod_g1_p3(mg1p3),
        .prod_g1_p4(mg1p4),   .prod_g1_p5(mg1p5),   .prod_g1_p6(mg1p6),   .prod_g1_p7(mg1p7),
        .prod_g1_p8(mg1p8),   .prod_g1_p9(mg1p9),   .prod_g1_p10(mg1p10), .prod_g1_p11(mg1p11),
        .prod_g1_p12(mg1p12), .prod_g1_p13(mg1p13), .prod_g1_p14(mg1p14), .prod_g1_p15(mg1p15),
        .prod_g2_p0(mg2p0),   .prod_g2_p1(mg2p1),   .prod_g2_p2(mg2p2),   .prod_g2_p3(mg2p3),
        .prod_g2_p4(mg2p4),   .prod_g2_p5(mg2p5),   .prod_g2_p6(mg2p6),   .prod_g2_p7(mg2p7),
        .prod_g2_p8(mg2p8),   .prod_g2_p9(mg2p9),   .prod_g2_p10(mg2p10), .prod_g2_p11(mg2p11),
        .prod_g2_p12(mg2p12), .prod_g2_p13(mg2p13), .prod_g2_p14(mg2p14), .prod_g2_p15(mg2p15),
        .prod_g3_p0(mg3p0),   .prod_g3_p1(mg3p1),   .prod_g3_p2(mg3p2),   .prod_g3_p3(mg3p3),
        .prod_g3_p4(mg3p4),   .prod_g3_p5(mg3p5),   .prod_g3_p6(mg3p6),   .prod_g3_p7(mg3p7),
        .prod_g3_p8(mg3p8),   .prod_g3_p9(mg3p9),   .prod_g3_p10(mg3p10), .prod_g3_p11(mg3p11),
        .prod_g3_p12(mg3p12), .prod_g3_p13(mg3p13), .prod_g3_p14(mg3p14), .prod_g3_p15(mg3p15),
        .valid_out(mac_valid)
    );

    // psum_array: accumulates G*P products
    psum_array u_psa (
        .clk(clk), .rst_n(rst_n), .en(mac_valid), .acc_clr(acc_clr),
        .prod_g0_p0(mg0p0),   .prod_g0_p1(mg0p1),   .prod_g0_p2(mg0p2),   .prod_g0_p3(mg0p3),
        .prod_g0_p4(mg0p4),   .prod_g0_p5(mg0p5),   .prod_g0_p6(mg0p6),   .prod_g0_p7(mg0p7),
        .prod_g0_p8(mg0p8),   .prod_g0_p9(mg0p9),   .prod_g0_p10(mg0p10), .prod_g0_p11(mg0p11),
        .prod_g0_p12(mg0p12), .prod_g0_p13(mg0p13), .prod_g0_p14(mg0p14), .prod_g0_p15(mg0p15),
        .prod_g1_p0(mg1p0),   .prod_g1_p1(mg1p1),   .prod_g1_p2(mg1p2),   .prod_g1_p3(mg1p3),
        .prod_g1_p4(mg1p4),   .prod_g1_p5(mg1p5),   .prod_g1_p6(mg1p6),   .prod_g1_p7(mg1p7),
        .prod_g1_p8(mg1p8),   .prod_g1_p9(mg1p9),   .prod_g1_p10(mg1p10), .prod_g1_p11(mg1p11),
        .prod_g1_p12(mg1p12), .prod_g1_p13(mg1p13), .prod_g1_p14(mg1p14), .prod_g1_p15(mg1p15),
        .prod_g2_p0(mg2p0),   .prod_g2_p1(mg2p1),   .prod_g2_p2(mg2p2),   .prod_g2_p3(mg2p3),
        .prod_g2_p4(mg2p4),   .prod_g2_p5(mg2p5),   .prod_g2_p6(mg2p6),   .prod_g2_p7(mg2p7),
        .prod_g2_p8(mg2p8),   .prod_g2_p9(mg2p9),   .prod_g2_p10(mg2p10), .prod_g2_p11(mg2p11),
        .prod_g2_p12(mg2p12), .prod_g2_p13(mg2p13), .prod_g2_p14(mg2p14), .prod_g2_p15(mg2p15),
        .prod_g3_p0(mg3p0),   .prod_g3_p1(mg3p1),   .prod_g3_p2(mg3p2),   .prod_g3_p3(mg3p3),
        .prod_g3_p4(mg3p4),   .prod_g3_p5(mg3p5),   .prod_g3_p6(mg3p6),   .prod_g3_p7(mg3p7),
        .prod_g3_p8(mg3p8),   .prod_g3_p9(mg3p9),   .prod_g3_p10(mg3p10), .prod_g3_p11(mg3p11),
        .prod_g3_p12(mg3p12), .prod_g3_p13(mg3p13), .prod_g3_p14(mg3p14), .prod_g3_p15(mg3p15),
        .psum_g0_p0(psum_g0_p0),   .psum_g0_p1(psum_g0_p1),   .psum_g0_p2(psum_g0_p2),   .psum_g0_p3(psum_g0_p3),
        .psum_g0_p4(psum_g0_p4),   .psum_g0_p5(psum_g0_p5),   .psum_g0_p6(psum_g0_p6),   .psum_g0_p7(psum_g0_p7),
        .psum_g0_p8(psum_g0_p8),   .psum_g0_p9(psum_g0_p9),   .psum_g0_p10(psum_g0_p10), .psum_g0_p11(psum_g0_p11),
        .psum_g0_p12(psum_g0_p12), .psum_g0_p13(psum_g0_p13), .psum_g0_p14(psum_g0_p14), .psum_g0_p15(psum_g0_p15),
        .psum_g1_p0(psum_g1_p0),   .psum_g1_p1(psum_g1_p1),   .psum_g1_p2(psum_g1_p2),   .psum_g1_p3(psum_g1_p3),
        .psum_g1_p4(psum_g1_p4),   .psum_g1_p5(psum_g1_p5),   .psum_g1_p6(psum_g1_p6),   .psum_g1_p7(psum_g1_p7),
        .psum_g1_p8(psum_g1_p8),   .psum_g1_p9(psum_g1_p9),   .psum_g1_p10(psum_g1_p10), .psum_g1_p11(psum_g1_p11),
        .psum_g1_p12(psum_g1_p12), .psum_g1_p13(psum_g1_p13), .psum_g1_p14(psum_g1_p14), .psum_g1_p15(psum_g1_p15),
        .psum_g2_p0(psum_g2_p0),   .psum_g2_p1(psum_g2_p1),   .psum_g2_p2(psum_g2_p2),   .psum_g2_p3(psum_g2_p3),
        .psum_g2_p4(psum_g2_p4),   .psum_g2_p5(psum_g2_p5),   .psum_g2_p6(psum_g2_p6),   .psum_g2_p7(psum_g2_p7),
        .psum_g2_p8(psum_g2_p8),   .psum_g2_p9(psum_g2_p9),   .psum_g2_p10(psum_g2_p10), .psum_g2_p11(psum_g2_p11),
        .psum_g2_p12(psum_g2_p12), .psum_g2_p13(psum_g2_p13), .psum_g2_p14(psum_g2_p14), .psum_g2_p15(psum_g2_p15),
        .psum_g3_p0(psum_g3_p0),   .psum_g3_p1(psum_g3_p1),   .psum_g3_p2(psum_g3_p2),   .psum_g3_p3(psum_g3_p3),
        .psum_g3_p4(psum_g3_p4),   .psum_g3_p5(psum_g3_p5),   .psum_g3_p6(psum_g3_p6),   .psum_g3_p7(psum_g3_p7),
        .psum_g3_p8(psum_g3_p8),   .psum_g3_p9(psum_g3_p9),   .psum_g3_p10(psum_g3_p10), .psum_g3_p11(psum_g3_p11),
        .psum_g3_p12(psum_g3_p12), .psum_g3_p13(psum_g3_p13), .psum_g3_p14(psum_g3_p14), .psum_g3_p15(psum_g3_p15)
    );

endmodule
