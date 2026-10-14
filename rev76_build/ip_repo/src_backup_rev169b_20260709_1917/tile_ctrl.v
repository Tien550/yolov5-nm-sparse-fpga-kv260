// tile_ctrl.v  --  Hardware tile controller: iterate spatial tiles over H×W
// Paper: tile loop (CLAUDE.md §8.14)
// Replaces ARM-side Python tiling. Purely counter-based, no FSM needed.
//
// Generates one tile descriptor per cycle when tile_next is asserted.
// tile_valid goes high the cycle after tile_next (combinational from counters).
//
// Tile iteration order: row-major (h_tile counter outer, w_tile counter inner).
// Each tile: [h_start, w_start] with halo expansion clamped to [0, Hin/Win].
//
// For conv1x1: HALO=0 (no spatial sliding), TILE_H=Hin, TILE_W=Win (single tile).
// For conv3x3/dwconv: TILE_H=28, TILE_W=28, HALO=1.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module tile_ctrl (
    input  wire        clk,
    input  wire        rst_n,

    // Config (from AXI regs, stable during layer)
    input  wire [9:0]  Hin,       // input feature map height
    input  wire [9:0]  Win,       // input feature map width
    input  wire [5:0]  tile_h,    // tile height (e.g. 28 for conv3x3, or Hin for conv1x1)
    input  wire [5:0]  tile_w,    // tile width
    input  wire [1:0]  halo,      // halo pixels (0 for conv1x1, 1 for conv3x3)
    input  wire        start,     // pulse: begin tile iteration

    // Control
    input  wire        tile_next, // pulse: advance to next tile (assert when current tile done)

    // Outputs (registered, valid one cycle after start/tile_next)
    output reg  [9:0]  tile_h_start,  // tile top-left row (no halo)
    output reg  [9:0]  tile_w_start,  // tile top-left col (no halo)
    output reg  [5:0]  tile_h_act,    // actual tile height (may be < tile_h at boundary)
    output reg  [5:0]  tile_w_act,    // actual tile width
    output reg  [1:0]  pad_top,       // zero-pad top (= halo if at top edge)
    output reg  [1:0]  pad_bot,       // zero-pad bottom
    output reg  [1:0]  pad_left,      // zero-pad left
    output reg  [1:0]  pad_right,     // zero-pad right
    output reg         tile_valid,    // high while a tile is being presented
    output reg         tile_last      // high on final tile of layer
);

    // ── Tile counters (in tile units, not pixel units) ───────────────────────
    reg [4:0] h_tile_cnt;  // max ceil(416/28) = 15 tiles → 5 bits
    reg [4:0] w_tile_cnt;

    // Compute number of tiles in each dimension (combinational, ceil division).
    // Max tiles: ceil(416/1) = 416 → need 9 bits. Use 10-bit to be safe.
    // Variable-divisor division is deep combinational logic. tile_h/tile_w/Hin/Win
    // are per-layer constants, so register the divider outputs: this lifts the
    // divider out of the combinational is_last → h_tile_cnt CE path (Rev57 WNS fix,
    // -0.124ns). h_tiles_r/w_tiles_r settle 1 cycle after config (stable before start),
    // and are only consumed in ST_OUTPUT many cycles later — behaviorally transparent.
    wire [9:0] h_tiles_c = (Hin + {4'b0, tile_h} - 10'd1) / {4'b0, tile_h};
    wire [9:0] w_tiles_c = (Win + {4'b0, tile_w} - 10'd1) / {4'b0, tile_w};
    reg  [9:0] h_tiles, w_tiles;
    always @(posedge clk) begin
        h_tiles <= h_tiles_c;
        w_tiles <= w_tiles_c;
    end

    // ── Combinational tile geometry ──────────────────────────────────────────
    wire [9:0] h_start_raw = {5'b0, h_tile_cnt} * {4'b0, tile_h};
    wire [9:0] w_start_raw = {5'b0, w_tile_cnt} * {4'b0, tile_w};

    wire [9:0] h_end_raw   = h_start_raw + {4'b0, tile_h};
    wire [9:0] w_end_raw   = w_start_raw + {4'b0, tile_w};

    // Actual end clamped to image boundary
    wire [9:0] h_end_clamp = (h_end_raw > Hin) ? Hin : h_end_raw;
    wire [9:0] w_end_clamp = (w_end_raw > Win) ? Win : w_end_raw;

    // Subtract in full 10-bit before truncating to 6-bit (bug fix: was [5:0]-[5:0])
    wire [9:0] h_act_w = h_end_clamp - h_start_raw;
    wire [9:0] w_act_w = w_end_clamp - w_start_raw;
    wire [5:0] h_act   = h_act_w[5:0];  // tile_h ≤ 63 → fits in 6 bits
    wire [5:0] w_act   = w_act_w[5:0];

    wire p_top   = (h_tile_cnt == 5'd0);
    wire p_bot   = (h_end_raw >= Hin);
    wire p_left  = (w_tile_cnt == 5'd0);
    wire p_right = (w_end_raw >= Win);

    wire [4:0] h_tile_next = (w_tile_cnt == w_tiles[4:0] - 5'd1) ? h_tile_cnt + 5'd1 : h_tile_cnt;
    wire [4:0] w_tile_next = (w_tile_cnt == w_tiles[4:0] - 5'd1) ? 5'd0 : w_tile_cnt + 5'd1;
    wire       is_last     = ({5'b0,h_tile_cnt} == h_tiles - 10'd1) && ({5'b0,w_tile_cnt} == w_tiles - 10'd1);

    // ── FSM ──────────────────────────────────────────────────────────────────
    localparam ST_IDLE    = 2'd0;  // waiting for start
    localparam ST_OUTPUT  = 2'd1;  // presenting current tile
    localparam ST_ADVANCE = 2'd2;  // advance counters on tile_next

    reg [1:0] state;

    always @(posedge clk) begin
        if (!rst_n) begin
            state       <= ST_IDLE;
            h_tile_cnt  <= 5'd0;
            w_tile_cnt  <= 5'd0;
            tile_valid  <= 1'b0;
            tile_last   <= 1'b0;
            tile_h_start <= 10'd0;
            tile_w_start <= 10'd0;
            tile_h_act   <= 6'd0;
            tile_w_act   <= 6'd0;
            pad_top  <= 2'd0; pad_bot  <= 2'd0;
            pad_left <= 2'd0; pad_right <= 2'd0;
        end else begin
            case (state)
                ST_IDLE: begin
                    if (start) begin
                        h_tile_cnt <= 5'd0;
                        w_tile_cnt <= 5'd0;
                        state      <= ST_OUTPUT;
                    end
                end

                ST_OUTPUT: begin
                    // Latch current tile geometry
                    tile_h_start <= h_start_raw;
                    tile_w_start <= w_start_raw;
                    tile_h_act   <= h_act;
                    tile_w_act   <= w_act;
                    pad_top   <= p_top   ? {1'b0, halo[0]} : 2'd0;
                    pad_bot   <= p_bot   ? {1'b0, halo[0]} : 2'd0;
                    pad_left  <= p_left  ? {1'b0, halo[0]} : 2'd0;
                    pad_right <= p_right ? {1'b0, halo[0]} : 2'd0;
                    tile_valid  <= 1'b1;
                    tile_last   <= is_last;

                    if (tile_next) begin
                        if (is_last) begin
                            tile_valid <= 1'b0;
                            tile_last  <= 1'b0;
                            state      <= ST_IDLE;
                        end else begin
                            h_tile_cnt <= h_tile_next;
                            w_tile_cnt <= w_tile_next;
                            // tile_valid stays high; new geometry latched next cycle
                        end
                    end
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
