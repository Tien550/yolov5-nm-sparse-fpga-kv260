// conv_tile_r1.v -- R1 compute core: "load tile once, sweep 1 term-set/cycle".
// Replaces the per-(ic,tap) serial-reload FSM (45 cyc/tap) with a pipelined sweep that
// feeds the 64-MAC array (4 out-ch group x 16 spatial P) EVERY cycle. One conv3x3 tile
// (4 out-ch x 16 spatial = 64 outputs) accumulates n_ic*9 terms in n_ic*9 cycles + drain,
// vs ~6480 cyc today. R1 uses ONE SPEC (64 MAC); R2 adds the other 3 SPECs (4 ic/cyc).
//
// Tile data (act_tile, wt_tile) is loaded into internal regs before `start` (in sim the
// testbench pokes them hierarchically; in integration a DMA/wide-load fills them). This
// unit-test step exposes the raw 32-bit PSUMs (pre-requant) for bit-exact checking of the
// feeder + MAC + accumulate path. Requant (rr_unit) is added in step 2.
//
// term t = ic*9 + tap ; tap = kh*3 + kw. For term (ic,kh,kw):
//   weight[oc] = wt_tile[oc][ic][tap]          (4 group weights, one per out-ch)
//   act[p]     = act_tile[ic][kh][kw + p]       (16 spatial, sliding window offset kw)
//   psum[oc][p] += weight[oc] * act[p]
`timescale 1ns/1ps
`include "cnn_defines.vh"

module conv_tile_r1 (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     start,      // 1-cycle pulse: begin sweep
    input  wire [5:0]               n_ic,       // input channels this tile (1..32; conv1x1 CIN_CHUNK=32)
    input  wire [3:0]               tap_max,    // last tap index: 8 = conv3x3 (9 taps), 0 = conv1x1 (1 tap, R3)
    input  wire                     keep_psum,  // 1 = accumulate onto existing psum (chan-tiling)
    input  wire                     no_flush,   // 1 = skip requant/flush (chan-tiling middle chunk)
    input  wire                     apply_relu, // leaky-relu enable (per layer)
    output reg                      done,       // 1-cycle pulse when sweep(+flush) complete

    // tile load write-ports (driven by conv_kernel_r1 loader; in unit tests the tb pokes
    // the tile regs directly — sim-only, coexists since loader writes only when *_we=1)
    input  wire                      ld_act_we,    // wide: write 16 bytes (act_flat[word*16 +: 16])
    input  wire [5:0]                ld_act_word,  // 0..63 (1024 bytes / 16)
    input  wire [127:0]              ld_act_d128,
    input  wire                      ld_wt_we,
    input  wire [1:0]                ld_wt_g,
    input  wire [4:0]                ld_wt_ic,
    input  wire [3:0]                ld_wt_tap,
    input  wire signed [`DATA_W-1:0] ld_wt_data,
    input  wire                      ld_rq_we,
    input  wire [1:0]                ld_rq_oc,
    input  wire [1:0]                ld_rq_field, // 0=m0, 1=shift, 2=bias
    input  wire [31:0]               ld_rq_data,

    input  wire [5:0]               rd_addr,    // {oc[1:0], p[3:0]} = oc*16 + p
    output wire signed [`ACC_W-1:0] rd_psum,    // raw 32-bit psum (debug)
    output wire signed [`DATA_W-1:0] rd_out,    // requantized int8 output
    output wire signed [`DATA_W-1:0] rd_wt      // DEBUG: loaded center weight wt_tile[oc][ic][4]
);
    // ── tile storage (poked by tb / loaded by DMA in integration) ───────────
    // act stored FLAT (act_flat[idx], idx = ic*stride + kh*18 + col; stride 54=conv3x3 / 16=conv1x1)
    // stored as 64 × 128-bit WORDS (1024B). loader writes 1 word/cycle; the sweep extracts a
    // 16-byte sliding window from ≤2 words (cheap 64:1 word-mux + barrel) — NOT 64×1024:1 byte
    // muxes (which over-utilized LUTs in the act_flat-as-byte-FF version).
    reg [127:0] act_w [0:63];                             // 1024B: conv1x1 CIN_CHUNK=32 (32×16=512), conv3x3 16ch (864)
    reg signed [`DATA_W-1:0] wt_tile  [0:3][0:31][0:8];   // [oc][ic][tap] — ic up to 32 (conv1x1 CIN_CHUNK=32)
    // per-out-ch requant params (poked by tb / loaded from WT tail in integration)
    reg        [`SCALE_W-1:0] rq_m0    [0:3];
    reg        [`SHIFT_W-1:0] rq_shift [0:3];
    reg signed [`ACC_W-1:0]   rq_bias  [0:3];
    // requantized outputs [oc][p]
    reg signed [`DATA_W-1:0]  out_tile [0:3][0:15];

    // ── tile load (from loader; synthesizable flops) ────────────────────────
    always @(posedge clk) begin
        if (ld_act_we)  act_w[ld_act_word] <= ld_act_d128;   // 1 word (16 bytes)/cycle
        if (ld_wt_we)  wt_tile [ld_wt_g][ld_wt_ic][ld_wt_tap]     <= ld_wt_data;
        if (ld_rq_we) begin
            case (ld_rq_field)
                2'd0: rq_m0   [ld_rq_oc] <= ld_rq_data[`SCALE_W-1:0];
                2'd1: rq_shift[ld_rq_oc] <= ld_rq_data[`SHIFT_W-1:0];
                2'd2: rq_bias [ld_rq_oc] <= ld_rq_data;
                default: ;
            endcase
        end
    end

    // ── sweep counters (R2: 4-SPEC = 4 input-channels per cycle) ────────────
    // Each cycle sweeps 4 ic (same tap) → 256 MACs (4 oc × 4 SPEC × 16 P), sum the 4
    // SPECs per (oc,P), accumulate. n_ic=16 (B0 pad) → 4 ic-groups × n_taps cycles.
    reg  [3:0] ic_grp;        // ic-group 0..n_grp-1 (ic_base = ic_grp*4) — up to 7 (n_ic=32)
    reg  [3:0] tap_cnt;       // 0..tap_max
    reg        sweeping;
    wire [3:0] n_grp = (n_ic + 6'd3) >> 2;            // ceil(n_ic/4) — up to 8 (n_ic=32)
    wire [5:0] ic_base = {ic_grp, 2'b00};             // ic_grp*4 — up to 28
    wire [5:0] act_stride = (tap_max == 4'd0) ? 6'd16 : 6'd54;  // act_flat stride: 1x1=16, 3x3=54
    wire       last_term = (ic_grp == n_grp - 4'd1) && (tap_cnt == tap_max);

    // tap -> (kh,kw)
    wire [1:0] kh = (tap_cnt < 4'd3) ? 2'd0 : (tap_cnt < 4'd6) ? 2'd1 : 2'd2;
    wire [1:0] kw = tap_cnt - {kh, 1'b0} - {1'b0, kh};   // tap_cnt - 3*kh

    // ── current term operands: 4 SPEC (ic_base+s), weight masked beyond n_ic ─
    wire signed [`DATA_W-1:0] cur_wt [0:3][0:3];      // [oc][spec]
    wire signed [`DATA_W-1:0] cur_act[0:3][0:15];     // [spec][P]
    genvar go, gs, gp, gi;
    generate
        for (go = 0; go < 4; go = go + 1) begin : g_wt_oc
            for (gs = 0; gs < 4; gs = gs + 1) begin : g_wt_s
                assign cur_wt[go][gs] = ((ic_base + gs[4:0]) < n_ic)
                                        ? wt_tile[go][ic_base + gs[4:0]][tap_cnt] : 8'sd0;
            end
        end
        for (gs = 0; gs < 4; gs = gs + 1) begin : g_act_s
            // flat byte base for this SPEC's window: ic*stride + kh*18 + kw (16 bytes follow)
            wire [9:0] base_s = (ic_base + gs[4:0]) * act_stride + {3'd0, kh}*10'd18 + {8'd0, kw};
            wire [5:0] word_lo = base_s[9:4];
            wire [255:0] two_w = {act_w[word_lo + 6'd1], act_w[word_lo]};
            wire [127:0] window = two_w[{base_s[3:0], 3'b000} +: 128];   // 16-byte window @ base_s
            wire        valid_s = ((ic_base + gs[4:0]) < n_ic);          // mask SPEC≥n_ic to 0
            for (gp = 0; gp < 16; gp = gp + 1) begin : g_act_p
                assign cur_act[gs][gp] = valid_s ? $signed(window[gp*8 +: 8]) : 8'sd0;
            end
        end
    endgenerate

    // ── 256-MAC array (4 SPEC) + 4-input sum + 64 accumulators ──────────────
    // mac_unit 2-cyc latency; sum4 combinational on the products; accumulate when
    // sweeping delayed 2 (en_acc). acc_clr clears before first product.
    reg sweep_d1, sweep_d2;
    always @(posedge clk) begin
        if (!rst_n) begin sweep_d1 <= 1'b0; sweep_d2 <= 1'b0; end
        else        begin sweep_d1 <= sweeping; sweep_d2 <= sweep_d1; end
    end
    wire en_acc = sweep_d2;
    reg acc_clr;

    wire signed [`ACC_W-1:0] psum_w [0:63];
    generate
        for (go = 0; go < 4; go = go + 1) begin : g_oc
            for (gp = 0; gp < 16; gp = gp + 1) begin : g_p
                wire signed [`ACC_W-1:0] pr [0:3];
                for (gs = 0; gs < 4; gs = gs + 1) begin : g_spec
                    mac_unit u_mac (
                        .clk(clk), .rst_n(rst_n),
                        .a_in(cur_wt[go][gs]), .b_in(cur_act[gs][gp]), .valid_in(sweeping),
                        .p_out(pr[gs]), .valid_out()
                    );
                end
                wire signed [`ACC_W-1:0] sum4 = pr[0] + pr[1] + pr[2] + pr[3];
                accumulator u_acc (
                    .clk(clk), .rst_n(rst_n),
                    .data_in(sum4), .en(en_acc), .clr(acc_clr), .psum(psum_w[go*16 + gp])
                );
            end
        end
    endgenerate

    assign rd_psum = psum_w[rd_addr];
    assign rd_out  = out_tile[rd_addr[5:4]][rd_addr[3:0]];
    assign rd_wt   = wt_tile[rd_addr[5:4]][rd_addr[3:0]][4];   // center tap (kh1,kw1)

    // ── requant flush: 4 rr_units (one per out-ch group), 16 spatial sequential ─
    // present psum[oc][flush_p] + bias[oc] each cycle; rr_unit 3-stage → collect.
    reg        flushing;
    reg  [4:0] flush_p;        // 0..16 (present p=0..15, then idle)
    reg  [4:0] wp;             // write pointer 0..15 (collect order)
    wire       fl_valid_in = flushing && (flush_p < 5'd16);
    wire signed [`DATA_W-1:0] rr_out  [0:3];
    wire                      rr_vld  [0:3];
    generate
        for (gi = 0; gi < 4; gi = gi + 1) begin : g_rr
            wire signed [`ACC_W-1:0] biased = psum_w[gi*16 + flush_p[3:0]] + rq_bias[gi];
            rr_unit u_rr (
                .clk(clk), .rst_n(rst_n),
                .psum_in(biased), .rq_scale(rq_m0[gi]), .rq_shift(rq_shift[gi]),
                .apply_relu(apply_relu), .valid_in(fl_valid_in),
                .data_out(rr_out[gi]), .valid_out(rr_vld[gi])
            );
        end
    endgenerate

    // ── control FSM ─────────────────────────────────────────────────────────
    localparam S_IDLE = 3'd0, S_CLR = 3'd1, S_SWEEP = 3'd2, S_DRAIN = 3'd3, S_FLUSH = 3'd4;
    reg [2:0] state;
    reg [2:0] drain_cnt;

    always @(posedge clk) begin
        if (!rst_n) begin
            state    <= S_IDLE; sweeping <= 1'b0; acc_clr <= 1'b0; done <= 1'b0;
            ic_grp   <= 3'd0;   tap_cnt  <= 4'd0;  drain_cnt <= 3'd0;
            flushing <= 1'b0;   flush_p  <= 5'd0;  wp <= 5'd0;
        end else begin
            done    <= 1'b0;
            acc_clr <= 1'b0;
            case (state)
                S_IDLE: begin
                    sweeping <= 1'b0;
                    if (start) begin
                        ic_grp  <= 3'd0; tap_cnt <= 4'd0;
                        acc_clr <= ~keep_psum;   // clear psum (1 cyc) before products arrive
                        state   <= S_CLR;
                    end
                end
                S_CLR: begin
                    // 1 cycle gap so acc_clr lands before en_acc (en_acc is 2-cyc delayed)
                    sweeping <= 1'b1;
                    state    <= S_SWEEP;
                end
                S_SWEEP: begin
                    // advance (ic_grp inner, tap outer) — 4 ic per cycle, same tap
                    if (ic_grp == n_grp - 3'd1) begin
                        ic_grp  <= 3'd0;
                        tap_cnt <= tap_cnt + 4'd1;
                    end else begin
                        ic_grp <= ic_grp + 3'd1;
                    end
                    if (last_term) begin
                        sweeping  <= 1'b0;
                        drain_cnt <= 3'd0;
                        state     <= S_DRAIN;
                    end
                end
                S_DRAIN: begin
                    // wait for last product to propagate (2 mac + 1 acc)
                    drain_cnt <= drain_cnt + 3'd1;
                    if (drain_cnt == 3'd3) begin
                        if (no_flush) begin
                            done  <= 1'b1;        // chan-tiling middle chunk: keep psum, no requant
                            state <= S_IDLE;
                        end else begin
                            flushing <= 1'b1; flush_p <= 5'd0; wp <= 5'd0;
                            state <= S_FLUSH;
                        end
                    end
                end
                S_FLUSH: begin
                    // present psum[oc][flush_p]+bias to the 4 rr_units; collect on rr_vld
                    if (flush_p < 5'd16) flush_p <= flush_p + 5'd1;
                    if (rr_vld[0]) begin
                        out_tile[0][wp[3:0]] <= rr_out[0];
                        out_tile[1][wp[3:0]] <= rr_out[1];
                        out_tile[2][wp[3:0]] <= rr_out[2];
                        out_tile[3][wp[3:0]] <= rr_out[3];
                        wp <= wp + 5'd1;
                        if (wp == 5'd15) begin
                            flushing <= 1'b0;
                            done     <= 1'b1;
                            state    <= S_IDLE;
                        end
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
