// pe_array.v -- N×N output-stationary MAC array. O[i][j] = sum_k A[i][k]·W[k][j].
// conv1x1: i=Cout, j=HW, k=Cin. conv3x3 H2 dual-K: each `en` event accumulates TWO k-terms
// (act_col·wt_row + act_col_b·wt_row_b); single-K ties the _b ports to 0 (Vivado folds them out).
//
// B3 200MHz PIPELINE: register the products (MREG) so the DSP48E2 uses its M-register between the
// multiply and the accumulate (the Vivado DRC DPOP-4 fix). en/clr are delayed 1 cycle to stay aligned
// with the registered products. NET EFFECT: the accumulate lands ONE clock later than the old
// combinatorial-multiply version -> consumers must drain +1 cycle (conv1x1 drain 1->2, conv3x3 2->3).
// This is a pure latency change (functionally identical at any clock); it does NOT touch the feed/URAM
// read timing, so it cannot reintroduce the dma-read race that bit the Rev80-89 feed-pipeline.
`timescale 1ns/1ps

module pe_array #(
    parameter integer N    = 16,
    parameter integer AW   = 8,
    parameter integer ACCW = 32,
    parameter integer DUAL_K = 1     // 1 = accumulate 2 k-terms/cyc (conv3x3 dual-tap). 0 = single-K
                                     // (conv1x1): prod_b NOT instantiated -> saves N*N=256 DEAD DSPs.
) (
    input  wire                     clk,
    input  wire                     rst_n,
    input  wire                     clr,
    input  wire                     en,
    input  wire [N*AW-1:0]          act_col,    // A[i][k]   at act_col [i*AW +: AW]
    input  wire [N*AW-1:0]          wt_row,     // W[k][j]   at wt_row  [j*AW +: AW]
    input  wire [N*AW-1:0]          act_col_b,  // A[i][k+1] (2nd k-term; tie 0 for single-K)
    input  wire [N*AW-1:0]          wt_row_b,   // W[k+1][j] (2nd k-term; tie 0 for single-K)
    input  wire [$clog2(N)-1:0]     rd_i,
    input  wire [$clog2(N)-1:0]     rd_j,
    output wire signed [ACCW-1:0]   rd_o,
    output wire [N*ACCW-1:0]        rd_row_o    // full row acc[rd_i][0..N-1] (wide readout, P4-B)
);
    reg signed [ACCW-1:0] acc [0:N-1][0:N-1];
    // en/clr delayed 1 cycle to align with the registered (MREG) products. Shared across all PEs.
    reg en_d, clr_d;
    always @(posedge clk) begin
        en_d  <= en;
        clr_d <= clr;
    end
    genvar gi, gj;
    // MREG-explicit DSP48E2 pipeline: separate always blocks so Vivado maps the multiply
    // directly into the DSP M-register (MREG=1) instead of a fabric FF (fixes DPOP-4 warnings).
    // Key: (* use_dsp="yes" *) on the REGISTER (not wire) tells Vivado the entire multiply+register
    // must be in one DSP. en/clr delayed 1 cycle (en_d/clr_d) to align with registered products.
    generate for (gi=0; gi<N; gi=gi+1) begin: g_i
        for (gj=0; gj<N; gj=gj+1) begin: g_j
            if (DUAL_K) begin: g_dual   // conv3x3: 2 DSP per PE, MREG separates multiply from accumulate
                (* use_dsp = "yes" *) reg signed [2*AW-1:0] prod_r;
                (* use_dsp = "yes" *) reg signed [2*AW-1:0] prodb_r;
                always @(posedge clk) prod_r  <= $signed(act_col  [gi*AW+:AW]) * $signed(wt_row  [gj*AW+:AW]);
                always @(posedge clk) prodb_r <= $signed(act_col_b[gi*AW+:AW]) * $signed(wt_row_b[gj*AW+:AW]);
                always @(posedge clk) begin
                    if (!rst_n || clr_d) acc[gi][gj] <= {ACCW{1'b0}};
                    else if (en_d) acc[gi][gj] <= acc[gi][gj]
                                                + {{(ACCW-2*AW){prod_r [2*AW-1]}}, prod_r }
                                                + {{(ACCW-2*AW){prodb_r[2*AW-1]}}, prodb_r};
                end
            end else begin: g_single    // conv1x1: 1 DSP, MREG breaks multiply->accumulate path
                (* use_dsp = "yes" *) reg signed [2*AW-1:0] prod_m;
                always @(posedge clk) prod_m <= $signed(act_col[gi*AW+:AW]) * $signed(wt_row[gj*AW+:AW]);
                always @(posedge clk) begin  // en_d/clr_d: 1-cycle delay to align with MREG product
                    if (!rst_n || clr_d) acc[gi][gj] <= {ACCW{1'b0}};
                    else if (en_d) acc[gi][gj] <= acc[gi][gj]
                                                + {{(ACCW-2*AW){prod_m[2*AW-1]}}, prod_m};
                end
            end
        end
    end endgenerate
    assign rd_o = acc[rd_i][rd_j];
    generate for (gj=0; gj<N; gj=gj+1) begin: g_row
        assign rd_row_o[gj*ACCW +: ACCW] = acc[rd_i][gj];
    end endgenerate
endmodule
