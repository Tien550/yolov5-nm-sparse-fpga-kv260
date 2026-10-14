// rr_unit.v  --  Re-quantize + LeakyReLU: INT32 PSUM -> INT8
// Paper: "RR module" Fig. 8  (CLAUDE.md §8.4)
// Pipeline : 3 stages -- multiply -> shift -> relu+clip
// DSP usage: up to 2 DSP48E2 (32x16 exceeds single 27x18 DSP limit)
//
// Multiply analysis:
//   psum_in : 32-bit signed
//   rq_scale: 16-bit unsigned
//   product  : 48-bit signed (needs full width, shift values are 16..27)
//   DSP48E2 max: 27x18 signed. A 32x16 unsigned needs 2 cascaded DSPs in Vivado.
//
// Stage-2 truncation: $signed(tmp48_s1) >>> shift_s1 produces 48-bit result
//   assigned to 32-bit tmp32_s2. With shift >= 16 (typical: shift=23..27 from
//   requant_params), the 48-bit product after shift fits in 32 bits:
//   max |psum|=2^31, max M0=65535 -> product<=2^47; shift>=16 -> result<=2^31 OK.
//   shift=0 edge case: result=full 48-bit product, would truncate. Not expected
//   in practice (shift always >= DATA_W=8 by construction of find_best_shift).

`timescale 1ns/1ps
`include "cnn_defines.vh"

module rr_unit (
    input  wire                        clk,
    input  wire                        rst_n,       // synchronous active-low

    input  wire signed  [`ACC_W-1:0]   psum_in,     // 32-bit partial sum
    input  wire         [`SCALE_W-1:0] rq_scale,    // M0 multiplier (unsigned UINT16)
    input  wire         [`SHIFT_W-1:0] rq_shift,    // right-shift amount (0..31)
    input  wire                        apply_relu,  // 1 = leaky relu (>>>LEAKY_SLOPE)
    input  wire                        valid_in,

    output wire signed  [`DATA_W-1:0]  data_out,    // INT8 result
    output wire                        valid_out
);

    // ── Stage 1: psum * rq_scale -> 48-bit ──────────────────────────────────
    // Use explicit 48-bit wires for operands so Verilog context-determines
    // the multiply width correctly (not truncated to max-operand width).
    // (* use_dsp = "yes" *) guides Vivado to use DSP tiles.
    wire signed [47:0] psum_ext;    // sign-extended psum_in to 48-bit
    wire signed [47:0] scale_ext;   // zero-extended rq_scale to 48-bit signed
    assign psum_ext  = {{16{psum_in[`ACC_W-1]}}, psum_in};
    assign scale_ext = {32'b0, rq_scale};           // MSB=0 -> always non-negative

    (* use_dsp = "yes" *) reg signed [47:0] tmp48_s1;
    reg [`SHIFT_W-1:0]  shift_s1;
    reg                 relu_s1, v_s1;

    // ── Stage 2a (H3 200MHz): round-add only (split from the variable shift) ─
    reg signed [47:0]   tmp48_2a;   // tmp48_s1 + round bias
    reg [`SHIFT_W-1:0]  shift_2a;
    reg                 relu_2a, v_2a;

    // ── Stage 2b: arithmetic right-shift, keep 48 bits to avoid truncation ──
    reg signed [47:0]   tmp48_s2;   // full-width after shift (safe: shift>=8 guaranteed)
    reg                 relu_s2, v_s2;

    // ── Stage 3a (H3 200MHz): LeakyReLU multiply only (split from the clip) ─
    reg signed [47:0]        arelu_3a;
    reg                      v_3a;
    // ── Stage 3b: saturating clip -> INT8 ───────────────────────────────────
    reg signed [`DATA_W-1:0] out_s3;
    reg                      v_s3;

    always @(posedge clk) begin
        if (!rst_n) begin
            tmp48_s1 <= 48'sd0;
            shift_s1 <= {`SHIFT_W{1'b0}};
            relu_s1  <= 1'b0;
            v_s1     <= 1'b0;
        end else begin
            tmp48_s1 <= psum_ext * scale_ext;
            shift_s1 <= rq_shift;
            relu_s1  <= apply_relu;
            v_s1     <= valid_in;
        end
    end

    // Stage 2a: ROUND-half-up bias add only (the cheap part); shift carried to 2b.
    //   (prod + (1<<(shift-1)))   — Rev38 round (round=98% exact vs golden; truncate=58-67%).
    always @(posedge clk) begin
        if (!rst_n) begin
            tmp48_2a <= 48'sd0; shift_2a <= {`SHIFT_W{1'b0}}; relu_2a <= 1'b0; v_2a <= 1'b0;
        end else begin
            tmp48_2a <= (shift_s1 == {`SHIFT_W{1'b0}})
                        ? $signed(tmp48_s1)
                        : $signed(tmp48_s1)
                          + ($signed(48'sd1) <<< (shift_s1 - {{(`SHIFT_W-1){1'b0}},1'b1}));
            shift_2a <= shift_s1;
            relu_2a  <= relu_s1;
            v_2a     <= v_s1;
        end
    end

    // Stage 2b: the variable 48-bit arithmetic right-shift (the long mux — own stage for 200MHz).
    always @(posedge clk) begin
        if (!rst_n) begin
            tmp48_s2 <= 48'sd0; relu_s2 <= 1'b0; v_s2 <= 1'b0;
        end else begin
            tmp48_s2 <= (shift_2a == {`SHIFT_W{1'b0}})
                        ? $signed(tmp48_2a)
                        : ($signed(tmp48_2a) >>> shift_2a);
            relu_s2  <= relu_2a;
            v_s2     <= v_2a;
        end
    end

    // Stage 3a: LeakyReLU(0.1) multiply — slope 0.1 ≈ 205/2048 round-half-up:
    //   (x*205 + 1024) >>> 11 = ×0.10009 (yolov5 default_act = nn.LeakyReLU(0.1)).
    // The 48×8 multiply is its own stage for 200MHz (was bundled with the clip).
    always @(posedge clk) begin
        if (!rst_n) begin
            arelu_3a <= 48'sd0;
            v_3a     <= 1'b0;
        end else begin
            arelu_3a <= (relu_s2 && tmp48_s2[47])
                        ? ($signed(tmp48_s2) * 48'sd205 + 48'sd1024) >>> 11
                        : tmp48_s2;
            v_3a     <= v_s2;
        end
    end

    // Stage 3b: saturating clip to INT8 [-128, 127].
    always @(posedge clk) begin
        if (!rst_n) begin
            out_s3 <= {`DATA_W{1'b0}};
            v_s3   <= 1'b0;
        end else begin
            v_s3 <= v_3a;
            if      (arelu_3a > 48'sd127)   out_s3 <= 8'sd127;
            else if (arelu_3a < -48'sd128)  out_s3 <= -8'sd128;
            else                            out_s3 <= arelu_3a[`DATA_W-1:0];
        end
    end

    assign data_out  = out_s3;
    assign valid_out = v_s3;

endmodule
