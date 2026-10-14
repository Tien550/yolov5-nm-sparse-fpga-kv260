// mac_unit.v  --  INT8 signed multiply -> INT32, 2-stage pipeline
// Paper: "MAC" inside Mul_Group (CLAUDE.md §8.2)
// DSP usage : 1 DSP48E2 per instance
// Latency   : 2 clock cycles (input -> output)

`timescale 1ns/1ps
`include "cnn_defines.vh"

(* keep_hierarchy = "yes" *)
module mac_unit (
    input  wire                  clk,
    input  wire                  rst_n,    // synchronous active-low

    input  wire signed [`DATA_W-1:0] a_in,     // INT8 weight
    input  wire signed [`DATA_W-1:0] b_in,     // INT8 activation
    input  wire                  valid_in,

    output wire signed [`ACC_W-1:0]  p_out,    // sign-extended INT32 product
    output wire                  valid_out
);

    // ── Stage 1: register inputs ────────────────────────────────────────────
    reg signed [`DATA_W-1:0] a_s1, b_s1;
    reg                      v_s1;

    // ── Stage 2: register product  (* use_dsp="yes" *) -> DSP48E2 ──────────
    (* use_dsp = "yes" *) reg signed [2*`DATA_W-1:0] product_s2;
    reg                      v_s2;

    always @(posedge clk) begin
        if (!rst_n) begin
            a_s1       <= {`DATA_W{1'b0}};
            b_s1       <= {`DATA_W{1'b0}};
            v_s1       <= 1'b0;
            product_s2 <= {(2*`DATA_W){1'b0}};
            v_s2       <= 1'b0;
        end else begin
            // Stage 1
            a_s1 <= a_in;
            b_s1 <= b_in;
            v_s1 <= valid_in;
            // Stage 2
            product_s2 <= a_s1 * b_s1;
            v_s2       <= v_s1;
        end
    end

    // Sign-extend 16-bit product to ACC_W=32 bits
    assign p_out     = {{(`ACC_W - 2*`DATA_W){product_s2[2*`DATA_W-1]}}, product_s2};
    assign valid_out = v_s2;

endmodule
