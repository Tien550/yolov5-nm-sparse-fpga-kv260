// sliding_win_ctrl.v -- 3x3 kernel sliding window controller
// Used by conv3x3_top and dwconv3x3_top.
//
// For each output pixel, the caller pulses start_pixel once.
// This module then sequences through all 9 kernel positions (kh=0..2, kw=0..2)
// and drives shuf_mode to depth_shuffle in i_spec:
//   kh=0, kw=0  : SHUF_LOAD  -- full reload of leftmost column
//   any kw>0    : SHUF_SHIFT -- shift left, load rightmost column
//   kw=0, kh>0  : SHUF_LOAD  -- reload for new row
// Between rows the caller reloads activation via the top FSM; this module
// only drives kh_out/kw_out and shuf_mode_out per position.
//
// kernel_done pulses for 1 cycle after position kh=2,kw=2 is emitted.

`timescale 1ns/1ps

module sliding_win_ctrl (
    input  wire       clk,
    input  wire       rst_n,

    // Pulse high for 1 cycle to begin a new output pixel scan
    input  wire       start_pixel,

    // Current kernel position
    output reg  [1:0] kh_out,        // 0..2
    output reg  [1:0] kw_out,        // 0..2

    // depth_shuffle control
    // 2'b00 = SHUF_LOAD, 2'b01 = SHUF_SHIFT, 2'b10 = SHUF_HOLD
    output reg  [1:0] shuf_mode_out,

    // Pulses 1 cycle when kh=2,kw=2 position has been emitted
    output reg        kernel_done
);

    // Encoding matches conv3x3_top / dwconv3x3_top localparam
    localparam SHUF_LOAD  = 2'b00;
    localparam SHUF_SHIFT = 2'b01;
    localparam SHUF_HOLD  = 2'b10;

    localparam ST_IDLE   = 2'd0;
    localparam ST_EMIT   = 2'd1;   // emitting current (kh,kw) position
    localparam ST_HOLD   = 2'd2;   // 1-cycle hold before advancing kw
    localparam ST_DONE   = 2'd3;

    reg [1:0] state;
    reg [1:0] kh_r, kw_r;

    always @(posedge clk) begin
        if (!rst_n) begin
            state        <= ST_IDLE;
            kh_r         <= 2'd0;
            kw_r         <= 2'd0;
            kh_out       <= 2'd0;
            kw_out       <= 2'd0;
            shuf_mode_out<= SHUF_HOLD;
            kernel_done  <= 1'b0;
        end else begin
            kernel_done  <= 1'b0;
            shuf_mode_out<= SHUF_HOLD;

            case (state)
                ST_IDLE: begin
                    if (start_pixel) begin
                        kh_r  <= 2'd0;
                        kw_r  <= 2'd0;
                        state <= ST_EMIT;
                    end
                end

                ST_EMIT: begin
                    // Drive outputs for current position
                    kh_out <= kh_r;
                    kw_out <= kw_r;
                    // kw=0 always reloads; kw>0 shifts
                    shuf_mode_out <= (kw_r == 2'd0) ? SHUF_LOAD : SHUF_SHIFT;

                    if (kh_r == 2'd2 && kw_r == 2'd2) begin
                        kernel_done <= 1'b1;
                        state       <= ST_DONE;
                    end else begin
                        state <= ST_HOLD;
                    end
                end

                ST_HOLD: begin
                    // Advance to next position
                    if (kw_r == 2'd2) begin
                        kw_r <= 2'd0;
                        kh_r <= kh_r + 2'd1;
                    end else begin
                        kw_r <= kw_r + 2'd1;
                    end
                    state <= ST_EMIT;
                end

                ST_DONE: begin
                    shuf_mode_out <= SHUF_HOLD;
                    state         <= ST_IDLE;
                end

                default: state <= ST_IDLE;
            endcase
        end
    end

endmodule
