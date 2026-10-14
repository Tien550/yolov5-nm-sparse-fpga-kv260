// config_decoder.v -- Combinational decoder: conv_type -> effective control signals
// Pure combinational: 0 FF, ~12 LUT
// CONV_TYPE_DW forces poc=0 and accu_mode=0 regardless of user register values.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module config_decoder (
    input  wire [1:0]  reg_conv_type,   // CONV_TYPE_1X1/3X3/DW from AXI_OFF_LAYER_TYPE
    input  wire [1:0]  reg_poc,
    input  wire [1:0]  reg_accu_mode,
    input  wire        reg_stride,      // stride bit [0] (0=stride1, 1=stride2)

    output reg         swc_en,          // 1=enable sliding_win_ctrl, 0=bypass (1x1)
    output reg  [1:0]  eff_poc,         // effective poc (forced=2'd0 for dw)
    output reg  [1:0]  eff_accu_mode,   // effective accu_mode (forced=2'd0 for dw)
    output reg         use_halo,        // 1=add 1-pixel halo (3x3/dw), 0=no halo (1x1)
    output reg         dw_mode          // 1=depthwise: Cin==Cout
);

    always @(*) begin
        case (reg_conv_type)
            `CONV_TYPE_1X1: begin
                swc_en       = 1'b0;
                eff_poc      = reg_poc;
                eff_accu_mode= reg_accu_mode;
                use_halo     = 1'b0;
                dw_mode      = 1'b0;
            end
            `CONV_TYPE_3X3: begin
                swc_en       = 1'b1;
                eff_poc      = reg_poc;
                eff_accu_mode= reg_accu_mode;
                use_halo     = 1'b1;
                dw_mode      = 1'b0;
            end
            `CONV_TYPE_DW: begin
                swc_en       = 1'b1;
                eff_poc      = 2'd0;           // FORCE poc=1 (1 channel per SPEC)
                eff_accu_mode= 2'd0;           // FORCE 1-in (no cross-channel accu)
                use_halo     = 1'b1;
                dw_mode      = 1'b1;
            end
            default: begin
                swc_en       = 1'b0;
                eff_poc      = reg_poc;
                eff_accu_mode= reg_accu_mode;
                use_halo     = 1'b0;
                dw_mode      = 1'b0;
            end
        endcase
    end

endmodule
