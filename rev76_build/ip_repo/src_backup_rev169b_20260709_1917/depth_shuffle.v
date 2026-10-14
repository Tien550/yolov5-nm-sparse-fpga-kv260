// depth_shuffle.v  --  Sliding window register array for activation reuse
// Paper: "Depth Shuffle" Fig. 3, Fig. 9  (CLAUDE.md §8.9)
// Resource: G*P*DATA_W = 4*16*8 = 512 FF, ~32 LUT (mux logic), 0 BRAM
//
// Three modes (shuffle_mode):
//   SHUF_LOAD  (2'b00): load entire G*P window from input data bus
//   SHUF_SHIFT (2'b01): shift right by 1 column; load 1 new column on right
//   SHUF_HOLD  (2'b10): no change
//
// act_regs[g][p]: group g, position p within the P-wide window
// Output act_out[g][p] feeds mul_group_array directly.
//
// Flat port naming: act_in_g{G}_p{P}, act_out_g{G}_p{P}  (G=4, P=16)

`timescale 1ns/1ps
`include "cnn_defines.vh"

module depth_shuffle (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  shuffle_mode,  // 00=LOAD 01=SHIFT 10=HOLD

    // New column inputs (P values, one per group, used in SHIFT mode)
    input  wire signed [`DATA_W-1:0] new_col_0,
    input  wire signed [`DATA_W-1:0] new_col_1,
    input  wire signed [`DATA_W-1:0] new_col_2,
    input  wire signed [`DATA_W-1:0] new_col_3,

    // Full window load inputs (G*P = 64 values, used in LOAD mode)
    // Naming: load_g{g}_p{p}
    input  wire signed [`DATA_W-1:0] load_g0_p0,  input  wire signed [`DATA_W-1:0] load_g0_p1,
    input  wire signed [`DATA_W-1:0] load_g0_p2,  input  wire signed [`DATA_W-1:0] load_g0_p3,
    input  wire signed [`DATA_W-1:0] load_g0_p4,  input  wire signed [`DATA_W-1:0] load_g0_p5,
    input  wire signed [`DATA_W-1:0] load_g0_p6,  input  wire signed [`DATA_W-1:0] load_g0_p7,
    input  wire signed [`DATA_W-1:0] load_g0_p8,  input  wire signed [`DATA_W-1:0] load_g0_p9,
    input  wire signed [`DATA_W-1:0] load_g0_p10, input  wire signed [`DATA_W-1:0] load_g0_p11,
    input  wire signed [`DATA_W-1:0] load_g0_p12, input  wire signed [`DATA_W-1:0] load_g0_p13,
    input  wire signed [`DATA_W-1:0] load_g0_p14, input  wire signed [`DATA_W-1:0] load_g0_p15,

    input  wire signed [`DATA_W-1:0] load_g1_p0,  input  wire signed [`DATA_W-1:0] load_g1_p1,
    input  wire signed [`DATA_W-1:0] load_g1_p2,  input  wire signed [`DATA_W-1:0] load_g1_p3,
    input  wire signed [`DATA_W-1:0] load_g1_p4,  input  wire signed [`DATA_W-1:0] load_g1_p5,
    input  wire signed [`DATA_W-1:0] load_g1_p6,  input  wire signed [`DATA_W-1:0] load_g1_p7,
    input  wire signed [`DATA_W-1:0] load_g1_p8,  input  wire signed [`DATA_W-1:0] load_g1_p9,
    input  wire signed [`DATA_W-1:0] load_g1_p10, input  wire signed [`DATA_W-1:0] load_g1_p11,
    input  wire signed [`DATA_W-1:0] load_g1_p12, input  wire signed [`DATA_W-1:0] load_g1_p13,
    input  wire signed [`DATA_W-1:0] load_g1_p14, input  wire signed [`DATA_W-1:0] load_g1_p15,

    input  wire signed [`DATA_W-1:0] load_g2_p0,  input  wire signed [`DATA_W-1:0] load_g2_p1,
    input  wire signed [`DATA_W-1:0] load_g2_p2,  input  wire signed [`DATA_W-1:0] load_g2_p3,
    input  wire signed [`DATA_W-1:0] load_g2_p4,  input  wire signed [`DATA_W-1:0] load_g2_p5,
    input  wire signed [`DATA_W-1:0] load_g2_p6,  input  wire signed [`DATA_W-1:0] load_g2_p7,
    input  wire signed [`DATA_W-1:0] load_g2_p8,  input  wire signed [`DATA_W-1:0] load_g2_p9,
    input  wire signed [`DATA_W-1:0] load_g2_p10, input  wire signed [`DATA_W-1:0] load_g2_p11,
    input  wire signed [`DATA_W-1:0] load_g2_p12, input  wire signed [`DATA_W-1:0] load_g2_p13,
    input  wire signed [`DATA_W-1:0] load_g2_p14, input  wire signed [`DATA_W-1:0] load_g2_p15,

    input  wire signed [`DATA_W-1:0] load_g3_p0,  input  wire signed [`DATA_W-1:0] load_g3_p1,
    input  wire signed [`DATA_W-1:0] load_g3_p2,  input  wire signed [`DATA_W-1:0] load_g3_p3,
    input  wire signed [`DATA_W-1:0] load_g3_p4,  input  wire signed [`DATA_W-1:0] load_g3_p5,
    input  wire signed [`DATA_W-1:0] load_g3_p6,  input  wire signed [`DATA_W-1:0] load_g3_p7,
    input  wire signed [`DATA_W-1:0] load_g3_p8,  input  wire signed [`DATA_W-1:0] load_g3_p9,
    input  wire signed [`DATA_W-1:0] load_g3_p10, input  wire signed [`DATA_W-1:0] load_g3_p11,
    input  wire signed [`DATA_W-1:0] load_g3_p12, input  wire signed [`DATA_W-1:0] load_g3_p13,
    input  wire signed [`DATA_W-1:0] load_g3_p14, input  wire signed [`DATA_W-1:0] load_g3_p15,

    // Outputs: G*P activation values
    output wire signed [`DATA_W-1:0] act_g0_p0,  output wire signed [`DATA_W-1:0] act_g0_p1,
    output wire signed [`DATA_W-1:0] act_g0_p2,  output wire signed [`DATA_W-1:0] act_g0_p3,
    output wire signed [`DATA_W-1:0] act_g0_p4,  output wire signed [`DATA_W-1:0] act_g0_p5,
    output wire signed [`DATA_W-1:0] act_g0_p6,  output wire signed [`DATA_W-1:0] act_g0_p7,
    output wire signed [`DATA_W-1:0] act_g0_p8,  output wire signed [`DATA_W-1:0] act_g0_p9,
    output wire signed [`DATA_W-1:0] act_g0_p10, output wire signed [`DATA_W-1:0] act_g0_p11,
    output wire signed [`DATA_W-1:0] act_g0_p12, output wire signed [`DATA_W-1:0] act_g0_p13,
    output wire signed [`DATA_W-1:0] act_g0_p14, output wire signed [`DATA_W-1:0] act_g0_p15,

    output wire signed [`DATA_W-1:0] act_g1_p0,  output wire signed [`DATA_W-1:0] act_g1_p1,
    output wire signed [`DATA_W-1:0] act_g1_p2,  output wire signed [`DATA_W-1:0] act_g1_p3,
    output wire signed [`DATA_W-1:0] act_g1_p4,  output wire signed [`DATA_W-1:0] act_g1_p5,
    output wire signed [`DATA_W-1:0] act_g1_p6,  output wire signed [`DATA_W-1:0] act_g1_p7,
    output wire signed [`DATA_W-1:0] act_g1_p8,  output wire signed [`DATA_W-1:0] act_g1_p9,
    output wire signed [`DATA_W-1:0] act_g1_p10, output wire signed [`DATA_W-1:0] act_g1_p11,
    output wire signed [`DATA_W-1:0] act_g1_p12, output wire signed [`DATA_W-1:0] act_g1_p13,
    output wire signed [`DATA_W-1:0] act_g1_p14, output wire signed [`DATA_W-1:0] act_g1_p15,

    output wire signed [`DATA_W-1:0] act_g2_p0,  output wire signed [`DATA_W-1:0] act_g2_p1,
    output wire signed [`DATA_W-1:0] act_g2_p2,  output wire signed [`DATA_W-1:0] act_g2_p3,
    output wire signed [`DATA_W-1:0] act_g2_p4,  output wire signed [`DATA_W-1:0] act_g2_p5,
    output wire signed [`DATA_W-1:0] act_g2_p6,  output wire signed [`DATA_W-1:0] act_g2_p7,
    output wire signed [`DATA_W-1:0] act_g2_p8,  output wire signed [`DATA_W-1:0] act_g2_p9,
    output wire signed [`DATA_W-1:0] act_g2_p10, output wire signed [`DATA_W-1:0] act_g2_p11,
    output wire signed [`DATA_W-1:0] act_g2_p12, output wire signed [`DATA_W-1:0] act_g2_p13,
    output wire signed [`DATA_W-1:0] act_g2_p14, output wire signed [`DATA_W-1:0] act_g2_p15,

    output wire signed [`DATA_W-1:0] act_g3_p0,  output wire signed [`DATA_W-1:0] act_g3_p1,
    output wire signed [`DATA_W-1:0] act_g3_p2,  output wire signed [`DATA_W-1:0] act_g3_p3,
    output wire signed [`DATA_W-1:0] act_g3_p4,  output wire signed [`DATA_W-1:0] act_g3_p5,
    output wire signed [`DATA_W-1:0] act_g3_p6,  output wire signed [`DATA_W-1:0] act_g3_p7,
    output wire signed [`DATA_W-1:0] act_g3_p8,  output wire signed [`DATA_W-1:0] act_g3_p9,
    output wire signed [`DATA_W-1:0] act_g3_p10, output wire signed [`DATA_W-1:0] act_g3_p11,
    output wire signed [`DATA_W-1:0] act_g3_p12, output wire signed [`DATA_W-1:0] act_g3_p13,
    output wire signed [`DATA_W-1:0] act_g3_p14, output wire signed [`DATA_W-1:0] act_g3_p15
);

    localparam SHUF_LOAD  = 2'b00;
    localparam SHUF_SHIFT = 2'b01;
    localparam SHUF_HOLD  = 2'b10;

    // Register array: act_regs[g][p]  (G=4, P=16, DATA_W=8)
    reg signed [`DATA_W-1:0] act_regs [0:`G-1] [0:`P-1];

    integer g, p;

    always @(posedge clk) begin
        if (!rst_n) begin
            for (g = 0; g < `G; g = g + 1)
                for (p = 0; p < `P; p = p + 1)
                    act_regs[g][p] <= {`DATA_W{1'b0}};
        end else begin
            case (shuffle_mode)
                SHUF_LOAD: begin
                    // Load entire window from flat input ports
                    act_regs[0][0]  <= load_g0_p0;  act_regs[0][1]  <= load_g0_p1;
                    act_regs[0][2]  <= load_g0_p2;  act_regs[0][3]  <= load_g0_p3;
                    act_regs[0][4]  <= load_g0_p4;  act_regs[0][5]  <= load_g0_p5;
                    act_regs[0][6]  <= load_g0_p6;  act_regs[0][7]  <= load_g0_p7;
                    act_regs[0][8]  <= load_g0_p8;  act_regs[0][9]  <= load_g0_p9;
                    act_regs[0][10] <= load_g0_p10; act_regs[0][11] <= load_g0_p11;
                    act_regs[0][12] <= load_g0_p12; act_regs[0][13] <= load_g0_p13;
                    act_regs[0][14] <= load_g0_p14; act_regs[0][15] <= load_g0_p15;

                    act_regs[1][0]  <= load_g1_p0;  act_regs[1][1]  <= load_g1_p1;
                    act_regs[1][2]  <= load_g1_p2;  act_regs[1][3]  <= load_g1_p3;
                    act_regs[1][4]  <= load_g1_p4;  act_regs[1][5]  <= load_g1_p5;
                    act_regs[1][6]  <= load_g1_p6;  act_regs[1][7]  <= load_g1_p7;
                    act_regs[1][8]  <= load_g1_p8;  act_regs[1][9]  <= load_g1_p9;
                    act_regs[1][10] <= load_g1_p10; act_regs[1][11] <= load_g1_p11;
                    act_regs[1][12] <= load_g1_p12; act_regs[1][13] <= load_g1_p13;
                    act_regs[1][14] <= load_g1_p14; act_regs[1][15] <= load_g1_p15;

                    act_regs[2][0]  <= load_g2_p0;  act_regs[2][1]  <= load_g2_p1;
                    act_regs[2][2]  <= load_g2_p2;  act_regs[2][3]  <= load_g2_p3;
                    act_regs[2][4]  <= load_g2_p4;  act_regs[2][5]  <= load_g2_p5;
                    act_regs[2][6]  <= load_g2_p6;  act_regs[2][7]  <= load_g2_p7;
                    act_regs[2][8]  <= load_g2_p8;  act_regs[2][9]  <= load_g2_p9;
                    act_regs[2][10] <= load_g2_p10; act_regs[2][11] <= load_g2_p11;
                    act_regs[2][12] <= load_g2_p12; act_regs[2][13] <= load_g2_p13;
                    act_regs[2][14] <= load_g2_p14; act_regs[2][15] <= load_g2_p15;

                    act_regs[3][0]  <= load_g3_p0;  act_regs[3][1]  <= load_g3_p1;
                    act_regs[3][2]  <= load_g3_p2;  act_regs[3][3]  <= load_g3_p3;
                    act_regs[3][4]  <= load_g3_p4;  act_regs[3][5]  <= load_g3_p5;
                    act_regs[3][6]  <= load_g3_p6;  act_regs[3][7]  <= load_g3_p7;
                    act_regs[3][8]  <= load_g3_p8;  act_regs[3][9]  <= load_g3_p9;
                    act_regs[3][10] <= load_g3_p10; act_regs[3][11] <= load_g3_p11;
                    act_regs[3][12] <= load_g3_p12; act_regs[3][13] <= load_g3_p13;
                    act_regs[3][14] <= load_g3_p14; act_regs[3][15] <= load_g3_p15;
                end

                SHUF_SHIFT: begin
                    // Shift left by 1 position (discard p=0, shift p[1..P-1] down, load new at p=P-1)
                    for (g = 0; g < `G; g = g + 1) begin
                        for (p = 0; p < `P-1; p = p + 1)
                            act_regs[g][p] <= act_regs[g][p+1];
                    end
                    act_regs[0][`P-1] <= new_col_0;
                    act_regs[1][`P-1] <= new_col_1;
                    act_regs[2][`P-1] <= new_col_2;
                    act_regs[3][`P-1] <= new_col_3;
                end

                default: begin
                    // SHUF_HOLD: no change
                end
            endcase
        end
    end

    // Flat output assignments
    assign act_g0_p0  = act_regs[0][0];  assign act_g0_p1  = act_regs[0][1];
    assign act_g0_p2  = act_regs[0][2];  assign act_g0_p3  = act_regs[0][3];
    assign act_g0_p4  = act_regs[0][4];  assign act_g0_p5  = act_regs[0][5];
    assign act_g0_p6  = act_regs[0][6];  assign act_g0_p7  = act_regs[0][7];
    assign act_g0_p8  = act_regs[0][8];  assign act_g0_p9  = act_regs[0][9];
    assign act_g0_p10 = act_regs[0][10]; assign act_g0_p11 = act_regs[0][11];
    assign act_g0_p12 = act_regs[0][12]; assign act_g0_p13 = act_regs[0][13];
    assign act_g0_p14 = act_regs[0][14]; assign act_g0_p15 = act_regs[0][15];

    assign act_g1_p0  = act_regs[1][0];  assign act_g1_p1  = act_regs[1][1];
    assign act_g1_p2  = act_regs[1][2];  assign act_g1_p3  = act_regs[1][3];
    assign act_g1_p4  = act_regs[1][4];  assign act_g1_p5  = act_regs[1][5];
    assign act_g1_p6  = act_regs[1][6];  assign act_g1_p7  = act_regs[1][7];
    assign act_g1_p8  = act_regs[1][8];  assign act_g1_p9  = act_regs[1][9];
    assign act_g1_p10 = act_regs[1][10]; assign act_g1_p11 = act_regs[1][11];
    assign act_g1_p12 = act_regs[1][12]; assign act_g1_p13 = act_regs[1][13];
    assign act_g1_p14 = act_regs[1][14]; assign act_g1_p15 = act_regs[1][15];

    assign act_g2_p0  = act_regs[2][0];  assign act_g2_p1  = act_regs[2][1];
    assign act_g2_p2  = act_regs[2][2];  assign act_g2_p3  = act_regs[2][3];
    assign act_g2_p4  = act_regs[2][4];  assign act_g2_p5  = act_regs[2][5];
    assign act_g2_p6  = act_regs[2][6];  assign act_g2_p7  = act_regs[2][7];
    assign act_g2_p8  = act_regs[2][8];  assign act_g2_p9  = act_regs[2][9];
    assign act_g2_p10 = act_regs[2][10]; assign act_g2_p11 = act_regs[2][11];
    assign act_g2_p12 = act_regs[2][12]; assign act_g2_p13 = act_regs[2][13];
    assign act_g2_p14 = act_regs[2][14]; assign act_g2_p15 = act_regs[2][15];

    assign act_g3_p0  = act_regs[3][0];  assign act_g3_p1  = act_regs[3][1];
    assign act_g3_p2  = act_regs[3][2];  assign act_g3_p3  = act_regs[3][3];
    assign act_g3_p4  = act_regs[3][4];  assign act_g3_p5  = act_regs[3][5];
    assign act_g3_p6  = act_regs[3][6];  assign act_g3_p7  = act_regs[3][7];
    assign act_g3_p8  = act_regs[3][8];  assign act_g3_p9  = act_regs[3][9];
    assign act_g3_p10 = act_regs[3][10]; assign act_g3_p11 = act_regs[3][11];
    assign act_g3_p12 = act_regs[3][12]; assign act_g3_p13 = act_regs[3][13];
    assign act_g3_p14 = act_regs[3][14]; assign act_g3_p15 = act_regs[3][15];

endmodule
