// act_bram_in.v -- Rev168 RESIDENT-ACT: BRAM bank chứa INPUT plane của layer c3 đang chạy
// (L00: CHW 3x416x416 = 519,168B <= 512KiB) để act_uram TRỐNG TRỌN 1.5MB nhận OUTPUT plane
// qua oc_deint scatter (staging @offset 0 = plane input của layer kế, khỏi DMA-out/deint/ACTL).
//
// Gương act_uram.v 1:1 (1W1R wide-write + byte/word time-mux read, RD_LAT byte=3 word=2 —
// mọi consumer FSM giữ nguyên cửa sổ bắt), chỉ khác ram_style="block" (128 RAMB36; URAM đã 87.5%).
// Mux byte 16:1 tách 4:1+4:1 hai tầng reg y hệt Rev166 (bài học WNS -0.060).
`timescale 1ns/1ps

module act_bram_in #(
    parameter WORD_W = 128,           // 16 bytes/word
    parameter DEPTH  = 32768,         // 32768 words x 16B = 524,288B (>= L00 plane 519,168B)
    parameter WAW    = 15,            // word-address width (2^15 = 32768)
    parameter BAW    = 19             // byte-address width (2^19 = 512KiB)
) (
    input  wire            clk,
    // wide write: full 128-bit beat (LS_ACTL khi act_in_bank=1)
    input  wire [WAW-1:0]  wr_waddr,
    input  wire [WORD_W-1:0] wr_wdata,
    input  wire            wr_wen,
    // byte read (kernel rd8 path, RD_LAT=3)
    input  wire [BAW-1:0]  rd_baddr,
    output reg  [7:0]      rd_data,
    // word read TIME-MUX chung 1 read port (C3W wide-ACT, RD_LAT=2)
    input  wire            rd_wsel,
    input  wire [WAW-1:0]  rd_waddr,
    output reg  [WORD_W-1:0] rd_wdata
);
    // cascade_height=1: DEPTH 32768 = RAMB36E2 32K×1 per bit — KHÔNG cascade DOUT (build168
    // lần 1: Vivado chuỗi cascade 7 tầng → logic 3.34ns, WNS -0.566/1019 endpoint fail).
    (* ram_style = "block", cascade_height = 1 *) reg [WORD_W-1:0] mem [0:DEPTH-1];
    wire [WAW-1:0]   rd_addr_mux = rd_wsel ? rd_waddr : rd_baddr[BAW-1:4];
    reg [WORD_W-1:0] rd_word;
    reg [3:0]        rd_b1;
    reg [31:0]       rd_q32;
    reg [1:0]        rd_b2;
    always @(posedge clk) begin
        if (wr_wen) mem[wr_waddr] <= wr_wdata;
        rd_word <= mem[rd_addr_mux];                       // word read (1 cyc, shared port)
        rd_b1   <= rd_baddr[3:0];
        rd_q32  <= rd_word[{rd_b1[3:2], 5'b00000} +: 32];  // stage2: chọn 32b (4:1)
        rd_b2   <= rd_b1[1:0];
        rd_data <= rd_q32[{rd_b2, 3'b000} +: 8];           // stage3: chọn byte (4:1), RD_LAT=3
        rd_wdata<= rd_word;                                // word out (RD_LAT=2)
    end
endmodule
