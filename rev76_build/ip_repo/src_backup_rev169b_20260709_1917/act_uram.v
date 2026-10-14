// act_uram.v -- ACT-stationary URAM (B-i): holds an ENTIRE layer's activation tiles so the
// layer_sequencer DMAs the ACT once per layer (LS_ACTL) instead of re-reading it per oc-block.
// Root cause fixed: the oc-outer loop re-DMA'd conv1x1 ACT N_OCB times (up to 128x). With ACT
// resident on-chip the sweep of all oc-blocks reads from URAM (no per-oc-block DDR round-trip).
//
// Mirror of wt_uram.v (WT-stationary): WIDE 128-bit write (one AXI beat/cycle from dma_in_w128)
// + byte read (kernel rd8 path). The kernel reads chunk (pos,ic) at byte address
//   ls_act_uram_off + within_tile_addr    (ls_act_uram_off = act_idx * ACT_CHUNK, added by caller)
// B-c3 (Rev139): also holds the raw CHW int8 input plane for conv3x3 9-pass; the engine reads
//   arbitrary byte addr = pixel_base + ch*HW + pos*stride (21-bit).
//
// Sizing (Rev139): 128-bit x 98304 = 1572864 bytes (1.5 MB). Max c3 plane = L01/L02 input
//   (32,208,208) = 1.32 MB; max conv1x1 packed ACT = 692 KB. ~44 URAM288.
// 1W1R (one wide write + one byte-select read) -> avoids the 2W2R/byte-WE config that crashes
// Vivado's block-RAM generator (same rule as wt_uram / in_bram_wide).
// RD_LAT=3 (Rev139, was 2): extra output register cuts the URAM->DSP route for 166 MHz.
// All FSM consumers retuned +1 (act_ld_cnt capture windows).
// Read-during-write never happens: LS_ACTL loads the whole layer before the kick loop reads.
`timescale 1ns/1ps

module act_uram #(
    parameter WORD_W = 128,           // 16 bytes/word
    parameter DEPTH  = 98304,         // 98304 words x 16 = 1572864 bytes (1.5 MB)
    parameter WAW    = 17,            // word-address width (2^17 = 131072 >= 98304)
    parameter BAW    = 21             // byte-address width (2^21 = 2 MB >= 1.5 MB)
) (
    input  wire            clk,
    // wide write: full 128-bit beat -> one word/cycle
    input  wire [WAW-1:0]  wr_waddr,  // word address
    input  wire [WORD_W-1:0] wr_wdata,
    input  wire            wr_wen,
    // byte read: byte address -> word read + byte select (RD_LAT=3)
    input  wire [BAW-1:0]  rd_baddr,  // byte address
    output reg  [7:0]      rd_data,
    // Rev139d wide-ACT (C3W): word read TIME-MUX chung 1 read port (như wt_uram — byte/word
    // không bao giờ đồng thời: word chỉ dùng trong ST_LOAD_ACT_W của c3). RD_LAT=2.
    input  wire            rd_wsel,   // 1 = word mode (rd_waddr), 0 = byte mode (rd_baddr)
    input  wire [WAW-1:0]  rd_waddr,
    output reg  [WORD_W-1:0] rd_wdata
);
    // Rev139c: cascade_height=4 — cắt chuỗi cascade addr/DIN của 48 URAM (7 tầng → ~3)
    // đổi lấy mux đầu ra rộng hơn (LUT rẻ, có sẵn output-reg RD_LAT=3 hấp thụ).
    (* ram_style = "ultra", cascade_height = 4 *) reg [WORD_W-1:0] mem [0:DEPTH-1];
    wire [WAW-1:0]   rd_addr_mux = rd_wsel ? rd_waddr : rd_baddr[BAW-1:4];
    reg [WORD_W-1:0] rd_word;
    reg [3:0]        rd_b1;
    // Rev166: tách mux byte 16:1 → 4:1 (32b) + 4:1 (byte) trải lên 2 tầng reg CÓ SẴN.
    // RD_LAT=3 GIỮ NGUYÊN, byte chọn y hệt (b1*8 = b1[3:2]*32 + b1[1:0]*8). Lý do: retiming
    // dồn URAM-cascade + mux16:1 vào 1 cycle → WNS -0.060 (Rev165). Mux nhỏ 2 tầng → dư slack.
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
