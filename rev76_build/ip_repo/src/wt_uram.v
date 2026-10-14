// wt_uram.v -- WT-stationary URAM (B2): holds an entire out-ch-block's weights (all N_ICB
// chunks) so the layer_sequencer DMAs WT once per oc-block instead of every kick. The kernel
// reads chunk `ic` at byte address ic*WT_BLK_BYTES + within_chunk_addr.
//
// WIDE (128-bit) version: the DMA writes a full 128-bit AXI beat per cycle (16x the old
// 1-byte/cycle WT load — the LS_WTU load was a big slice of the kernel-idle window). The
// kernel still reads 1 byte/cycle, so the read path reads the 128-bit word then byte-selects.
// Rev139: 256 KB (c3 9-pass worst case = L23/L53: 144 chunks × 1408B = 202 KB with 64B-aligned
// entries). Word-read mode (rd_wsel=1, c3 ST_WT_LOAD_W) TIME-MULTIPLEXES the single read port
// (byte reads and word reads are in different FSM states, never concurrent) — stays 1W1R so
// Vivado's URAM inference is unchanged (multi-port configs crash the generator, see in_bram_wide).
// Byte-read RD_LAT=3 (Rev139, was 2): extra output register cuts the URAM->DSP route for
// 166 MHz; all FSM consumers retuned +1. Word-read RD_LAT=3 same path (rd_wdata).
// Read-during-write never happens (drain guard gates the kernel read until dma_engine is idle).
`timescale 1ns/1ps

module wt_uram #(
    parameter WORD_W = 128,           // 16 bytes/word
    parameter DEPTH  = 16384,         // 16384 words × 16 = 262144 bytes (256 KB)
    parameter WAW    = 14,            // word-address width (16384 = 2^14)
    parameter BAW    = 18             // byte-address width (262144 = 2^18)
) (
    input  wire            clk,
    // wide write: full 128-bit beat → one word/cycle
    input  wire [WAW-1:0]  wr_waddr,  // word address
    input  wire [WORD_W-1:0] wr_wdata,
    input  wire            wr_wen,
    // byte read: byte address → word read + byte select (RD_LAT=3)
    input  wire [BAW-1:0]  rd_baddr,  // byte address
    output reg  [7:0]      rd_data,
    // word read (c3 wide WT load): shares the read port via rd_wsel mux (RD_LAT=3)
    input  wire            rd_wsel,   // 1 = word mode (rd_waddr), 0 = byte mode (rd_baddr)
    input  wire [WAW-1:0]  rd_waddr,
    output reg  [WORD_W-1:0] rd_wdata
);
    (* ram_style = "ultra", cascade_height = 4 *) reg [WORD_W-1:0] mem [0:DEPTH-1];  // Rev139c
    wire [WAW-1:0]   rd_addr_mux = rd_wsel ? rd_waddr : rd_baddr[BAW-1:4];
    reg [WORD_W-1:0] rd_word;
    reg [3:0]        rd_b1;
    reg [7:0]        rd_byte;   // Rev139: byte select stage; rd_data = output reg (RD_LAT=3)
    always @(posedge clk) begin
        if (wr_wen) mem[wr_waddr] <= wr_wdata;
        rd_word <= mem[rd_addr_mux];                       // word read (1 cyc, shared port)
        rd_b1   <= rd_baddr[3:0];
        rd_byte <= rd_word[{rd_b1, 3'b000} +: 8];          // byte select (2 cyc)
        rd_data <= rd_byte;                                // output reg (3 cyc)
        rd_wdata<= rd_word;                                // word out (2 cyc)... +consumer reg = 3
    end
endmodule
