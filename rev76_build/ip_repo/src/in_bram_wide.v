// in_bram_wide.v -- IN activation BRAM with a WIDE (128-bit) read for the R1 wide-ACT loader
// (read 16 bytes/cycle into act_flat) PLUS an 8-bit read for the legacy per-tile (mode-0) path.
// Organized 128-bit-wide (64 words × 16 bytes = 1024B), 2 banks (ping-pong, same parity as
// bram_ping_pong: pp_bank_sel→ DMA writes one bank / readers read the other). DMA writes 1 byte
// (byte-write via part-select). Both reads RD_LAT=2 (matches the old IN BRAM timing).
`timescale 1ns/1ps

module in_bram_wide (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         pp_bank_sel,
    // DMA write (1 byte/cycle, 10-bit byte address) — legacy/mode-0 only
    input  wire [9:0]   dma_wr_addr,
    input  wire [7:0]   dma_wr_data,
    input  wire         dma_wr_en,
    // Wide DMA write (full 128-bit beat → one word/cycle, 6-bit word address) — fast ACT load
    input  wire [5:0]   dma_w128_addr,
    input  wire [127:0] dma_w128_data,
    input  wire         dma_w128_en,
    // 8-bit read (mode-0 legacy path), RD_LAT=3 (Rev139, was 2: +1 output reg — uniform
    // latency with act_uram/wt_uram so the shared FSM capture offsets stay mode-independent)
    input  wire [9:0]   rd8_addr,
    output reg  [7:0]   rd8_data
);
    wire wr_sel = pp_bank_sel;   // DMA writes bank_a when 0, bank_b when 1
    wire rd_sel = pp_bank_sel;   // readers read bank_b when 0, bank_a when 1 (opposite)

    // distributed (LUT) RAM, NOT block: a 128-bit write + two 128-bit reads + byte-WE on a
    // 128-bit-wide block RAM crashes Vivado 2022.2's hardened RAM generator (HARTGXRamGen
    // assert, Line 4844). LUT-RAM is inferred via a different path (no crash). Only 64×128b×2
    // = 2KB → ~1-2K LUTs (cheap), and it frees the IN block RAMs (design was at 89% BRAM).
    (* ram_style = "distributed" *) reg [127:0] bank_a [0:63];
    (* ram_style = "distributed" *) reg [127:0] bank_b [0:63];

    // ONE write port per bank (byte-write-enable BRAM). Merges the wide 128-bit beat write
    // (fast ACT) and the legacy byte write (mode-0/probe tbs) via priority if-else so it stays
    // a single write port — a SEPARATE 2nd write always-block made each bank 2W2R, exceeding
    // BRAM ports → Vivado RAM-generator crash (HARTGXRamGen assert). x-safe: tbs that drive the
    // byte port but leave dma_w128_en unconnected (x) take the else (if(x)→false) → byte write.
    always @(posedge clk) begin
        if (dma_w128_en && ~wr_sel)
            bank_a[dma_w128_addr] <= dma_w128_data;
        else if (dma_wr_en && ~wr_sel)
            bank_a[dma_wr_addr[9:4]][{dma_wr_addr[3:0], 3'b000} +: 8] <= dma_wr_data;
    end
    always @(posedge clk) begin
        if (dma_w128_en &&  wr_sel)
            bank_b[dma_w128_addr] <= dma_w128_data;
        else if (dma_wr_en &&  wr_sel)
            bank_b[dma_wr_addr[9:4]][{dma_wr_addr[3:0], 3'b000} +: 8] <= dma_wr_data;
    end

    // 8-bit read (mode-0): RD_LAT=3 → stage1 word fetch + byte latch, stage2 byte select,
    // stage3 output reg (Rev139: R1 rd128 port removed — R1 pruned for NS=12 DSP budget)
    reg [127:0] w8_a, w8_b;  reg [3:0] b1;  reg [7:0] rd8_byte;
    always @(posedge clk) begin
        w8_a <= bank_a[rd8_addr[9:4]]; w8_b <= bank_b[rd8_addr[9:4]]; b1 <= rd8_addr[3:0];
    end
    wire [127:0] w8_sel = rd_sel ? w8_a : w8_b;
    always @(posedge clk) begin
        rd8_byte <= w8_sel[{b1, 3'b000} +: 8];
        rd8_data <= rd8_byte;
    end
endmodule
