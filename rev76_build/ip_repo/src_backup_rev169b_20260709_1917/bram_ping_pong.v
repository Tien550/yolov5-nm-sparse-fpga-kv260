// bram_ping_pong.v  --  Double-buffered BRAM: DMA writes one bank while SPEC reads other
// Paper: ping-pong BRAM (CLAUDE.md §8.13)
// Resource: 2 × RAMB36E2 per instance (inferred for simulation; Vivado maps to BRAM36)
//
// Parameters:
//   DATA_W   : data width in bits (default 8 = INT8)
//   DEPTH    : address depth (default 784 = 28×28 tile)
//   RD_LAT   : read latency in cycles (2 = registered MUX, 1 = combinational MUX).
//              IN/WT BRAM use RD_LAT=2 (FSM act/wt-load windows expect 2-cycle).
//              OUT BRAM uses RD_LAT=1 because dma_engine's write-to-DDR fetch
//              path assumes 1-cycle BRAM latency ("1 cycle latency each", see
//              dma_engine ST_WR_FETCH). With RD_LAT=2 the DMA re-reads addr 0
//              (it lingers 2 cycles) → byte1 = bank[0] → 1-pixel output offset
//              ("lệch 1": out[0]=g0, out[1]=g0 dup, out[k]=g[k-1]). Rev37 fix.
//   PINGPONG : 1 = true double-buffer (toggle bank via pp_bank_sel).
//              0 = single buffer (always write bank_a AND read bank_a). OUT BRAM
//              is read only AFTER compute fully completes (dma_sequencer waits
//              ap_done), so it never needs concurrent read/write → ping-pong is
//              unnecessary and its bank parity caused Run1-after-reload to read
//              the wrong (empty) bank ("Run1=zeros"). Rev37: OUT uses PINGPONG=0.
//
// pp_bank_sel (PINGPONG=1 only): 0 → Bank A written by DMA / Bank B read by SPEC
//                                1 → Bank B written by DMA / Bank A read by SPEC
// Swap occurs only on tile boundary (asserted one cycle by caller).
// Both banks are simple-dual-port: port A = write-only, port B = read-only.

`timescale 1ns/1ps

module bram_ping_pong #(
    parameter DATA_W   = 8,
    parameter DEPTH    = 784,  // 28*28 pixels per tile per channel
    parameter RD_LAT   = 2,    // 2 = registered MUX (IN/WT), 1 = comb MUX (OUT)
    parameter PINGPONG = 1     // 1 = double-buffer, 0 = single buffer (OUT)
) (
    input  wire                  clk,
    input  wire                  rst_n,

    // Bank select: toggles at tile boundary (ignored when PINGPONG=0)
    input  wire                  pp_bank_sel,   // 0=A←DMA/B→SPEC, 1=B←DMA/A→SPEC

    // DMA write port (writes to inactive-for-SPEC bank)
    input  wire [$clog2(DEPTH)-1:0] dma_wr_addr,
    input  wire [DATA_W-1:0]        dma_wr_data,
    input  wire                     dma_wr_en,

    // SPEC read port (reads from active bank)
    input  wire [$clog2(DEPTH)-1:0] spec_rd_addr,
    output reg  [DATA_W-1:0]        spec_rd_data   // RD_LAT-cycle latency
);

    // Single-buffer (PINGPONG=0): force write→bank_a AND read→bank_a (same bank).
    // Double-buffer (PINGPONG=1): write/read opposite banks selected by pp_bank_sel.
    wire wr_sel = (PINGPONG != 0) ? pp_bank_sel : 1'b0;  // which bank DMA writes
    wire rd_sel = (PINGPONG != 0) ? pp_bank_sel : 1'b1;  // which bank SPEC reads

    // Separate always blocks per bank — required for Vivado RAMB36E2 inference.
    // Mixed read/write MUX in a single always block prevents BRAM recognition.
    // ram_style="block" forces BRAM over distributed RAM.
    (* ram_style = "block" *) reg [DATA_W-1:0] bank_a [0:DEPTH-1];
    (* ram_style = "block" *) reg [DATA_W-1:0] bank_b [0:DEPTH-1];

    // Write enables per bank
    wire wr_en_a = dma_wr_en & ~wr_sel;  // DMA → bank_a when wr_sel=0
    wire wr_en_b = dma_wr_en &  wr_sel;  // DMA → bank_b when wr_sel=1

    // Read outputs (registered, 1-cycle latency)
    reg [DATA_W-1:0] rd_data_a, rd_data_b;

    // Bank A: write port and read port in SEPARATE always blocks so Vivado
    // infers true Simple Dual Port BRAM (not mixed-port which can cause
    // write-mode conflicts or non-BRAM inference).
    always @(posedge clk) begin
        if (wr_en_a)
            bank_a[dma_wr_addr] <= dma_wr_data;
    end
    always @(posedge clk) begin
        rd_data_a <= bank_a[spec_rd_addr];
    end

    // Bank B: same separation
    always @(posedge clk) begin
        if (wr_en_b)
            bank_b[dma_wr_addr] <= dma_wr_data;
    end
    always @(posedge clk) begin
        rd_data_b <= bank_b[spec_rd_addr];
    end

    // Output MUX: select which bank SPEC reads.
    //   RD_LAT=2 → registered MUX (total 2-cycle latency: addr→rd_data→spec_rd_data)
    //   RD_LAT=1 → combinational MUX (total 1-cycle latency: addr→rd_data, MUX comb)
    generate
        if (RD_LAT >= 2) begin : g_rdlat2
            always @(posedge clk)
                spec_rd_data <= rd_sel ? rd_data_a : rd_data_b;
        end else begin : g_rdlat1
            always @(*)
                spec_rd_data = rd_sel ? rd_data_a : rd_data_b;
        end
    endgenerate

endmodule
