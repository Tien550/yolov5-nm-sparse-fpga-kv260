// dma_engine.v -- AXI4 Master DMA Engine (128-bit burst)
// Read:  DDR -> ACT/WT BRAM  (keep rready=1 throughout burst)
// Write: OUT BRAM -> DDR
//
// Key change vs previous version:
//   Read path keeps m_axi_rready=1 at all times during ST_RD_DATA.
//   Each beat is written into BRAM immediately (no separate UNPACK state).
//   Beat data bytes are stored and written one-per-cycle in a small counter.
//
// Write path: read BRAM one byte/cycle, pack into 128-bit register, then send.

`timescale 1ns/1ps

(* keep_hierarchy = "yes" *)
module dma_engine (
    input  wire        aclk,
    input  wire        aresetn,

    input  wire        dma_req,
    input  wire [1:0]  dma_req_tgt,
    input  wire [63:0] dma_req_addr,
    input  wire [23:0] dma_req_bytes,

    output reg         dma_active,
    output reg         dma_done,

    output reg  [63:0]  m_axi_awaddr,
    output reg  [7:0]   m_axi_awlen,
    output wire [2:0]   m_axi_awsize,
    output wire [1:0]   m_axi_awburst,
    output wire [3:0]   m_axi_awcache,
    output wire [2:0]   m_axi_awprot,
    output wire         m_axi_awlock,
    output wire [3:0]   m_axi_awqos,
    output reg          m_axi_awvalid,
    input  wire         m_axi_awready,
    output reg  [127:0] m_axi_wdata,
    output wire [15:0]  m_axi_wstrb,
    output reg          m_axi_wlast,
    output reg          m_axi_wvalid,
    input  wire         m_axi_wready,
    input  wire         m_axi_bvalid,
    output wire         m_axi_bready,

    output reg  [63:0]  m_axi_araddr,
    output reg  [7:0]   m_axi_arlen,
    output wire [2:0]   m_axi_arsize,
    output wire [1:0]   m_axi_arburst,
    output wire [3:0]   m_axi_arcache,
    output wire [2:0]   m_axi_arprot,
    output wire         m_axi_arlock,
    output wire [3:0]   m_axi_arqos,
    output reg          m_axi_arvalid,
    input  wire         m_axi_arready,
    input  wire [127:0] m_axi_rdata,
    input  wire         m_axi_rlast,
    input  wire         m_axi_rvalid,
    output wire         m_axi_rready,   // WIRE: always 1 during read

    output reg  [9:0]   dma_in_addr,
    output reg  [7:0]   dma_in_data,
    output reg          dma_in_wr_en,

    // Wide ACT write: full 128-bit AXI beat → one IN-BRAM word/cycle (16× the byte path).
    // Used for TGT_ACT only; keeps rready high so the burst streams at full AXI rate.
    output reg  [5:0]   dma_in_w128_addr,
    output reg  [127:0] dma_in_w128_data,
    output reg          dma_in_w128_en,

    output reg  [11:0]  dma_wt_addr,
    output reg  [7:0]   dma_wt_data,
    output reg          dma_wt_wr_en,

    // Wide WT write: full 128-bit AXI beat → one WT-URAM word/cycle (16× the byte path).
    output reg  [12:0]  dma_wt_w128_addr,   // WT-URAM word address (8192 words)
    output reg  [127:0] dma_wt_w128_data,
    output reg          dma_wt_w128_en,

    output reg  [18:0]  dma_out_addr,
    input  wire [7:0]   dma_out_data,

    // WIDE OUTPUT (Rev130): read omem 128-bit/beat instead of byte-serial (16× fewer cycles).
    // out_wide=1 (sys/c3/spec engines, omem is 128-bit) → stream 1 word/beat. dw uses byte path.
    input  wire         out_wide,
    input  wire [127:0] dma_out_data128,

    // DESCRIPTOR read (Rev132, TGT_DESC=3): DDR descriptor table → network_seq desc buffer.
    // ADDITIVE: existing engines never issue tgt=3, so ACT/WT/OUT paths are untouched. 64-byte
    // descriptor: Đòn 4b v2 = 8 beats (128B/layer). beat index = bram_addr[6:4].
    output reg  [2:0]   dma_desc_w128_addr,
    output reg  [127:0] dma_desc_w128_data,
    output reg          dma_desc_w128_en
);

    assign m_axi_awsize  = 3'd4;
    assign m_axi_awburst = 2'd1;
    assign m_axi_awcache = 4'b0011;
    assign m_axi_awprot  = 3'b000;
    assign m_axi_awlock  = 1'b0;
    assign m_axi_awqos   = 4'b0000;
    assign m_axi_arsize  = 3'd4;
    assign m_axi_arburst = 2'd1;
    assign m_axi_arcache = 4'b0011;
    assign m_axi_arprot  = 3'b000;
    assign m_axi_arlock  = 1'b0;
    assign m_axi_arqos   = 4'b0000;
    assign m_axi_wstrb   = 16'hFFFF;
    assign m_axi_bready  = 1'b1;

    // rready is always 1 when we are ready to receive data (ST_RD_DATA or ST_RD_UNPACK)
    // This prevents HP0 from stalling waiting for rready
    reg rready_r;
    assign m_axi_rready = rready_r;

    localparam TGT_ACT = 2'd0;
    localparam TGT_WT  = 2'd1;
    localparam TGT_OUT = 2'd2;
    localparam TGT_DESC= 2'd3;   // Rev132: descriptor read → network_seq

    localparam ST_IDLE      = 4'd0;
    localparam ST_RD_ADDR   = 4'd1;
    localparam ST_RD_DATA   = 4'd2;   // receive beat, start unpacking
    localparam ST_RD_UNPACK = 4'd3;   // unpack remaining 15 bytes of beat
    localparam ST_WR_ADDR   = 4'd4;
    localparam ST_WR_FETCH  = 4'd5;
    localparam ST_WR_SEND   = 4'd6;
    localparam ST_WR_BRESP  = 4'd7;
    localparam ST_WR_WPRIME = 4'd8;   // wide write: 1-cycle wait for omem read latency
    localparam ST_WR_WSEND  = 4'd9;   // wide write: present 128-bit beat, wait wready

    reg [3:0]   st;
    reg [23:0]  tot_remain;
    reg [8:0]   beats_remain;  // beats left in current burst (9-bit: up to 256 beats/burst)
    reg [18:0]  bram_addr;
    reg [1:0]   tgt;
    reg [63:0]  ddr_addr;
    reg [3:0]   bidx;          // 0..15 byte within beat (4-bit)
    reg [127:0] beat_latch;
    reg [127:0] wdata_build;

    // ── BURST SIZING (Rev129): up to 4KB (256-beat) bursts with correct 4KB-boundary splitting.
    // AXI4 forbids a burst crossing a 4KB boundary, so cap each burst at min(remaining,
    // bytes-to-next-4KB-boundary, 4096). ddr_addr is always 16-byte aligned (engine streams 16B
    // beats from a 16B-aligned base), so to_4k is a multiple of 16 and the rounded-up beat count
    // never overruns the boundary. This replaces the old fixed 256B burst → 16× fewer bursts →
    // 16× less AR/AW latency overhead (the dominant term in HW-kick at ~82MB/s effective before).
    // The proven beat-streaming logic is UNCHANGED; only the AR/AW grouping grows.
    wire [12:0] to_4k    = 13'd4096 - {1'b0, ddr_addr[11:0]};      // bytes to next 4KB boundary (16..4096)
    wire [23:0] cap_4k   = {11'd0, to_4k};
    wire [23:0] bst_bytes = (tot_remain < cap_4k) ? tot_remain : cap_4k;  // <= 4096
    wire [7:0]  bst_len    = ((bst_bytes + 24'd15) >> 4) - 8'd1;   // ARLEN = beats-1 (max 255)
    wire [8:0]  bst_beats  = {1'b0, bst_len} + 9'd1;               // 1..256

    always @(posedge aclk) begin
        if (!aresetn) begin
            st            <= ST_IDLE;
            dma_active    <= 1'b0;
            dma_done      <= 1'b0;
            tot_remain    <= 24'd0;
            beats_remain  <= 9'd0;
            bram_addr     <= 19'd0;
            tgt           <= 2'd0;
            ddr_addr      <= 64'd0;
            bidx          <= 4'd0;
            beat_latch    <= 128'd0;
            wdata_build   <= 128'd0;
            m_axi_awaddr  <= 64'd0;
            m_axi_awlen   <= 8'd0;
            m_axi_awvalid <= 1'b0;
            m_axi_wdata   <= 128'd0;
            m_axi_wlast   <= 1'b0;
            m_axi_wvalid  <= 1'b0;
            m_axi_araddr  <= 64'd0;
            m_axi_arlen   <= 8'd0;
            m_axi_arvalid <= 1'b0;
            rready_r      <= 1'b0;
            dma_in_addr   <= 10'd0;
            dma_in_data   <= 8'd0;
            dma_in_wr_en  <= 1'b0;
            dma_in_w128_addr <= 6'd0;
            dma_in_w128_data <= 128'd0;
            dma_in_w128_en   <= 1'b0;
            dma_wt_addr   <= 12'd0;
            dma_wt_data   <= 8'd0;
            dma_wt_wr_en  <= 1'b0;
            dma_wt_w128_addr <= 13'd0;
            dma_wt_w128_data <= 128'd0;
            dma_wt_w128_en   <= 1'b0;
            dma_out_addr  <= 19'd0;
            dma_desc_w128_addr <= 3'd0;
            dma_desc_w128_data <= 128'd0;
            dma_desc_w128_en   <= 1'b0;
        end else begin
            dma_in_wr_en   <= 1'b0;
            dma_in_w128_en <= 1'b0;
            dma_wt_wr_en   <= 1'b0;
            dma_wt_w128_en <= 1'b0;
            dma_desc_w128_en <= 1'b0;
            dma_done     <= 1'b0;

            case (st)

                ST_IDLE: begin
                    rready_r <= 1'b0;
                    if (dma_req) begin
                        dma_active <= 1'b1;
                        tgt        <= dma_req_tgt;
                        ddr_addr   <= dma_req_addr;
                        tot_remain <= dma_req_bytes;
                        bram_addr  <= 19'd0;
                        dma_out_addr <= 19'd0;
                        bidx       <= 4'd0;
                        st <= (dma_req_tgt != TGT_OUT) ? ST_RD_ADDR : ST_WR_ADDR;
                    end
                end

                // ----------------------------------------------------------
                // READ PATH: issue AR, keep rready=1, latch beats
                // ----------------------------------------------------------

                ST_RD_ADDR: begin
                    m_axi_araddr   <= ddr_addr;
                    m_axi_arlen    <= bst_len;
                    m_axi_arvalid  <= 1'b1;
                    rready_r       <= 1'b1;    // keep high throughout burst
                    beats_remain   <= bst_beats;
                    bidx           <= 4'd0;
                    st             <= ST_RD_DATA;
                end

                // Wait for first (or next) beat; deassert arvalid once accepted
                ST_RD_DATA: begin
                    if (m_axi_arvalid && m_axi_arready)
                        m_axi_arvalid <= 1'b0;

                    if (m_axi_rvalid) begin
                        beats_remain <= beats_remain - 9'd1;
                        // WIDE for BOTH ACT and WT: write the whole 128-bit beat into one
                        // destination word this cycle (16× the byte path), keep rready high →
                        // stream the burst at full AXI rate. ST_RD_UNPACK now unused (reads only).
                        if (tot_remain > 24'd0) begin
                            if (tgt == TGT_ACT) begin
                                dma_in_w128_addr <= bram_addr[9:4];
                                dma_in_w128_data <= m_axi_rdata;
                                dma_in_w128_en   <= 1'b1;
                            end else if (tgt == TGT_DESC) begin   // Rev132: 4 beats → desc buffer
                                dma_desc_w128_addr <= bram_addr[6:4];
                                dma_desc_w128_data <= m_axi_rdata;
                                dma_desc_w128_en   <= 1'b1;
                            end else begin   // TGT_WT → WT-URAM wide word (within-chunk offset)
                                dma_wt_w128_addr <= {1'b0, bram_addr[15:4]};  // bram_addr is [15:0]; [16:4] reads x on bit16
                                dma_wt_w128_data <= m_axi_rdata;
                                dma_wt_w128_en   <= 1'b1;
                            end
                            bram_addr  <= bram_addr + 19'd16;
                            tot_remain <= (tot_remain > 24'd16) ? (tot_remain - 24'd16) : 24'd0;
                        end
                        ddr_addr <= ddr_addr + 64'd16;
                        if (tot_remain <= 24'd16) begin
                            dma_active <= 1'b0; dma_done <= 1'b1; rready_r <= 1'b0; st <= ST_IDLE;
                        end else if (beats_remain == 9'd1) begin
                            rready_r <= 1'b0; st <= ST_RD_ADDR;   // burst end → new AR
                        end else begin
                            rready_r <= 1'b1;                     // stream next beat
                        end
                    end
                end

                // Unpack bytes 1..15 of latched beat
                // rready=0 during unpack so HP0 holds next beat until we return to ST_RD_DATA
                ST_RD_UNPACK: begin
                    rready_r <= 1'b0;
                    if (tot_remain > 24'd0) begin
                        if (tgt == TGT_ACT) begin
                            dma_in_addr  <= bram_addr[9:0];
                            dma_in_data  <= beat_latch[{bidx, 3'b000} +: 8];
                            dma_in_wr_en <= 1'b1;
                        end else begin
                            dma_wt_addr  <= bram_addr[11:0];
                            dma_wt_data  <= beat_latch[{bidx, 3'b000} +: 8];
                            dma_wt_wr_en <= 1'b1;
                        end
                        bram_addr  <= bram_addr  + 19'd1;
                        tot_remain <= tot_remain - 24'd1;
                    end

                    if (bidx == 4'd15) begin
                        ddr_addr <= ddr_addr + 64'd16;
                        if (tot_remain <= 24'd1) begin
                            // Transfer done
                            dma_active <= 1'b0;
                            dma_done   <= 1'b1;
                            rready_r   <= 1'b0;
                            st         <= ST_IDLE;
                        end else if (beats_remain == 9'd0) begin
                            // End of burst — need new AR transaction
                            rready_r <= 1'b0;
                            st       <= ST_RD_ADDR;
                        end else begin
                            // More beats in this burst — re-enable rready for next beat
                            rready_r <= 1'b1;
                            bidx <= 4'd0;
                            st   <= ST_RD_DATA;
                        end
                    end else begin
                        bidx <= bidx + 4'd1;
                    end
                end

                // ----------------------------------------------------------
                // WRITE PATH: pack 16 bytes from BRAM, send 1 beat
                // ----------------------------------------------------------

                ST_WR_ADDR: begin
                    m_axi_awaddr  <= ddr_addr;
                    m_axi_awlen   <= bst_len;
                    m_axi_awvalid <= 1'b1;
                    beats_remain  <= bst_beats;
                    bidx          <= 4'd0;
                    wdata_build   <= 128'd0;
                    dma_out_addr  <= bram_addr[18:0];   // word addr [9:4] for wide, byte 0 for byte-path
                    st            <= out_wide ? ST_WR_WPRIME : ST_WR_FETCH;
                end

                // ── WIDE write: 1 word (128-bit) per beat. 2 cyc/beat (prime+send), robust no-skid. ──
                // dma_out_addr was set last cycle → dma_out_data128 (=omem[word]) valid now.
                ST_WR_WPRIME: begin
                    if (m_axi_awvalid && m_axi_awready) m_axi_awvalid <= 1'b0;
                    st <= ST_WR_WSEND;
                end
                ST_WR_WSEND: begin
                    if (m_axi_awvalid && m_axi_awready) m_axi_awvalid <= 1'b0;
                    if (!m_axi_wvalid) begin
                        m_axi_wdata  <= dma_out_data128;   // full 128-bit omem word
                        m_axi_wvalid <= 1'b1;
                        m_axi_wlast  <= (beats_remain == 9'd1) ? 1'b1 : 1'b0;
                    end
                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        beats_remain <= beats_remain - 9'd1;
                        ddr_addr     <= ddr_addr  + 64'd16;
                        bram_addr    <= bram_addr + 19'd16;
                        tot_remain   <= (tot_remain >= 24'd16) ? tot_remain - 24'd16 : 24'd0;
                        if (m_axi_wlast) begin
                            m_axi_wlast <= 1'b0; st <= ST_WR_BRESP;
                        end else begin
                            dma_out_addr <= bram_addr[18:0] + 19'd16;  // issue next word read
                            st <= ST_WR_WPRIME;                       // 1 cyc for omem latency
                        end
                    end
                end

                // Collect 16 bytes from BRAM (1 cycle latency each)
                // bidx: which byte we are waiting for data (addr issued previous cycle)
                // On bidx=0: addr=base issued, wait data next cycle
                // On bidx=1..15: collect data[bidx-1], issue addr[bidx]
                // After bidx=15: move to ST_WR_SEND to collect data[15]
                ST_WR_FETCH: begin
                    if (m_axi_awvalid && m_axi_awready)
                        m_axi_awvalid <= 1'b0;

                    if (bidx > 4'd0) begin
                        wdata_build[{(bidx - 4'd1), 3'b000} +: 8] <= dma_out_data;
                    end

                    if (bidx == 4'd15) begin
                        st <= ST_WR_SEND;
                    end else begin
                        bidx         <= bidx + 4'd1;
                        dma_out_addr <= bram_addr[18:0] + {15'd0, bidx} + 19'd1;
                    end
                end

                ST_WR_SEND: begin
                    // Collect last byte, assemble full 128-bit word
                    wdata_build[127:120] <= dma_out_data;

                    if (!m_axi_wvalid) begin
                        m_axi_wdata  <= {dma_out_data,
                                         wdata_build[119:0]};  // byte15 | bytes 0..14
                        m_axi_wvalid <= 1'b1;
                        beats_remain <= beats_remain - 9'd1;
                        m_axi_wlast  <= (beats_remain == 9'd1) ? 1'b1 : 1'b0;

                        ddr_addr   <= ddr_addr  + 64'd16;
                        bram_addr  <= bram_addr + 19'd16;
                        tot_remain <= (tot_remain >= 24'd16) ? tot_remain - 24'd16 : 24'd0;
                    end

                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 1'b0;
                        if (m_axi_wlast) begin
                            m_axi_wlast <= 1'b0;
                            st <= ST_WR_BRESP;
                        end else begin
                            // More beats in burst
                            bidx         <= 4'd0;
                            wdata_build  <= 128'd0;
                            dma_out_addr <= bram_addr[18:0]; // bram_addr already +16
                            st           <= ST_WR_FETCH;
                        end
                    end
                end

                ST_WR_BRESP: begin
                    if (m_axi_bvalid) begin
                        if (tot_remain == 24'd0) begin
                            dma_active <= 1'b0;
                            dma_done   <= 1'b1;
                            st         <= ST_IDLE;
                        end else begin
                            bidx        <= 4'd0;
                            wdata_build <= 128'd0;
                            dma_out_addr <= bram_addr[18:0];
                            st          <= ST_WR_ADDR;
                        end
                    end
                end

                default: st <= ST_IDLE;
            endcase
        end
    end

endmodule
