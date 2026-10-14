// axi_lite_slave.v  --  AXI4-Lite slave register file  (CLAUDE.md §8.17)
// Frozen register map: conv1x1 base=0xa0000000, conv3x3=0xa0010000, dwconv=0xa0020000
//
// Protocol fixes vs old HLS version (DECERR root cause):
//  - Accept write only after BOTH AWVALID and WVALID are seen
//  - Hold BVALID until BREADY seen (HLS bug: released too early)
//  - SLVERR on invalid address (prevents bus hang)
//  - ap_start is W1SC: cleared one cycle after write
//  - ap_isr is W1C
//
// This module is parameterised for the three IP variants:
//   IP_TYPE=0: conv1x1 (regs 0x00..0x44)
//   IP_TYPE=1: conv3x3 (regs 0x00..0x4C, adds stride+pad)
//   IP_TYPE=2: dwconv3x3 (same as conv3x3, but Cout=C implicit)
//
// Outputs: registered config values + ap_start pulse + ap_done/idle status inputs.

`timescale 1ns/1ps
`include "cnn_defines.vh"

module axi_lite_slave #(
    parameter IP_TYPE = 0   // 0=conv1x1, 1=conv3x3, 2=dwconv3x3
) (
    input  wire        aclk,
    input  wire        aresetn,    // active-low synchronous reset

    // AXI4-Lite write address channel
    input  wire [8:0]  awaddr,
    input  wire        awvalid,
    output reg         awready,

    // AXI4-Lite write data channel
    input  wire [31:0] wdata,
    input  wire [3:0]  wstrb,      // byte enable (used for 4-byte aligned writes)
    input  wire        wvalid,
    output reg         wready,

    // AXI4-Lite write response channel
    output reg  [1:0]  bresp,
    output reg         bvalid,
    input  wire        bready,

    // AXI4-Lite read address channel
    input  wire [8:0]  araddr,
    input  wire        arvalid,
    output reg         arready,

    // AXI4-Lite read data channel
    output reg  [31:0] rdata,
    output reg  [1:0]  rresp,
    output reg         rvalid,
    input  wire        rready,

    // ── IP status inputs (driven by top-level FSM) ───────────────────────────
    input  wire        ap_done_in,   // IP signals completion
    input  wire        ap_idle_in,   // IP signals idle (ready for next ap_start)
    input  wire [3:0]  dbg_seq_state_in,    // debug: dma_sequencer state (readable at 0x170)
    input  wire [3:0]  dbg_kernel_state_in, // debug: unified_acc_top FSM state (readable at 0x180)

    // ── Config register outputs (to IP datapath) ─────────────────────────────
    output reg  [31:0] reg_cin,
    output reg  [31:0] reg_cout,
    output reg  [31:0] reg_hin,
    output reg  [31:0] reg_win,
    output reg  [4:0]  reg_n_config,  // 5-bit (Rev139): c3 dense = 16 entries/M-block (4-bit max 15)
    output reg  [1:0]  reg_poc,
    output reg  [1:0]  reg_accu_mode,
    output reg         reg_apply_relu,
    output reg  [63:0] reg_wt_ddr_addr,
    output reg  [63:0] reg_in_ddr_addr,
    output reg  [63:0] reg_out_ddr_addr,
    output reg  [31:0] reg_stride,    // conv3x3/dwconv only
    output reg  [31:0] reg_pad,       // conv3x3/dwconv only
    output reg  [3:0]  reg_conv_type, // 4-bit: LT0-15 (was [2:0] -> LT8-11 ALIASED to LT0-3! 3'd11==3==conv1x1)
    output reg  [23:0] reg_out_bytes, // SW-computed Cout*Hout*Wout, written before ap_start
    output reg  [23:0] reg_wt_bytes,  // weight blob byte count, written before ap_start

    // ── B0 layer_sequencer config (on-chip layer loop) ──────────────────────────
    output reg         reg_layer_mode,
    output reg  [14:0] reg_feat_en,   // feature enable: bits[15:1] of AXI_OFF_LAYER_MODE (0x190)
                                      // [0]=B-i [1]=c3 [2]=fastkick [3]=C3W wide-c3 [4]=W11 wide-1x1
    output reg  [7:0]  reg_n_ocb,
    output reg  [15:0] reg_n_pos,
    output reg  [7:0]  reg_n_icb,     // 8-bit: Cin up to 4096 (was 5-bit→Cin=512 overflow)
    output reg  [23:0] reg_wt_blk_bytes,
    output reg  [23:0] reg_act_chunk,
    // ── B-c3 9-pass (Rev139): write-side của 0x170/0x180 (read vẫn là debug state) ──
    output reg  [31:0] reg_c3_cfg,    // 0x170: [8:0]=W_in [12:9]=ncolg [16:13]=nicc [24:17]=nrow [25]=stride2
    output reg  [15:0] reg_act_total, // 0x180: số chunk LS_ACTL cần DMA (l_nact)

    // ── Control signals ───────────────────────────────────────────────────────
    output reg         ap_start,     // W1SC: pulses 1 cycle when ARM writes ap_ctrl[0]
    output wire        irq_out,      // interrupt to ARM (ap_done && GIE && IER)

    // ── Đòn 4b: cổng ghi thứ 2 từ network_seq_v2 ("host on PL") ─────────────────
    // Dùng CHUNG decode với AXI → mọi consumer không đổi. Protocol: netseq chỉ ghi
    // khi host quiesced (host kick 1 lần/segment rồi poll) — không tranh chấp.
    input  wire        ns_wr_en,
    input  wire [8:0]  ns_wr_addr,
    input  wire [31:0] ns_wr_data,
    input  wire        ns_err_in,     // Rev145 debug
    input  wire [3:0]  ns_dbg_in
);

    // ── Interrupt registers ───────────────────────────────────────────────────
    reg        reg_gie;   // global interrupt enable
    reg        reg_ier;   // ap_done interrupt enable
    reg        reg_isr;   // interrupt status (W1C), set by ap_done_in

    assign irq_out = reg_gie & reg_ier & reg_isr;

    // ap_done edge detection — reg_isr driven solely in write FSM below
    reg ap_done_prev;
    wire ap_done_rise = ap_done_in && !ap_done_prev;
    always @(posedge aclk) begin
        if (!aresetn) ap_done_prev <= 1'b0;
        else          ap_done_prev <= ap_done_in;
    end

    // ── Write FSM ─────────────────────────────────────────────────────────────
    // States: wait for AWVALID+WVALID → latch addr+data → write reg → send BRESP
    localparam WR_IDLE   = 2'd0;  // waiting for AWVALID or WVALID
    localparam WR_ADDR   = 2'd1;  // have addr, waiting for data
    localparam WR_DATA   = 2'd2;  // have data, waiting for addr (rare)
    localparam WR_EXEC   = 3'd3;  // both latched, execute write + send BRESP

    // Use 3-bit state to accommodate WR_EXEC=3
    reg [2:0]  wr_state;
    reg [8:0]  wr_addr_lat;
    reg [31:0] wr_data_lat;
    reg [1:0]  wr_resp_next;  // blocking-assigned to capture before bresp latch

    always @(posedge aclk) begin
        if (!aresetn) begin
            wr_state   <= WR_IDLE;
            awready    <= 1'b0;
            wready     <= 1'b0;
            bvalid     <= 1'b0;
            bresp      <= 2'b00;
            ap_start   <= 1'b0;
            // Config defaults
            reg_cin         <= 32'd0; reg_cout        <= 32'd0;
            reg_hin         <= 32'd0; reg_win         <= 32'd0;
            reg_n_config    <= 5'd2;  reg_poc         <= 2'd0;
            reg_c3_cfg      <= 32'd0; reg_act_total   <= 16'd0;
            reg_accu_mode   <= 2'd0;  reg_apply_relu  <= 1'b0;
            reg_wt_ddr_addr <= 64'd0; reg_in_ddr_addr <= 64'd0;
            reg_out_ddr_addr<= 64'd0; reg_stride      <= 32'd1;
            reg_pad         <= 32'd0; reg_conv_type   <= 4'd0;
            reg_out_bytes   <= 24'd0;
            reg_wt_bytes    <= 24'd0;
            reg_layer_mode  <= 1'b0;  reg_feat_en <= 15'd0; reg_n_ocb <= 8'd0; reg_n_pos <= 16'd0;
            reg_n_icb       <= 8'd0;  reg_wt_blk_bytes <= 24'd0; reg_act_chunk <= 24'd1024;
            reg_gie <= 1'b0; reg_ier <= 1'b0; reg_isr <= 1'b0;
        end else begin
            ap_start <= 1'b0;  // default W1SC clear
            awready  <= 1'b0;  // default deassert (pulsed for 1 cycle)
            wready   <= 1'b0;

            case (wr_state)
                WR_IDLE: begin
                    if (awvalid && wvalid) begin
                        // Both channels presented simultaneously
                        awready     <= 1'b1;
                        wready      <= 1'b1;
                        wr_addr_lat <= awaddr;
                        wr_data_lat <= wdata;
                        wr_state    <= WR_EXEC;
                    end else if (awvalid) begin
                        // Address channel first
                        awready     <= 1'b1;
                        wr_addr_lat <= awaddr;
                        wr_state    <= WR_ADDR;
                    end else if (wvalid) begin
                        // Data channel first (unusual but legal)
                        wready      <= 1'b1;
                        wr_data_lat <= wdata;
                        wr_state    <= WR_DATA;
                    end
                end

                WR_ADDR: begin
                    // Have address, waiting for write data
                    if (wvalid) begin
                        wready      <= 1'b1;
                        wr_data_lat <= wdata;
                        wr_state    <= WR_EXEC;
                    end
                end

                WR_DATA: begin
                    // Have data, waiting for write address
                    if (awvalid) begin
                        awready     <= 1'b1;
                        wr_addr_lat <= awaddr;
                        wr_state    <= WR_EXEC;
                    end
                end

                WR_EXEC: begin
                    // Both addr and data latched — decode (shared, dưới) chạy cùng nhịp
                    if (!bvalid) begin
                        bvalid   <= 1'b1;
                        bresp    <= wr_resp_next;
                    end
                    // Hold BVALID until BREADY
                    if (bvalid && bready) begin
                        bvalid   <= 1'b0;
                        wr_state <= WR_IDLE;
                    end
                end

                default: wr_state <= WR_IDLE;
            endcase

            // ── SHARED WRITE DECODE (Đòn 4b): AXI (WR_EXEC nhịp đầu) hoặc netseq (ns_wr) ──
            // ns thắng nếu trùng nhịp (protocol cấm trùng — host quiesced khi netseq chạy).
            begin : wr_decode
                reg        wx_en;
                reg [8:0]  wx_addr;
                reg [31:0] wx_data;
                wx_en   = ns_wr_en || (wr_state == WR_EXEC && !bvalid);
                wx_addr = ns_wr_en ? ns_wr_addr : wr_addr_lat;
                wx_data = ns_wr_en ? ns_wr_data : wr_data_lat;
                if (wx_en) begin
                        wr_resp_next = 2'b00;  // default OKAY (blocking, overridden below if SLVERR)
                        case (wx_addr)
                            `AXI_OFF_AP_CTRL: begin
                                if (wx_data[0]) ap_start <= 1'b1;  // W1SC
                            end
                            `AXI_OFF_AP_GIE:  reg_gie         <= wx_data[0];
                            `AXI_OFF_AP_IER:  reg_ier         <= wx_data[0];
                            `AXI_OFF_AP_ISR:  begin
                                // W1C: clear bits where write data is 1
                                if (wx_data[0]) reg_isr <= 1'b0;
                            end
                            `AXI_OFF_CIN:     reg_cin         <= wx_data;
                            `AXI_OFF_COUT:    reg_cout        <= wx_data;
                            `AXI_OFF_HIN:     reg_hin         <= wx_data;
                            `AXI_OFF_WIN:     reg_win         <= wx_data;
                            `AXI_OFF_N_CONFIG: reg_n_config   <= wx_data[4:0];
                            `AXI_OFF_POC:     reg_poc         <= wx_data[1:0];
                            `AXI_OFF_ACCU_MODE: reg_accu_mode <= wx_data[1:0];
                            `AXI_OFF_APPLY_RELU: reg_apply_relu <= wx_data[0];
                            `AXI_OFF_WT_ADDR_LO: reg_wt_ddr_addr[31:0]  <= wx_data;
                            `AXI_OFF_WT_ADDR_HI: reg_wt_ddr_addr[63:32] <= wx_data;
                            `AXI_OFF_IN_ADDR_LO: reg_in_ddr_addr[31:0]  <= wx_data;
                            `AXI_OFF_IN_ADDR_HI: reg_in_ddr_addr[63:32] <= wx_data;
                            `AXI_OFF_OUT_ADDR_LO: reg_out_ddr_addr[31:0]  <= wx_data;
                            `AXI_OFF_OUT_ADDR_HI: reg_out_ddr_addr[63:32] <= wx_data;
                            `AXI_OFF_STRIDE:  if (IP_TYPE != 0) reg_stride <= wx_data;
                                              else wr_resp_next = 2'b10;  // SLVERR for conv1x1
                            `AXI_OFF_PAD:     if (IP_TYPE != 0) reg_pad    <= wx_data;
                                              else wr_resp_next = 2'b10;
                            `AXI_OFF_LAYER_TYPE: reg_conv_type <= wx_data[3:0];
                            `AXI_OFF_OUT_BYTES:  reg_out_bytes <= wx_data[23:0];
                            `AXI_OFF_WT_BYTES:   reg_wt_bytes  <= wx_data[23:0];
                            `AXI_OFF_LAYER_MODE: begin reg_layer_mode<= wx_data[0]; reg_feat_en <= wx_data[15:1]; end
                            `AXI_OFF_N_OCB:      reg_n_ocb     <= wx_data[7:0];
                            `AXI_OFF_N_POS:      reg_n_pos     <= wx_data[15:0];
                            `AXI_OFF_N_ICB:      reg_n_icb     <= wx_data[7:0];
                            `AXI_OFF_WT_BLK_BYTES: reg_wt_blk_bytes <= wx_data[23:0];
                            `AXI_OFF_ACT_CHUNK:  reg_act_chunk <= wx_data[23:0];
                            `AXI_OFF_C3_CFG:     reg_c3_cfg    <= wx_data;
                            `AXI_OFF_ACT_TOTAL:  reg_act_total <= wx_data[15:0];
                            default: wr_resp_next = 2'b10;  // SLVERR: invalid address
                        endcase
                end
            end
            // ap_done rising edge sets ISR — placed after case so set wins over W1C clear
            if (ap_done_rise) reg_isr <= 1'b1;
        end
    end

    // ── Read FSM ──────────────────────────────────────────────────────────────
    localparam RD_IDLE = 1'b0;
    localparam RD_DATA = 1'b1;

    reg rd_state;

    always @(posedge aclk) begin
        if (!aresetn) begin
            rd_state <= RD_IDLE;
            arready  <= 1'b0;
            rvalid   <= 1'b0;
            rdata    <= 32'h0;
            rresp    <= 2'b00;
        end else begin
            case (rd_state)
                RD_IDLE: begin
                    arready <= 1'b0;
                    rvalid  <= 1'b0;
                    if (arvalid) begin
                        arready <= 1'b1;
                        // Decode read address
                        rresp <= 2'b00;
                        case (araddr)
                            `AXI_OFF_AP_CTRL: rdata <= {29'd0, ap_idle_in, ap_done_in, 1'b0};
                            `AXI_OFF_AP_GIE:  rdata <= {31'd0, reg_gie};
                            `AXI_OFF_AP_IER:  rdata <= {31'd0, reg_ier};
                            `AXI_OFF_AP_ISR:  rdata <= {31'd0, reg_isr};
                            `AXI_OFF_CIN:     rdata <= reg_cin;
                            `AXI_OFF_COUT:    rdata <= reg_cout;
                            `AXI_OFF_HIN:     rdata <= reg_hin;
                            `AXI_OFF_WIN:     rdata <= reg_win;
                            `AXI_OFF_N_CONFIG: rdata <= {27'd0, reg_n_config};
                            `AXI_OFF_POC:     rdata <= {30'd0, reg_poc};
                            `AXI_OFF_ACCU_MODE: rdata <= {30'd0, reg_accu_mode};
                            `AXI_OFF_APPLY_RELU: rdata <= {31'd0, reg_apply_relu};
                            `AXI_OFF_WT_ADDR_LO: rdata <= reg_wt_ddr_addr[31:0];
                            `AXI_OFF_WT_ADDR_HI: rdata <= reg_wt_ddr_addr[63:32];
                            `AXI_OFF_IN_ADDR_LO: rdata <= reg_in_ddr_addr[31:0];
                            `AXI_OFF_IN_ADDR_HI: rdata <= reg_in_ddr_addr[63:32];
                            `AXI_OFF_OUT_ADDR_LO: rdata <= reg_out_ddr_addr[31:0];
                            `AXI_OFF_OUT_ADDR_HI: rdata <= reg_out_ddr_addr[63:32];
                            `AXI_OFF_STRIDE:  rdata <= reg_stride;
                            `AXI_OFF_PAD:     rdata <= reg_pad;
                            `AXI_OFF_LAYER_TYPE: rdata <= {28'd0, reg_conv_type};
                            `AXI_OFF_OUT_BYTES:  rdata <= {8'd0, reg_out_bytes};
                            `AXI_OFF_WT_BYTES:   rdata <= {8'd0, reg_wt_bytes};
                            9'h170:              rdata <= {28'd0, dbg_seq_state_in};
                            9'h180:              rdata <= {28'd0, dbg_kernel_state_in};
                            `AXI_OFF_LAYER_MODE: rdata <= {ns_err_in, ns_dbg_in, 11'd0, reg_feat_en, reg_layer_mode};
                            `AXI_OFF_N_OCB:      rdata <= {24'd0, reg_n_ocb};
                            `AXI_OFF_N_POS:      rdata <= {16'd0, reg_n_pos};
                            `AXI_OFF_N_ICB:      rdata <= {24'd0, reg_n_icb};
                            `AXI_OFF_WT_BLK_BYTES: rdata <= {8'd0, reg_wt_blk_bytes};
                            `AXI_OFF_ACT_CHUNK:  rdata <= {8'd0, reg_act_chunk};
                            default: begin rdata <= 32'hDEAD_BEEF; rresp <= 2'b10; end
                        endcase
                        rvalid   <= 1'b1;
                        rd_state <= RD_DATA;
                    end
                end

                RD_DATA: begin
                    arready <= 1'b0;
                    if (rready) begin
                        rvalid   <= 1'b0;
                        rd_state <= RD_IDLE;
                    end
                end

                default: rd_state <= RD_IDLE;
            endcase
        end
    end

endmodule
