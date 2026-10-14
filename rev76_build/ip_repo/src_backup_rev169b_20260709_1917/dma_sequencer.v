// dma_sequencer.v -- DMA Sequencer FSM for cnn_acc_top
// Orchestrates the order: ACT load -> WT load -> notify kernel -> wait -> OUT writeback
// Optionally prefetches next ACT tile while kernel is computing.
//
// Handshake with dma_engine:
//   dma_req / dma_req_tgt / dma_req_addr / dma_req_bytes  -> request to engine
//   dma_active / dma_done                                  <- status from engine

`timescale 1ns/1ps

module dma_sequencer (
    input  wire        aclk,
    input  wire        aresetn,

    // From unified_acc_top config registers
    input  wire        ap_start,
    input  wire        irq_in,      // kept for compatibility (unused — use ap_done_in instead)
    input  wire        ap_done_in,  // kernel done signal directly from unified_acc_top
    input  wire        dma_nxt_req,
    input  wire [31:0] reg_cin,
    input  wire [31:0] reg_hin,
    input  wire [31:0] reg_win,
    input  wire [3:0]  reg_ncfg,
    input  wire [23:0] reg_out_bytes,
    input  wire [23:0] reg_wt_bytes,
    input  wire [63:0] reg_in_addr,
    input  wire [63:0] reg_wt_addr,
    input  wire [63:0] reg_out_addr,

    // To dma_engine: issue a transfer request
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,

    // From dma_engine: transfer status
    input  wire        dma_active,
    input  wire        dma_done,

    // To unified_acc_top: signal that ACT+WT are loaded
    output reg         dma_in_done,

    // Debug: expose sequencer state for AXI-readable register
    output wire [3:0]  dbg_seq_state
);

    localparam DMA_TGT_ACT = 2'd0;
    localparam DMA_TGT_WT  = 2'd1;
    localparam DMA_TGT_OUT = 2'd2;

    localparam SEQ_IDLE     = 4'd0;
    localparam SEQ_ACT      = 4'd1;
    localparam SEQ_WT       = 4'd2;
    localparam SEQ_NOTIFY   = 4'd3;
    localparam SEQ_WAIT     = 4'd4;
    localparam SEQ_OUT_RD   = 4'd5;
    localparam SEQ_OUT_WR   = 4'd6;
    localparam SEQ_DONE     = 4'd7;
    localparam SEQ_PREFETCH = 4'd8;

    reg [3:0]  seq_state;
    assign dbg_seq_state = seq_state;
    reg [23:0] act_bytes;
    reg [23:0] wt_bytes;
    reg [23:0] out_bytes;
    reg        ap_start_seen;

    // 32-bit intermediate for act_bytes = HIN * WIN * CIN (avoid 16-bit overflow)
    wire [31:0] act_hw   = {16'd0, reg_hin[15:0]} * {16'd0, reg_win[15:0]};
    wire [31:0] act_bytes_full = act_hw * {16'd0, reg_cin[15:0]};

    always @(posedge aclk) begin
        if (!aresetn) begin
            seq_state     <= SEQ_IDLE;
            dma_req       <= 1'b0;
            dma_in_done   <= 1'b0;
            ap_start_seen <= 1'b0;
        end else begin
            dma_req     <= 1'b0;
            dma_in_done <= 1'b0;

            case (seq_state)
                SEQ_IDLE: begin
                    if (ap_start && !ap_start_seen) begin
                        ap_start_seen <= 1'b1;
                        act_bytes  <= act_bytes_full[23:0];
                        wt_bytes   <= reg_wt_bytes;
                        out_bytes  <= reg_out_bytes;
                        seq_state  <= SEQ_ACT;
                    end
                    if (!ap_start) ap_start_seen <= 1'b0;
                end

                SEQ_ACT: begin
                    if (!dma_active && !dma_req) begin
                        dma_req       <= 1'b1;
                        dma_req_tgt   <= DMA_TGT_ACT;
                        dma_req_addr  <= reg_in_addr;
                        dma_req_bytes <= act_bytes;
                    end
                    if (dma_done) seq_state <= SEQ_WT;
                end

                SEQ_WT: begin
                    if (!dma_active && !dma_req) begin
                        dma_req       <= 1'b1;
                        dma_req_tgt   <= DMA_TGT_WT;
                        dma_req_addr  <= reg_wt_addr;
                        dma_req_bytes <= wt_bytes;
                    end
                    if (dma_done) seq_state <= SEQ_NOTIFY;
                end

                SEQ_NOTIFY: begin
                    dma_in_done <= 1'b1;
                    seq_state   <= SEQ_WAIT;
                end

                SEQ_WAIT: begin
                    dma_in_done <= 1'b1;  // hold HIGH so kernel ST_WAIT_DMA can detect it
                    if (ap_start) begin
                        // SW re-asserted ap_start — force restart
                        ap_start_seen <= 1'b0;
                        dma_in_done   <= 1'b0;
                        seq_state     <= SEQ_IDLE;
                    end else begin
                        if (dma_nxt_req && !dma_active && !dma_req) begin
                            dma_req       <= 1'b1;
                            dma_req_tgt   <= DMA_TGT_ACT;
                            dma_req_addr  <= reg_in_addr;
                            dma_req_bytes <= act_bytes;
                            seq_state     <= SEQ_PREFETCH;
                        end
                        if (ap_done_in) begin
                            dma_in_done <= 1'b0;
                            seq_state   <= SEQ_OUT_RD;
                        end
                    end
                end

                SEQ_PREFETCH: begin
                    if (dma_done)                        seq_state <= SEQ_WAIT;
                    if (ap_done_in && !dma_active)       seq_state <= SEQ_OUT_RD;
                end

                SEQ_OUT_RD: begin
                    if (!dma_active && !dma_req) begin
                        dma_req       <= 1'b1;
                        dma_req_tgt   <= DMA_TGT_OUT;
                        dma_req_addr  <= reg_out_addr;
                        dma_req_bytes <= out_bytes;
                    end
                    if (dma_done) seq_state <= SEQ_DONE;
                end

                SEQ_DONE: begin
                    ap_start_seen <= 1'b0;
                    seq_state     <= SEQ_IDLE;
                end

                default: seq_state <= SEQ_IDLE;
            endcase
        end
    end

endmodule
