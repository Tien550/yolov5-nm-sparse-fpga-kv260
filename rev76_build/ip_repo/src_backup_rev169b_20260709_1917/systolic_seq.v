// systolic_seq.v -- Option 2 P4-C: orchestrator for conv1x1_systolic_top on the board.
// Mirrors dma_sequencer's handshake with dma_engine, but the flow is:
//   load A (fmap) -> amem  |  load W (weights) -> wmem  |  load P (requant params) -> pmem
//   -> kick systolic engine -> wait done -> writeback O (omem) -> DDR -> irq.
// DMA targets: ACT(0) routes to amem, WT(1) to wmem, PARAM(3) to pmem, OUT(2) reads omem.
//
// Each load/store state uses an `issued` flag so it advances only on the dma_done of ITS OWN
// request — NOT on a lingering dma_done pulse from the previous transfer (that bug let compute
// overlap the param load).
`timescale 1ns/1ps

module systolic_seq (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        ap_start,
    input  wire        skip_act,        // Dir-B: amem pre-filled on-chip (rebank/chain) -> skip ACT DMA
    input  wire [63:0] act_addr,
    input  wire [63:0] wt_addr,
    input  wire [63:0] par_addr,
    input  wire [63:0] out_addr,
    input  wire [23:0] act_bytes,
    input  wire [23:0] wt_bytes,
    input  wire [23:0] par_bytes,
    input  wire [23:0] out_bytes,
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,
    output reg         sys_start,
    input  wire        sys_done,
    output reg         busy,
    output reg         irq,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2, TGT_PAR=2'd3;
    localparam S_IDLE=4'd0, S_LDA=4'd1, S_LDW=4'd2, S_LDP=4'd3,
               S_KICK=4'd4, S_RUN=4'd5, S_OUT=4'd6, S_DONE=4'd7;
    reg [3:0] st;
    reg       ap_seen, issued;
    reg [23:0] out_rem, out_off;            // OUT chunking: dma_out_addr is 10-bit => <=1024B/chunk
    assign dbg_state = st;

    // helper: in a DMA state, issue once then advance on its own dma_done.
    // (expanded inline per-state for clarity)
    always @(posedge aclk) begin
        if (!aresetn) begin
            st<=S_IDLE; dma_req<=0; sys_start<=0; busy<=0; irq<=0; ap_seen<=0; issued<=0;
        end else begin
            dma_req<=0; sys_start<=0; irq<=0;
            case (st)
                S_IDLE: begin
                    busy<=0; issued<=0;
                    if (ap_start && !ap_seen) begin ap_seen<=1; busy<=1; st<=S_LDA; end
                    if (!ap_start) ap_seen<=0;
                end
                S_LDA: begin
                    if (skip_act) st<=S_LDW;            // amem pre-filled on-chip -> no ACT DMA
                    else if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_ACT;
                            dma_req_addr<=act_addr; dma_req_bytes<=act_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_LDW; end
                end
                S_LDW: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_WT;
                            dma_req_addr<=wt_addr; dma_req_bytes<=wt_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_LDP; end
                end
                S_LDP: begin
                    if (par_bytes==24'd0) st<=S_KICK;
                    else if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_PAR;
                            dma_req_addr<=par_addr; dma_req_bytes<=par_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_KICK; end
                end
                S_KICK: begin sys_start<=1; out_rem<=out_bytes; out_off<=24'd0; st<=S_RUN; end
                S_RUN:  if (sys_done) st<=S_OUT;
                S_OUT: begin                              // chunked: <=1024B per OUT DMA
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_OUT;
                            dma_req_addr<=out_addr + {40'd0, out_off};
                            dma_req_bytes<=(out_rem > 24'd1024) ? 24'd1024 : out_rem;
                            issued<=1;
                        end
                    end else if (dma_done) begin
                        issued<=0;
                        if (out_rem > 24'd1024) begin
                            out_rem<=out_rem-24'd1024; out_off<=out_off+24'd1024;  // next chunk
                        end else st<=S_DONE;
                    end
                end
                S_DONE: begin irq<=1; busy<=0; ap_seen<=0; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
