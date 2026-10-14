// network_seq.v -- Network Sequencer (LAYER_TYPE=14, is_netseq) -- DESCRIPTOR MODE (Rev132).
// Runs n_layers conv1x1 layers under ONE ap_start, NO host round-trip between them.
// Layer i+1's activation = layer i's output (DDR roundtrip, host never touches it).
//
// WHY DESCRIPTOR (not AXI registers): the PoC register-based version (Rev131) died — this design's
// AXI path only acknowledges 16-byte-aligned register slots in the existing ≤0x1E0 range, and that
// 9-bit space is FULL. So per-layer config is read from a DDR DESCRIPTOR TABLE instead: host writes
// the table to DDR + its base addr to ONE existing register (reg_in_ddr_addr) + n_layers to reg_cin.
// This adds ZERO new AXI registers and scales to any N. See project_netseq_axi_addr_wall.
//
// DESCRIPTOR = 64 bytes/layer = 4 beats of 128-bit (dma_engine TGT_DESC=3). Layout (w0..w15, 32b):
//   w0=cin w1=cout w2=hw w3=act_bytes w4=wt_bytes w5=out_bytes
//   w6/w7=act_addr lo/hi  w8/w9=wt_addr lo/hi  w10/w11=out_addr lo/hi  (w12-15 reserved)
//   beat0={w3,w2,w1,w0} beat1={w7,w6,w5,w4} beat2={w11,w10,w9,w8}
//
// NO-RACE: the single shared dma_engine processes one request at a time, and this FSM waits for
// each op's dma_done before advancing (the `issued` discipline). So layer i's OUT-DMA fully
// completes before layer i+1's DESC/ACT-DMA is issued. Race-free by construction.
//
// cnn_acc_top muxes the ENGINE config (sys_cin/cout/hw/wb) on this module's cur_cin/cout/hw outputs.
`timescale 1ns/1ps

module network_seq (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        ap_start,

    // ── network config (from existing registers, reused during is_netseq) ──
    input  wire [63:0] desc_base,            // DDR base of descriptor table (= reg_in_ddr_addr)
    input  wire [3:0]  n_layers,             // number of layers (= reg_cin[3:0])

    // ── DMA orchestration (shared dma_engine, identical handshake to systolic_seq) ──
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    // ── descriptor buffer write-back from dma_engine (TGT_DESC, 4×128-bit beats) ──
    input  wire        dma_desc_w128_en,
    input  wire [1:0]  dma_desc_w128_addr,
    input  wire [127:0] dma_desc_w128_data,

    // ── engine kick ──
    output reg         sys_start,
    input  wire        sys_done,

    // ── current-layer dims → cnn_acc_top muxes engine config (sys_cin/cout/hw/wb) ──
    output reg  [13:0] cur_cin,
    output reg  [9:0]  cur_cout,
    output reg  [9:0]  cur_hw,
    output reg  [3:0]  cur_layer,

    output reg         busy,
    output reg         done_latch,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2, TGT_DESC=2'd3;
    localparam S_IDLE=4'd0, S_LDDESC=4'd1, S_LDA=4'd2, S_LDW=4'd3, S_KICK=4'd4,
               S_RUN=4'd5, S_OUT=4'd6, S_NEXT=4'd7, S_DONE=4'd8;
    reg [3:0] st;
    reg       ap_seen, issued;
    reg [23:0] out_rem, out_off;            // OUT chunking: ≤1024B/chunk
    assign dbg_state = st;

    // descriptor buffer (3 used beats of 4)
    reg [127:0] desc0, desc1, desc2;
    always @(posedge aclk) begin
        if (dma_desc_w128_en) begin
            case (dma_desc_w128_addr)
                2'd0: desc0 <= dma_desc_w128_data;
                2'd1: desc1 <= dma_desc_w128_data;
                2'd2: desc2 <= dma_desc_w128_data;
                default: ;   // beat3 reserved
            endcase
        end
    end
    // parsed fields (combinational view of desc buffer)
    wire [23:0] p_act_bytes = desc0[127:96];   // w3
    wire [23:0] p_wt_bytes  = desc1[31:0];     // w4
    wire [23:0] p_out_bytes = desc1[63:32];    // w5
    wire [63:0] p_act_addr  = desc1[127:64];   // {w7,w6}
    wire [63:0] p_wt_addr   = {desc2[63:0]};   // {w9,w8}
    wire [63:0] p_out_addr  = desc2[127:64];   // {w11,w10}

    // current-layer config latched at descriptor-parse time, held through KICK/RUN/OUT
    reg [63:0] cur_act_addr, cur_wt_addr, cur_out_addr;
    reg [23:0] cur_act_bytes, cur_wt_bytes, cur_out_bytes;

    always @(posedge aclk) begin
        if (!aresetn) begin
            st<=S_IDLE; dma_req<=0; sys_start<=0; busy<=0; done_latch<=0;
            ap_seen<=0; issued<=0; cur_layer<=0; out_rem<=0; out_off<=0;
            cur_cin<=0; cur_cout<=0; cur_hw<=0;
            cur_act_addr<=0; cur_wt_addr<=0; cur_out_addr<=0;
            cur_act_bytes<=0; cur_wt_bytes<=0; cur_out_bytes<=0;
        end else begin
            dma_req<=0; sys_start<=0;
            case (st)
                S_IDLE: begin
                    busy<=0; issued<=0;
                    if (ap_start && !ap_seen) begin
                        ap_seen<=1; busy<=1; cur_layer<=0; done_latch<=0; st<=S_LDDESC;
                    end
                    if (!ap_start) ap_seen<=0;
                end
                // ── fetch 64-byte descriptor for cur_layer ──
                S_LDDESC: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_DESC;
                            dma_req_addr<=desc_base + {52'd0, cur_layer, 6'd0};  // base + layer*64
                            dma_req_bytes<=24'd64; issued<=1;
                        end
                    end else if (dma_done) begin
                        // parse descriptor → latch this layer's config
                        cur_cin       <= desc0[13:0];     // w0
                        cur_cout      <= desc0[41:32];    // w1
                        cur_hw        <= desc0[73:64];    // w2
                        cur_act_bytes <= p_act_bytes; cur_wt_bytes <= p_wt_bytes; cur_out_bytes <= p_out_bytes;
                        cur_act_addr  <= p_act_addr;  cur_wt_addr  <= p_wt_addr;  cur_out_addr  <= p_out_addr;
                        issued<=0; st<=S_LDA;
                    end
                end
                // ── load activation (layer i) ──
                S_LDA: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_ACT;
                            dma_req_addr<=cur_act_addr; dma_req_bytes<=cur_act_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_LDW; end
                end
                // ── load weights+params (layer i) ──
                S_LDW: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_WT;
                            dma_req_addr<=cur_wt_addr; dma_req_bytes<=cur_wt_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_KICK; end
                end
                // ── kick the systolic engine (config muxed by cur_* in cnn_acc_top) ──
                S_KICK: begin sys_start<=1; out_rem<=cur_out_bytes; out_off<=24'd0; st<=S_RUN; end
                S_RUN:  if (sys_done) st<=S_OUT;
                // ── store output to DDR, chunked ≤1024B ──
                S_OUT: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_OUT;
                            dma_req_addr<=cur_out_addr + {40'd0, out_off};
                            dma_req_bytes<=(out_rem > 24'd1024) ? 24'd1024 : out_rem;
                            issued<=1;
                        end
                    end else if (dma_done) begin
                        issued<=0;
                        if (out_rem > 24'd1024) begin
                            out_rem<=out_rem-24'd1024; out_off<=out_off+24'd1024;
                        end else st<=S_NEXT;
                    end
                end
                // ── advance to next layer or finish ──
                S_NEXT: begin
                    if (cur_layer < (n_layers - 4'd1)) begin
                        cur_layer<=cur_layer+4'd1; issued<=0; st<=S_LDDESC;  // chain: next layer
                    end else st<=S_DONE;
                end
                S_DONE: begin done_latch<=1; busy<=0; ap_seen<=0; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
