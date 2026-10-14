// spec_layer_seq.v -- On-chip layer loop for the I-SPEC engine (spec_conv1x1_top).
// Replaces the Python per-spatial-tile host loop (5890 kicks → 1 ap_start/layer). On ONE ap_start
// (LAYER_MODE=1, is_sp) it: DMAs the COO weight+param blob ONCE → spec coo/params, then loops the
// HW spatial-tiles (st): DMA act tile → amem, kick spec compute, DMA out tile back — all on-chip.
// The spec engine processes ALL Cout (ob16 blocks) + ALL Cin (sparse M-tiles) per kick, so the loop
// is a single spatial-tile sweep (simpler than systolic_layer_seq's hwt×cot).
//
// DDR tile-stream (host bakes 1x/layer; 4KB-aligned strides for dma_engine burst safety):
//   WT  : wt_base                        (COO blob + requant params; loaded ONCE)
//   ACT : act_base + st * act_stride     (one MM-tile group = Cin*16 bytes; stride 4KB-aligned)
//   OUT : out_base + st * out_stride     (Cout*16 bytes; stride 4KB-aligned)
//
// Owns dma_engine while is_sp & LAYER_MODE=1 (muxed in cnn_acc_top, like systolic_layer_seq/dw_layer).
// Mirrors systolic_seq's per-DMA `issued` handshake (advance only on OWN dma_done).
`timescale 1ns/1ps

module spec_layer_seq (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        layer_mode,     // 1 = this sequencer drives spec engine + dma_engine
    input  wire        ap_start,
    input  wire        sp_done,        // spec engine compute done (per tile-kick)

    // config (latched at ap_start)
    input  wire [15:0] n_st,           // # spatial-tiles = ceil(HW/16)
    input  wire [23:0] act_stride,     // DDR stride/tile (4KB-aligned)
    input  wire [23:0] act_bytes,      // DMA bytes/tile into amem = Cin*16
    input  wire [23:0] wt_bytes,       // COO+param blob bytes (loaded once)
    input  wire [23:0] out_stride,     // DDR stride/tile (4KB-aligned)
    input  wire [23:0] out_bytes,      // DMA bytes/tile from omem = Cout*16
    input  wire [63:0] act_base,
    input  wire [63:0] wt_base,
    input  wire [63:0] out_base,

    // dma_engine handshake (muxed owner in cnn_acc_top)
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,    // 0=ACT 1=WT 2=OUT
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    output reg         sp_start,       // kick spec engine compute (one spatial-tile)
    output reg         busy,
    output reg         irq,
    output reg         ls_done_latch,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2;
    localparam S_IDLE=4'd0, S_LDW=4'd1, S_LDA=4'd2, S_KICK=4'd3,
               S_RUN=4'd4, S_OUT=4'd5, S_NEXT=4'd6, S_DONE=4'd7;
    reg [3:0]  st;
    reg        ap_seen, issued;
    reg [15:0] sti;                         // spatial-tile counter
    reg [15:0] l_nst;
    reg [23:0] l_actc, l_actb_bytes, l_wtc, l_outc, l_outb_bytes;
    reg [63:0] l_actb, l_wtb, l_outb;
    reg [23:0] out_rem, out_off;            // OUT chunking (dma_out_addr 10-bit => <=1024B)

    assign dbg_state = st;

    wire [63:0] act_addr = l_actb + ({48'd0, sti} * {40'd0, l_actc});
    wire [63:0] out_addr = l_outb + ({48'd0, sti} * {40'd0, l_outc});

    always @(posedge aclk) begin
        if (!aresetn) begin
            st<=S_IDLE; dma_req<=0; dma_req_tgt<=0; dma_req_addr<=0; dma_req_bytes<=0;
            sp_start<=0; busy<=0; irq<=0; ls_done_latch<=0; ap_seen<=0; issued<=0;
            sti<=0; out_rem<=0; out_off<=0;
        end else begin
            dma_req<=1'b0; sp_start<=1'b0; irq<=1'b0;
            if (ap_start && layer_mode) ls_done_latch<=1'b0;
            case (st)
                S_IDLE: begin
                    busy<=1'b0; issued<=1'b0;
                    if (ap_start && layer_mode && !ap_seen) begin
                        ap_seen<=1'b1; busy<=1'b1; sti<=16'd0;
                        l_nst<=n_st;
                        l_actc<=act_stride; l_actb_bytes<=act_bytes; l_wtc<=wt_bytes;
                        l_outc<=out_stride; l_outb_bytes<=out_bytes;
                        l_actb<=act_base; l_wtb<=wt_base; l_outb<=out_base;
                        st<=S_LDW;
                    end
                    if (!ap_start) ap_seen<=1'b0;
                end
                // ── load COO weights + params ONCE (whole layer) ──
                S_LDW: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_WT;
                            dma_req_addr<=l_wtb; dma_req_bytes<=l_wtc; issued<=1'b1;
                        end
                    end else if (dma_done) begin issued<=1'b0; st<=S_LDA; end
                end
                // ── load ACT for spatial-tile sti → amem ──
                S_LDA: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_ACT;
                            dma_req_addr<=act_addr; dma_req_bytes<=l_actb_bytes; issued<=1'b1;
                        end
                    end else if (dma_done) begin issued<=1'b0; st<=S_KICK; end
                end
                S_KICK: begin sp_start<=1'b1; out_rem<=l_outb_bytes; out_off<=24'd0; st<=S_RUN; end
                S_RUN:  if (sp_done) st<=S_OUT;
                // ── store OUT for sti, chunked <=1024B ──
                S_OUT: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_OUT;
                            dma_req_addr<=out_addr + {40'd0, out_off};
                            dma_req_bytes<=(out_rem > 24'd1024) ? 24'd1024 : out_rem;
                            issued<=1'b1;
                        end
                    end else if (dma_done) begin
                        issued<=1'b0;
                        if (out_rem > 24'd1024) begin
                            out_rem<=out_rem-24'd1024; out_off<=out_off+24'd1024;
                        end else st<=S_NEXT;
                    end
                end
                S_NEXT: begin
                    if (sti + 16'd1 < l_nst) begin sti<=sti+16'd1; st<=S_LDA; end
                    else st<=S_DONE;
                end
                S_DONE: begin irq<=1'b1; busy<=1'b0; ls_done_latch<=1'b1; ap_seen<=1'b0; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
