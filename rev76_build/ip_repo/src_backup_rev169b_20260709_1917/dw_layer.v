// dw_layer.v -- on-chip depthwise LAYER processor: loops C channels, 1 ap_start/layer.
// For each channel: DMA padded act -> dw_engine.in_mem, kick dw_engine, DMA out_mem -> DDR.
// Weights for ALL channels are DMA'd once into weight_mem (1 128-bit word/channel):
//   word = {3'b0, shift[4:0], m0[15:0], bias[31:0], w8..w0 (9*8)}  (host packs this layout)
// Shares the dma_engine with the systolic path (cnn_acc_top muxes by layer_type).
// This is the on-chip channel loop that makes the FPGA dw beat dw_fast (1 kick/layer, not C).
`timescale 1ns/1ps

module dw_layer #(
    parameter DW=8, ACCW=32, ADW=15, AW128=11, ODW=15, MAXWP=512, WCW=9, PRMAW=14
)(
    input  wire        aclk, aresetn,
    input  wire        ap_start,
    input  wire [8:0]  C,                 // channel count (<=512)
    input  wire [9:0]  Hp, Wp, Ho, Wo,
    input  wire [1:0]  stride,
    input  wire        apply_relu,
    input  wire [63:0] act_addr, wt_addr, out_addr,
    input  wire [23:0] act_chan_bytes,    // Hp*Wp (one padded channel)
    input  wire [23:0] out_chan_bytes,    // Ho*Wo
    input  wire [23:0] wt_bytes,          // C*16
    // T2-3 dwoc: read padded input ON-CHIP from c1 omem (pad_rebank) instead of DMA. out still DMA'd.
    input  wire        dwoc,                  // 1 = on-chip input via pad_rebank
    input  wire [PRMAW-1:0] omem_WBsrc,       // ceil(Ho*Wo/16): conv1x1 output WB (source words/channel)
    output wire [PRMAW-1:0] omem_rd_addr,     // -> cnn_acc_top sys_o_rd_addr (c1 omem read)
    input  wire [127:0] omem_rd_data,         // <- sys_o_rd_data (registered 1-cyc)
    // C/T2-4 dwout_amem: write dw output ON-CHIP to c1 amem (no DMA out). INDEPENDENT of dwoc.
    // Dest layout = c1 amem contiguous banked [ch*out_WB + word]; dw out_mem is the same contiguous
    // banked layout -> STRAIGHT word copy (xcopy lag-1), NO re-banking. Only valid when C*out_WB<=ADEPTH
    // (FIT CONSTRAINT: small-spatial layers 13/26 only). No DMA -> no dma_req in this path.
    input  wire        dwout_amem,            // 1 = output -> c1 amem (next conv1x1 input)
    input  wire [PRMAW-1:0] out_WB,           // ceil(Ho*Wo/16): dest words/channel (dw out)
    output reg         amem_we,
    output reg  [PRMAW-1:0] amem_waddr,       // c1 amem write addr (amem_base + word)
    output reg  [127:0] amem_wdata,           // = dw_engine.out_wword (registered)
    // dma_engine handshake (mirrors systolic_seq)
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active, dma_done,
    // dma_engine BRAM streams (gated/owned by cnn_acc_top when layer_type==4)
    input  wire        dma_in_w128_en,
    input  wire [127:0]dma_in_w128_data,
    input  wire        dma_wt_w128_en,
    input  wire [127:0]dma_wt_w128_data,
    input  wire [9:0]  dma_out_addr,
    output wire [7:0]  dma_out_data,
    output reg         busy, irq,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2;
    localparam S_IDLE=0, S_LDWT=1, S_LDA=2, S_KICK=3, S_RUN=4, S_OUT=5, S_NEXT=6, S_DONE=7;
    reg [2:0] st;
    reg [8:0] ch;                         // current channel
    reg       ap_seen, issued;
    reg [AW128-1:0] a_cnt;                // in_mem write counter (per channel)
    reg [8:0] wt_cnt;                     // weight_mem write counter
    reg [23:0] out_rem, out_off;
    // C/T2-4 dwout_amem state: word sweep + running dest base (= ch*out_WB, avoids per-cycle multiply)
    reg [PRMAW-1:0] ow_cnt;                // output word counter (drives out_raddr=ow_cnt<<4)
    reg [PRMAW-1:0] amem_base;             // = ch*out_WB (running)
    assign dbg_state = st;

    // ── weight memory (1 word/channel) ── registered read (ch stable per channel) breaks the
    // ch -> weight_mem(async) -> MAC/requant critical path (WNS) + infers BRAM not distributed RAM.
    (* ram_style="block" *) reg [127:0] weight_mem [0:511];
    always @(posedge aclk) if (dma_wt_w128_en) weight_mem[wt_cnt] <= dma_wt_w128_data;
    reg [127:0] ww;
    always @(posedge aclk) ww <= weight_mem[ch];
    wire signed [DW-1:0]   dw_w0=ww[7:0],   dw_w1=ww[15:8],  dw_w2=ww[23:16],
                           dw_w3=ww[31:24], dw_w4=ww[39:32], dw_w5=ww[47:40],
                           dw_w6=ww[55:48], dw_w7=ww[63:56], dw_w8=ww[71:64];
    wire signed [ACCW-1:0] dw_bias = ww[103:72];
    wire [15:0]            dw_m0   = ww[119:104];
    wire [4:0]             dw_shift= ww[124:120];

    // ── T2-3 pad_rebank: per-channel reads c1 omem (src_off=ch*WBsrc) -> dw_engine.in_mem (padded banked).
    // Replaces the per-channel ACT DMA when dwoc. One pad_rebank pass writes WBin=ceil(Hp*Wp/16) words.
    reg  padrb_start;
    wire padrb_busy, padrb_done, padrb_wr_en;
    wire [PRMAW-1:0] padrb_wr_addr;
    wire [127:0] padrb_wr_data;
    wire [22:0] ch_src_full = ch * omem_WBsrc;             // ch*WBsrc (fits omem -> <=8191)
    pad_rebank #(.MAW(PRMAW), .N(16), .AW(8)) u_padrb (
        .clk(aclk), .rst_n(aresetn), .start(padrb_start),
        .Cout(10'd1), .Ho(Ho), .Wo(Wo), .WBsrc(omem_WBsrc), .src_off(ch_src_full[PRMAW-1:0]),
        .rd_addr(omem_rd_addr), .rd_data(omem_rd_data),
        .wr_en(padrb_wr_en), .wr_addr(padrb_wr_addr), .wr_data(padrb_wr_data),
        .busy(padrb_busy), .done_latch(padrb_done)
    );
    // dw_engine in_mem write: ACT DMA (normal) or pad_rebank (dwoc on-chip)
    wire             dw_in_we    = dwoc ? padrb_wr_en               : dma_in_w128_en;
    wire [AW128-1:0] dw_in_waddr = dwoc ? padrb_wr_addr[AW128-1:0]  : a_cnt;
    wire [127:0]     dw_in_wdata = dwoc ? padrb_wr_data            : dma_in_w128_data;

    // ── dw_engine (its in_mem fed by the ACT DMA stream OR pad_rebank, out_mem read by the OUT DMA) ──
    reg  dw_start;
    wire dw_done, dw_busy;
    wire [127:0] dw_out_wword;                // C/T2-4: 128-bit word read of dw out_mem (for amem handoff)
    // out_raddr: DMA byte-addr (normal) OR word-sweep ow_cnt<<4 (dwout_amem on-chip output)
    wire [ODW-1:0] dw_out_raddr  = dwout_amem ? {ow_cnt[ODW-5:0], 4'd0}
                                              : (out_off[ODW-1:0] + {{(ODW-10){1'b0}}, dma_out_addr});
    dw_engine #(.DW(DW),.ACCW(ACCW),.ADW(ADW),.AW128(AW128),.ODW(ODW),.MAXWP(MAXWP)) u_dw (
        .clk(aclk), .rst_n(aresetn), .start(dw_start),
        .Hp(Hp), .Wp(Wp), .Ho(Ho), .Wo(Wo), .stride(stride),
        .w0(dw_w0),.w1(dw_w1),.w2(dw_w2),.w3(dw_w3),.w4(dw_w4),
        .w5(dw_w5),.w6(dw_w6),.w7(dw_w7),.w8(dw_w8),
        .bias(dw_bias), .m0(dw_m0), .shift(dw_shift), .apply_relu(apply_relu),
        .in_we(dw_in_we), .in_waddr(dw_in_waddr), .in_wdata(dw_in_wdata),
        .out_raddr(dw_out_raddr), .out_rdata(dma_out_data), .out_wword(dw_out_wword),
        .done(dw_done), .busy(dw_busy)
    );

    // ── C/T2-4 amem write (xcopy lag-1): out_raddr driven by ow_cnt (combinational) -> out_wword valid
    // 1 cyc later. When ow_cnt=k, out_wword = word[k-1]. Write word[k-1] @ amem_base+(ow_cnt-1) for
    // ow_cnt in [1..out_WB]. NO dma_req here -> cannot runaway-DMA. amem_wdata = registered out_wword.
    always @(*) begin
        amem_we    = (st==S_OUT) && dwout_amem && (ow_cnt != {PRMAW{1'b0}}) && (ow_cnt <= out_WB);
        amem_waddr = amem_base + (ow_cnt - {{(PRMAW-1){1'b0}}, 1'b1});
        amem_wdata = dw_out_wword;
    end

    // in_mem write counter: reset at each channel's ACT load, ++ per word
    always @(posedge aclk) begin
        if (!aresetn) a_cnt<=0;
        else if (st==S_LDA && !issued) a_cnt<=0;
        else if (dma_in_w128_en) a_cnt<=a_cnt+1'b1;
    end
    always @(posedge aclk) begin
        if (!aresetn) wt_cnt<=0;
        else if (st==S_LDWT && !issued) wt_cnt<=0;
        else if (dma_wt_w128_en) wt_cnt<=wt_cnt+1'b1;
    end

    // ── main FSM ──
    always @(posedge aclk) begin
        if (!aresetn) begin
            st<=S_IDLE; dma_req<=0; busy<=0; irq<=0; ap_seen<=0; issued<=0;
            ch<=0; dw_start<=0; out_rem<=0; out_off<=0; padrb_start<=0;
            ow_cnt<=0; amem_base<=0;
        end else begin
            dma_req<=0; irq<=0; dw_start<=0; padrb_start<=0;
            case (st)
                S_IDLE: begin
                    busy<=0; issued<=0; ch<=0; amem_base<=0;   // amem_base=ch*out_WB, reset per layer
                    if (ap_start && !ap_seen) begin ap_seen<=1; busy<=1; st<=S_LDWT; end
                    if (!ap_start) ap_seen<=0;
                end
                S_LDWT: begin                         // load all channels' weights once
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1; dma_req_tgt<=TGT_WT;
                            dma_req_addr<=wt_addr; dma_req_bytes<=wt_bytes; issued<=1;
                        end
                    end else if (dma_done) begin issued<=0; st<=S_LDA; end
                end
                S_LDA: begin                          // load channel ch's padded activation
                    if (dwoc) begin                   // on-chip: pad_rebank c1 omem -> dw in_mem
                        if (!issued) begin padrb_start<=1; issued<=1; end
                        else if (!padrb_busy && !padrb_start) begin issued<=0; st<=S_KICK; end
                    end else begin                    // normal: DMA padded act -> dw in_mem
                        if (!issued) begin
                            if (!dma_active && !dma_req) begin
                                dma_req<=1; dma_req_tgt<=TGT_ACT;
                                dma_req_addr<=act_addr + ({40'd0,act_chan_bytes} * ch);
                                dma_req_bytes<=act_chan_bytes; issued<=1;
                            end
                        end else if (dma_done) begin issued<=0; st<=S_KICK; end
                    end
                end
                S_KICK: begin dw_start<=1; out_rem<=out_chan_bytes; out_off<=0; ow_cnt<=0; st<=S_RUN; end
                S_RUN:  if (dw_done) st<=S_OUT;
                S_OUT: begin
                    if (dwout_amem) begin             // C/T2-4: sweep out words -> c1 amem ON-CHIP (NO DMA)
                        if (ow_cnt == out_WB) begin ow_cnt<=0; st<=S_NEXT; end
                        else ow_cnt <= ow_cnt + {{(PRMAW-1){1'b0}}, 1'b1};
                    end else begin                    // normal: store channel ch output (DMA, chunked <=1024B)
                        if (!issued) begin
                            if (!dma_active && !dma_req) begin
                                dma_req<=1; dma_req_tgt<=TGT_OUT;
                                dma_req_addr<=out_addr + ({40'd0,out_chan_bytes} * ch) + {40'd0,out_off};
                                dma_req_bytes<=(out_rem>24'd1024)?24'd1024:out_rem; issued<=1;
                            end
                        end else if (dma_done) begin
                            issued<=0;
                            if (out_rem>24'd1024) begin out_rem<=out_rem-24'd1024; out_off<=out_off+24'd1024; end
                            else st<=S_NEXT;
                        end
                    end
                end
                S_NEXT: begin
                    if (ch==C-9'd1) st<=S_DONE;
                    else begin ch<=ch+9'd1; st<=S_LDA; amem_base<=amem_base+out_WB; end  // next ch dest base
                end
                S_DONE: begin irq<=1; busy<=0; ap_seen<=0; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
