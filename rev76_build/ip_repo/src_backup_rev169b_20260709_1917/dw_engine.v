// dw_engine.v -- streaming 3x3 depthwise conv, ONE channel per pass, line-buffered.
// Reads PADDED single-channel activation from in_mem (row-major [Hp*Wp]; loader pre-pads so
// borders are real 0s in the data), streams 1 pixel/cycle, maintains 2 line buffers + a 3x3
// window, MACs the 9 per-channel weights, requants via rr_unit (bit-exact leaky 0.10009),
// writes out_mem. ~1 output/cycle -- the on-chip dw datapath the systolic cannot do
// efficiently (depthwise = block-diagonal sparse on the Cin-sum MAC array).
//
// Pipeline alignment (the subtle part):
//   T   : issue in_mem/lb read for stream pos (r,c)
//   T+1 : s1 -- reads ready (in_rd/lb0_rd/lb1_rd), idx r_d/c_d ; shift window with this column,
//         update line buffers. Window is REGISTERED so it reflects (r_d,c_d) only at T+2.
//   T+2 : s2 -- window valid for (r_d2,c_d2); MAC -> rr_unit (3-cyc) -> write out_mem.
`timescale 1ns/1ps
`include "cnn_defines.vh"

module dw_engine #(
    parameter DW   = 8,
    parameter ACCW = 32,
    parameter ADW  = 15,      // pixel addr width (padded chan up to 106*106 < 32768)
    parameter AW128= 11,      // in_mem128 word addr (ADW-4): 2048 words = 32768 px
    parameter ODW  = 15,
    parameter MAXWP= 512
)(
    input  wire             clk,
    input  wire             rst_n,
    input  wire             start,
    input  wire [9:0]       Hp, Wp,          // padded input dims
    input  wire [9:0]       Ho, Wo,          // output dims
    input  wire [1:0]       stride,
    input  wire signed [DW-1:0]   w0,w1,w2,w3,w4,w5,w6,w7,w8,  // row-major 3x3
    input  wire signed [ACCW-1:0] bias,
    input  wire [15:0]      m0,
    input  wire [4:0]       shift,
    input  wire             apply_relu,
    // in_mem load: 128-bit words (16 row-major pixels/word), counter-addressed by the DMA
    input  wire             in_we,
    input  wire [AW128-1:0] in_waddr,
    input  wire [127:0]     in_wdata,
    // out_mem read: byte path for the OUT DMA.
    input  wire [ODW-1:0]   out_raddr,
    output wire [DW-1:0]    out_rdata,
    // C/T2-4: 128-bit WORD read of out_mem for ON-CHIP output handoff (dw out -> c1 amem, no DMA).
    // Pure exposure of the already-registered banked word (out_word_full = banks @ out_raddr>>4).
    // NOT a new BRAM port -- the 16 byte-banks already drive out_word_full for the byte path; we just
    // wire it out. Drive out_raddr=word<<4, read out_wword 1 cycle later (same latency as out_rdata).
    output wire [127:0]     out_wword,
    output reg              done,
    output reg              busy
);
    // ── memories ──  in_mem 128-bit (matches DMA word stream).
    (* ram_style="block" *) reg [127:0] in_mem  [0:(1<<AW128)-1];
    always @(posedge clk) if (in_we) in_mem[in_waddr] <= in_wdata;
    // out_mem: 16 BYTE-BANKS (one per output lane). The per-pixel byte-write uses a VARIABLE lane index
    // which defeats BRAM byte-enable -> a single wide out_mem was mapped to ~128 width-1 BRAMs (huge BRAM
    // waste). Banking by lane makes each bank a clean RAMB18 (16 total). Banked write+read are below
    // (g_obank), after the requant pipeline declares oa_s7/rq_vld/rq_out. Read = concat banks @ word addr.
    reg [3:0] out_bsel_r;
    always @(posedge clk) out_bsel_r <= out_raddr[3:0];
    wire [127:0] out_word_full;
    assign out_rdata = out_word_full[out_bsel_r*8 +: 8];
    assign out_wword = out_word_full;   // C/T2-4: full registered word (out_raddr>>4), 1-cyc latency

    // ── line buffers (2 rows) ──
    reg [DW-1:0] lb0 [0:MAXWP-1];   // row r-2
    reg [DW-1:0] lb1 [0:MAXWP-1];   // row r-1
    reg signed [DW-1:0] win00,win01,win02,win10,win11,win12,win20,win21,win22;

    // ── weights packed ──
    wire signed [DW-1:0] wv0=w0,wv1=w1,wv2=w2,wv3=w3,wv4=w4,wv5=w5,wv6=w6,wv7=w7,wv8=w8;

    // ── stream counters (T) ──
    reg [9:0]  r, c;
    reg [ADW-1:0] raddr;
    reg        streaming;

    // ── s1: registered reads + idx ──  in_rd is a wire: byte-select on the registered word
    reg  [127:0] in_word;
    reg  [3:0]   lane_q;
    wire signed [DW-1:0] in_rd = in_word[lane_q*8 +: 8];
    reg signed [DW-1:0] lb0_rd, lb1_rd;
    reg [9:0]  r_d, c_d;
    reg        rd_valid;

    // ── s2: window-valid idx (1 cycle after the shift) ──
    reg [9:0]  r_d2, c_d2;
    reg        rd_valid2;

    wire [9:0] orow = r_d2 - 10'd2;
    wire [9:0] ocol = c_d2 - 10'd2;
    wire stride_ok  = (stride==2'd1) ? 1'b1 : ((orow[0]==1'b0) && (ocol[0]==1'b0));
    wire emit = rd_valid2 && (r_d2>=10'd2) && (c_d2>=10'd2) && stride_ok;
    wire [ODW-1:0] oaddr = (((stride==2'd1) ? orow : (orow>>1))*Wo)
                         +  ((stride==2'd1) ? ocol : (ocol>>1));

    // ── MAC over the registered 3x3 window — H3 200MHz: PIPELINED into 2 stages.
    // Stage A: three partial sums (3 products each, use_dsp -> DSP).  Stage B: final sum + bias.
    // The old single combinational 9-product add-tree was the worst path (28 logic levels @8.85ns).
    (* use_dsp = "yes" *) reg signed [ACCW-1:0] ps0, ps1, ps2; reg emit_a;
    always @(posedge clk) begin
        ps0 <= win00*wv0 + win01*wv1 + win02*wv2;
        ps1 <= win10*wv3 + win11*wv4 + win12*wv5;
        ps2 <= win20*wv6 + win21*wv7 + win22*wv8;
        emit_a <= emit;
    end
    wire signed [ACCW-1:0] psum = ps0 + ps1 + ps2 + bias;

    // ── register psum before rr_unit (breaks the MAC add-tree -> DSP multiply path for timing) ──
    reg signed [ACCW-1:0] psum_r; reg emit_r;
    always @(posedge clk) begin psum_r <= psum; emit_r <= emit_a; end   // emit_a: MAC stage-A delayed emit

    // ── requant via rr_unit (3-cycle latency); total emit->out = 4 cycles (psum_r + 3) ──
    wire signed [DW-1:0] rq_out;
    wire rq_vld;
    rr_unit u_rq (
        .clk(clk), .rst_n(rst_n),
        .psum_in(psum_r), .rq_scale(m0), .rq_shift(shift),
        .apply_relu(apply_relu), .valid_in(emit_r),
        .data_out(rq_out), .valid_out(rq_vld)
    );
    reg [ODW-1:0] oa_s1, oa_s2, oa_s3, oa_s4, oa_s5, oa_s6, oa_s7;   // s7: H3 MAC 2-stage + rr 5-stage
    always @(posedge clk) begin oa_s1<=oaddr; oa_s2<=oa_s1; oa_s3<=oa_s2; oa_s4<=oa_s3; oa_s5<=oa_s4; oa_s6<=oa_s5; oa_s7<=oa_s6; end
    // out_mem = 16 byte-banks. Write bank oa_s7[3:0] @ word oa_s7>>4 when rq_vld. Read all banks @ word.
    genvar gob;
    generate for (gob=0; gob<16; gob=gob+1) begin: g_obank
        (* ram_style="block" *) reg [7:0] obank [0:(1<<(ODW-4))-1];
        always @(posedge clk)
            if (rq_vld && (oa_s7[3:0]==gob[3:0])) obank[oa_s7[ODW-1:4]] <= rq_out;  // byte-write -> 1 bank
        reg [7:0] obank_rd;
        always @(posedge clk) obank_rd <= obank[out_raddr[ODW-1:4]];
        assign out_word_full[gob*8 +: 8] = obank_rd;
    end endgenerate

    // ── streaming FSM ──
    reg [3:0] tail;   // H3: widened 3->4 bit (drain now MAC 2 + rr 5 = 7, + margin)
    always @(posedge clk) begin
        if (!rst_n) begin
            r<=0; c<=0; raddr<=0; streaming<=0; rd_valid<=0; rd_valid2<=0;
            done<=0; busy<=0; tail<=0;
        end else begin
            done <= 1'b0;
            rd_valid  <= 1'b0;
            rd_valid2 <= rd_valid;        // s1 -> s2
            r_d2<=r_d; c_d2<=c_d;
            if (start && !busy) begin
                r<=0; c<=0; raddr<=0; streaming<=1; busy<=1; tail<=4'd9;   // drain (MAC 2 + rr 5 = 7 + margin)
            end else if (streaming) begin
                in_word <= in_mem[raddr[ADW-1:4]];   // 128-bit word (16 px); byte-select via lane_q
                lane_q  <= raddr[3:0];
                lb0_rd <= lb0[c];
                lb1_rd <= lb1[c];
                r_d<=r; c_d<=c; rd_valid<=1'b1;
                if (c==Wp-10'd1) begin
                    c<=0;
                    if (r==Hp-10'd1) streaming<=0;
                    else r<=r+10'd1;
                end else c<=c+10'd1;
                raddr<=raddr+1'b1;
            end else if (busy) begin
                if (tail!=0) tail<=tail-4'd1;
                else begin busy<=0; done<=1'b1; end
            end
        end
    end

    // ── window shift + line-buffer update (s1) ──
    always @(posedge clk) begin
        if (rd_valid) begin
            win00<=win01; win01<=win02; win02<=lb0_rd;   // top row (r-2)
            win10<=win11; win11<=win12; win12<=lb1_rd;   // mid row (r-1)
            win20<=win21; win21<=win22; win22<=in_rd;    // bot row (r)
            lb0[c_d]<=lb1_rd;
            lb1[c_d]<=in_rd;
        end
    end
endmodule
