// conv3x3_systolic_top.v -- conv3x3 with ON-CHIP im2col: the host puts the RAW padded activation
// (no 9x A' expansion, no host banking), the engine windows the 3x3 taps on the fly while feeding the
// pe_array (Cin'=9*Cin virtual channels), and writes the output row-padded (de-banking ~free on host).
// Eliminates host im2col (0.13s) + conv3x3 amem-build. STRIDE-1 only (stride-2 = strided gather -> host).
//
// Reuses pe_array + rr_unit + the requant/omem pipe of conv1x1_systolic_top. The NEW part is the FEED:
//   amem holds raw padded act banked: amem[ci*WBin + (hw>>4)][hw&15] = A_pad[ci][hw], hw = row*Wp+col.
//   For output block (ho, wo_blk) [16 cols] and tap (kh,kw): input row=ho+kh, col_base=wo_blk*16+kw,
//   in_base=(ho+kh)*Wp+col_base. The 16 windowed lanes = A[ci][in_base..in_base+15] (consecutive, s=1).
//   2-WORD read (in_base may be unaligned) via EVEN/ODD amem banks (single-port URAM can't read 2 adj
//   words/cycle), then BARREL-SHIFT {w_odd,w_even} by (in_base&15) bytes.
//   Output omem is ROW-PADDED: omem[co*Ho*WBrow + ho*WBrow + (wo>>4)] -> 16-block always word-aligned.
`timescale 1ns/1ps
`include "cnn_defines.vh"

module conv3x3_systolic_top #(
    parameter integer N=16, LOGN=4, AW=8, ACCW=32,
    // WDEPTH 16384->4096 (H3 200MHz): the model's max conv3x3 Cin9=2304 (Cin=256) < 4096, and cchunk
    // is omem-bound (=16) so no throughput loss. 4096-deep wmem = SINGLE URAM tile (no cascade) =>
    // fast read (was the worst 200MHz path: wmem URAM cascade -> pe DSP).
    parameter integer WDEPTH=4096, ADEPTH=8192, ODEPTH=8192, MAW=14, CDEPTH=512
)(
    input  wire                 clk, rst_n,
    // wide loads (DMA-fed). amem split even/odd by word index inside this module.
    input  wire                 ld_w_we,  input wire [MAW-1:0] ld_w_addr, input wire [N*AW-1:0] ld_w_data,
    input  wire                 ld_a_we,  input wire [MAW-1:0] ld_a_addr, input wire [N*AW-1:0] ld_a_data,
    input  wire                 ld_p_we,  input wire [MAW-1:0] ld_p_addr, input wire [127:0]    ld_p_data,
    // run config
    input  wire                 start,
    input  wire [9:0]           Cout,
    input  wire [13:0]          Cin,            // real Cin (engine multiplies by 9 internally)
    input  wire [9:0]           Hp, Wp,         // padded input dims
    input  wire [9:0]           Ho, Wo,         // output dims (stride 1: Ho=Hp-2, Wo=Wp-2)
    input  wire [9:0]           WBin,           // ceil(Hp*Wp/16) -> amem words per channel
    input  wire [9:0]           WBrow,          // ceil(Wo/16)    -> omem words per output row
    input  wire                 apply_relu,
    output reg                  done,
    input  wire [MAW-1:0]       rd_o_addr,
    output reg  [N*AW-1:0]      rd_o_data
);
    integer t; genvar gb;

    // ── memories ── amem split into EVEN/ODD word banks for the 2-adjacent-word read
    (* ram_style="ultra" *) reg [N*AW-1:0] wmem [0:WDEPTH-1];
    // H2 dual-K: 2nd weight read port (kw+1 tap) — duplicate of wmem (single-port URAM can't read 2/cyc)
    (* ram_style="ultra" *) reg [N*AW-1:0] wmem_b [0:WDEPTH-1];
    (* ram_style="ultra" *) reg [N*AW-1:0] amem_e [0:ADEPTH-1];   // even words (idx 0,2,4..)
    (* ram_style="ultra" *) reg [N*AW-1:0] amem_o [0:ADEPTH-1];   // odd  words (idx 1,3,5..)
    (* ram_style="block" *) reg [N*AW-1:0] omem [0:ODEPTH-1];
    reg [127:0] pmem [0:(CDEPTH/2)-1];

    // ── requant pipe (same as conv1x1) ──
    reg signed [ACCW-1:0] rq_psum_r [0:N-1];
    reg [`SCALE_W-1:0] rq_m0_r; reg [`SHIFT_W-1:0] rq_shift_r; reg rq_v_r;
    reg [MAW-1:0] oa_p0,oa_p1,oa_p2,oa_p3,oa_p4,oa_p5; reg [N-1:0] cv_p0,cv_p1,cv_p2,cv_p3,cv_p4,cv_p5;  // p5: H3 rr_unit 5-stage
    wire signed [`DATA_W-1:0] rq_int8 [0:N-1]; wire [N-1:0] rq_vo;

    // ── loads + omem write ──
    always @(posedge clk) begin
        if (ld_w_we) begin wmem[ld_w_addr] <= ld_w_data; wmem_b[ld_w_addr] <= ld_w_data; end
        if (ld_a_we) begin
            if (ld_a_addr[0]==1'b0) amem_e[ld_a_addr[MAW-1:1]] <= ld_a_data;   // even word
            else                    amem_o[ld_a_addr[MAW-1:1]] <= ld_a_data;   // odd word
        end
        if (ld_p_we) pmem[ld_p_addr] <= ld_p_data;
        for (t=0;t<N;t=t+1)
            if (rq_vo[t] && cv_p5[t]) omem[oa_p5][t*AW +: AW] <= rq_int8[t];
    end

    // ── pe_array ── (act_col=W, wt_row=A; same convention as conv1x1)
    reg  [N*AW-1:0] pe_act, pe_wt, pe_act_b, pe_wt_b; reg pe_en, pe_clr; reg [LOGN-1:0] pe_ri;
    wire [N*ACCW-1:0] pe_row; wire signed [ACCW-1:0] pe_o;
    pe_array #(.N(N),.AW(AW),.ACCW(ACCW),.DUAL_K(1)) u_pe (   // DUAL_K=1: real 2-tap/cyc (prod_b used)
        .clk(clk),.rst_n(rst_n),.clr(pe_clr),.en(pe_en),
        .act_col(pe_act),.wt_row(pe_wt),
        .act_col_b(pe_act_b),.wt_row_b(pe_wt_b),            // H2 dual-K: 2nd tap (kw+1) of the pair
        .rd_i(pe_ri),.rd_j({LOGN{1'b0}}),
        .rd_o(pe_o),.rd_row_o(pe_row));

    generate for (gb=0; gb<N; gb=gb+1) begin: RQ
        rr_unit u_rr (.clk(clk),.rst_n(rst_n),.psum_in(rq_psum_r[gb]),.rq_scale(rq_m0_r),
            .rq_shift(rq_shift_r),.apply_relu(apply_relu),.valid_in(rq_v_r),
            .data_out(rq_int8[gb]),.valid_out(rq_vo[gb]));
    end endgenerate

    // ── FSM ──
    localparam S_IDLE=0,S_CLR=1,S_FEED=2,S_DRAIN=3,S_STORE=4,S_RQDRAIN=5,S_NEXT=6;
    reg [2:0] st;
    reg [9:0]  co, ho, wo_blk;           // output iterators (co block, output row, wo 16-block)
    reg [13:0] ci;                       // real input channel
    // H2 dual-K event iterator: per (kh row) two events — ph=0 dual (kw 0,1), ph=1 single (kw 2).
    // The dual event shares ONE amem read (barrel at sh and sh+1) so 6 events/ci replace 9 taps.
    reg [1:0]  kh;                       // kernel row 0..2
    reg        ph;                       // 0 = dual (kw 0,1), 1 = single (kw 2)
    reg [13:0] ci9;                      // ci*9 (running, for wmem addr)
    reg        feed_vld, feed_vld2, feed_vld3; reg [4:0] si; reg [2:0] drain; reg [2:0] rqd;  // feed_vld2/3: H3 addr+read-output reg
    wire [3:0] kw_base  = ph ? 4'd2 : 4'd0;             // first tap column of the event
    wire [3:0] tap_main = kh*3 + kw_base;               // main tap (+1 = the kw1 tap in dual)

    // wmem addr: W'[co][ci*9+tap_main]; the _b read takes tap_main+1 (kw1 weight) for dual events
    wire [MAW-1:0] wrd_addr = (co >> LOGN)*(Cin*9) + ci9 + tap_main;
    // windowed input base (stride 1): row=(ho+kh), col=wo_blk*16+kw_base
    wire [19:0] in_row  = ho + kh;
    wire [19:0] in_col  = (wo_blk << LOGN) + kw_base;
    wire [19:0] in_base = in_row*Wp + in_col;          // padded linear index of the kw_base tap
    wire [MAW-1:0] aw_word = ci*WBin + in_base[19:LOGN];   // word index within channel
    // even/odd bank addresses: words aw_word and aw_word+1 (one even, one odd)
    wire [MAW-1:0] aw_lo = aw_word;
    wire [MAW-1:0] aw_hi = aw_word + 1'b1;
    wire [MAW-2:0] e_addr = aw_lo[0]==1'b0 ? aw_lo[MAW-1:1] : aw_hi[MAW-1:1];  // even of the pair
    wire [MAW-2:0] o_addr = aw_lo[0]==1'b0 ? aw_hi[MAW-1:1] : aw_lo[MAW-1:1];  // odd  of the pair

    // H3 200MHz: register the ADDRESS (break the in_base-mult + ci*WBin -> URAM-address path,
    // 5.6ns logic) one cycle before the URAM read. +1 feed latency (handled by feed_vld2 + drain).
    reg [MAW-2:0] e_addr_q, o_addr_q; reg [MAW-1:0] wrd_addr_q;
    reg [LOGN-1:0] sh_q; reg lo_is_even_q; reg ph_q1;
    always @(posedge clk) begin
        e_addr_q     <= e_addr;
        o_addr_q     <= o_addr;
        wrd_addr_q   <= wrd_addr;
        sh_q         <= in_base[LOGN-1:0];
        lo_is_even_q <= (aw_lo[0]==1'b0);
        ph_q1        <= ph;
    end
    reg  [N*AW-1:0] rd_e, rd_o; reg [LOGN-1:0] sh_r; reg lo_is_even_r;
    always @(posedge clk) begin
        rd_e <= amem_e[e_addr_q];
        rd_o <= amem_o[o_addr_q];
        sh_r <= sh_q;
        lo_is_even_r <= lo_is_even_q;
    end
    // reassemble {word_hi, word_lo} then barrel-shift right by sh bytes
    wire [N*AW-1:0] w_lo = lo_is_even_r ? rd_e : rd_o;
    wire [N*AW-1:0] w_hi = lo_is_even_r ? rd_o : rd_e;
    wire [2*N*AW-1:0] cat = {w_hi, w_lo};
    wire [N*AW-1:0] aq128   = cat[ sh_r        *AW +: N*AW];   // A at kw_base (main)
    wire [N*AW-1:0] aq128_b = cat[(sh_r+1'b1)  *AW +: N*AW];   // A at kw_base+1 (dual 2nd tap) — same read

    // wmem reads from the REGISTERED address (aligned with the amem read above): main=tap_main, _b=tap+1
    reg [N*AW-1:0] wq128, wq128_b; reg ph_r;
    always @(posedge clk) begin
        wq128   <= wmem  [wrd_addr_q];
        wq128_b <= wmem_b[wrd_addr_q + 1'b1];
        ph_r    <= ph_q1;                                     // align the dual/single flag with the data
    end
    // H3 200MHz: register the feed outputs (the URAM read -> pe DSP route was 3.86ns/70%). A pipeline
    // register here lets the placer split that long cross-region route. +1 latency (feed_vld3 + drain).
    reg [N*AW-1:0] wq128_2, wq128_b_2, aq128_2, aq128_b_2; reg ph_r2;
    always @(posedge clk) begin
        wq128_2   <= wq128;    wq128_b_2 <= wq128_b;
        aq128_2   <= aq128;    aq128_b_2 <= aq128_b;
        ph_r2     <= ph_r;
    end

    wire [9:0]   rq_co = co + si;
    wire [127:0] pword = pmem[rq_co[9:1]];
    wire [63:0]  phalf = rq_co[0] ? pword[127:64] : pword[63:0];

    always @(posedge clk) begin
        if (!rst_n) begin
            st<=S_IDLE; done<=0; pe_clr<=0; pe_en<=0;
            co<=0; ho<=0; wo_blk<=0; ci<=0; kh<=0; ph<=0; ci9<=0; feed_vld<=0; feed_vld2<=0; feed_vld3<=0; si<=0; drain<=0; rqd<=0; rq_v_r<=0;
        end else begin
            done<=0; pe_clr<=0; pe_en<=0; rq_v_r<=0;
            oa_p1<=oa_p0; oa_p2<=oa_p1; oa_p3<=oa_p2; oa_p4<=oa_p3; oa_p5<=oa_p4;
            cv_p1<=cv_p0; cv_p2<=cv_p1; cv_p3<=cv_p2; cv_p4<=cv_p3; cv_p5<=cv_p4;
            // H3: read pipeline = address-reg + read + read-output-reg = +2 stages. Feed pe_array on
            // feed_vld3 (= feed_vld delayed 2) for the registered-output data; drains in S_DRAIN.
            feed_vld2 <= feed_vld;
            feed_vld3 <= feed_vld2;
            if (feed_vld3) begin
                pe_act = wq128_2;  pe_wt = aq128_2;
                if (ph_r2 == 1'b0) begin pe_act_b = wq128_b_2;     pe_wt_b = aq128_b_2;     end
                else               begin pe_act_b = {N*AW{1'b0}};  pe_wt_b = {N*AW{1'b0}};  end
                pe_en <= 1;
            end
            case (st)
                S_IDLE: if (start) begin co<=0; ho<=0; wo_blk<=0; st<=S_CLR; end
                S_CLR:  begin pe_clr<=1; ci<=0; kh<=0; ph<=0; ci9<=0; feed_vld<=0; st<=S_FEED; end
                S_FEED: begin
                    if (ci == Cin) begin feed_vld<=1'b0; drain<=0; st<=S_DRAIN; end  // stop new; in-flight drains via feed_vld2
                    else begin
                        feed_vld<=1'b1;
                        if (ph == 1'b1) begin                          // single done -> next row / channel
                            ph<=1'b0;
                            if (kh==2'd2) begin kh<=2'd0; ci<=ci+14'd1; ci9<=ci9+14'd9; end
                            else kh<=kh+2'd1;
                        end else ph<=1'b1;                             // dual done -> do the single (kw 2)
                    end
                end
                S_DRAIN: begin feed_vld<=1'b0; drain<=drain+3'd1; if (drain==3'd3) begin si<=0; pe_ri<=0; st<=S_STORE; end end  // B3 MREG PE: +1 drain (3-stage feed + MREG, was 2)
                S_STORE: begin
                    for (t=0;t<N;t=t+1)
                        rq_psum_r[t] <= $signed(pe_row[t*ACCW +: ACCW]) + $signed(phalf[55:24]);
                    rq_m0_r<=phalf[15:0]; rq_shift_r<=phalf[16 +: `SHIFT_W];
                    rq_v_r <= (co+si) < Cout;
                    oa_p0  <= (co+si)*(Ho*WBrow) + ho*WBrow + wo_blk;        // ROW-PADDED omem
                    for (t=0;t<N;t=t+1) cv_p0[t] <= (((wo_blk<<LOGN)+t) < Wo);
                    if (si==N-1) begin rqd<=0; st<=S_RQDRAIN; end
                    else begin si<=si+5'd1; pe_ri<=pe_ri+1'b1; end
                end
                S_RQDRAIN: begin rqd<=rqd+3'd1; if (rqd==3'd6) st<=S_NEXT; end  // H3: rr_unit 5-stage
                S_NEXT: begin
                    if ((wo_blk+1) < WBrow)      begin wo_blk<=wo_blk+10'd1; st<=S_CLR; end
                    else if ((ho+1) < Ho)        begin wo_blk<=0; ho<=ho+10'd1; st<=S_CLR; end
                    else if (co + N < Cout)      begin wo_blk<=0; ho<=0; co<=co+10'd16; st<=S_CLR; end
                    else done<=1;
                end
                default: st<=S_IDLE;
            endcase
            if (st==S_NEXT && !((wo_blk+1)<WBrow) && !((ho+1)<Ho) && !(co+N<Cout)) st<=S_IDLE;
        end
    end

    always @(posedge clk) rd_o_data <= omem[rd_o_addr];
    wire _unused = &{1'b0, pe_o};
endmodule
