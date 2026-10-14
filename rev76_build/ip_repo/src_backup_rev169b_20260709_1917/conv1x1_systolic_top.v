// conv1x1_systolic_top.v -- Option 2 P4: board-ready conv1x1 engine = wide on-chip fmap memory +
// pe_array systolic + N-parallel rr_unit requant. Streams Cin with N²=256 MAC/cycle, ~86-100% util.
//
// MEMORIES are 128-bit-wide (N lanes × AW) BRAM -- a wide word IS the N-lane parallel read, and maps
// directly onto the existing 128-bit wide-DMA (in_bram_wide) path. Host lays out DDR accordingly:
//   wmem[(co>>LOGN)*Cin + ci]      = { W[co+N-1][ci], ..., W[co][ci] }   (N consecutive Cout)
//   amem[ci*WB + (hw>>LOGN)]       = { A[ci][hw+N-1], ..., A[ci][hw] },  WB=ceil(HW/N)
//   omem[co*WB + (hw>>LOGN)]       = { O[co][hw+N-1], ..., O[co][hw] }   (== next layer's amem)
//
// P4-B WIDE READOUT: N rr_units requant a full pe_array row/cycle. BRAM read latency=1 (addr leads
// feed by 1, prime cycle). Requant addr/col-valid pipe = 3 advances (mirrors P3/P4-A, verified).
`timescale 1ns/1ps
`include "cnn_defines.vh"

module conv1x1_systolic_top #(
    parameter integer N     = 16,
    parameter integer LOGN  = 4,
    parameter integer AW    = 8,
    parameter integer ACCW  = 32,
    parameter integer WDEPTH = 16384,   // URAM: wmem=ceil(Cout/16)*Cin (conv3x3 im2col 9x => big)
    parameter integer ADEPTH = 16384,   // URAM: amem=Cin'*ceil(HW/16), Cin'=9*Cin for conv3x3
    parameter integer ODEPTH = 8192,    // omem=Cout*ceil(HW/16) words (BRAM, byte-write)
    parameter integer MAW   = 14,
    parameter integer CDEPTH = 512
) (
    input  wire                 clk,
    input  wire                 rst_n,
    // ── load interface (128-bit wide, DMA-fed) ──
    input  wire                 ld_w_we,
    input  wire [MAW-1:0]       ld_w_addr,
    input  wire [N*AW-1:0]      ld_w_data,
    input  wire                 ld_a_we,
    input  wire [MAW-1:0]       ld_a_addr,
    input  wire [N*AW-1:0]      ld_a_data,
    input  wire                 ld_p_we,        // requant params: 2 Cout per 128-bit word
    input  wire [MAW-1:0]       ld_p_addr,      // param-word index (= co>>1)
    input  wire [127:0]         ld_p_data,      // {co=2p+1 half [127:64], co=2p half [63:0]}
    // ── run ──
    input  wire                 start,
    input  wire [9:0]           Cout,
    input  wire [13:0]          Cin,            // up to 16383 (conv3x3 im2col Cin'=9*Cin)
    input  wire [9:0]           HW,
    input  wire [9:0]           WB,             // ceil(HW/N)
    input  wire                 apply_relu,
    output reg                  done,
    // ── output read-back (128-bit wide word -> DDR / next layer) ──
    input  wire [MAW-1:0]       rd_o_addr,
    output reg  [N*AW-1:0]      rd_o_data
);
    integer t;
    genvar  gb;

    // ───────────── wide memories (128-bit = N-lane) ─────────────
    (* ram_style="ultra" *) reg [N*AW-1:0] wmem [0:WDEPTH-1];   // URAM (16384x128)
    (* ram_style="ultra" *) reg [N*AW-1:0] amem [0:ADEPTH-1];   // URAM
    (* ram_style="block" *) reg [N*AW-1:0] omem [0:ODEPTH-1];   // BRAM (per-lane byte write)
    // requant params: 2 Cout per 128-bit word. half[15:0]=m0, half[20:16]=shift, half[55:24]=bias.
    reg [127:0] pmem [0:(CDEPTH/2)-1];

    // ───────────── requant pipe state (declared early for O-write) ─────────────
    reg  signed [ACCW-1:0] rq_psum_r [0:N-1];
    reg  [`SCALE_W-1:0]    rq_m0_r;
    reg  [`SHIFT_W-1:0]    rq_shift_r;
    reg                    rq_v_r;
    reg  [MAW-1:0]         oa_p0, oa_p1, oa_p2, oa_p3, oa_p4, oa_p5;   // p5: H3 rr_unit 5-stage
    reg  [N-1:0]           cv_p0, cv_p1, cv_p2, cv_p3, cv_p4, cv_p5;
    wire signed [`DATA_W-1:0] rq_int8 [0:N-1];
    wire [N-1:0]           rq_vo;

    // loads + requant O-write (single write port to omem: byte-enabled per valid lane)
    always @(posedge clk) begin
        if (ld_w_we) wmem[ld_w_addr] <= ld_w_data;
        if (ld_a_we) amem[ld_a_addr] <= ld_a_data;          // DMA load / on-chip rebank (via ld_a port)
        if (ld_p_we) pmem[ld_p_addr] <= ld_p_data;
        if (rq_vo[0]) begin
            for (t=0;t<N;t=t+1)
                if (cv_p5[t]) omem[oa_p5][t*AW +: AW] <= rq_int8[t];
        end
    end

    // ───────────── pe_array ─────────────
    reg                  pe_clr, pe_en;
    reg  [N*AW-1:0]      pe_act, pe_wt;
    reg  [LOGN-1:0]      pe_ri;
    wire signed [ACCW-1:0] pe_o;
    wire [N*ACCW-1:0]    pe_row;
    pe_array #(.N(N), .AW(AW), .ACCW(ACCW), .DUAL_K(0)) u_pe (   // DUAL_K=0: no prod_b DSP (saves 256 DSP)
        .clk(clk), .rst_n(rst_n), .clr(pe_clr), .en(pe_en),
        .act_col(pe_act), .wt_row(pe_wt),
        .act_col_b({N*AW{1'b0}}), .wt_row_b({N*AW{1'b0}}),   // single-K: 2nd k-term unused (H2)
        .rd_i(pe_ri), .rd_j({LOGN{1'b0}}),
        .rd_o(pe_o), .rd_row_o(pe_row)
    );

    // ───────────── N parallel rr_units ─────────────
    generate for (gb=0; gb<N; gb=gb+1) begin: g_rr
        rr_unit u_rr (
            .clk(clk), .rst_n(rst_n),
            .psum_in(rq_psum_r[gb]), .rq_scale(rq_m0_r), .rq_shift(rq_shift_r),
            .apply_relu(apply_relu), .valid_in(rq_v_r),
            .data_out(rq_int8[gb]), .valid_out(rq_vo[gb])
        );
    end endgenerate

    // ───────────── FSM ─────────────
    localparam S_IDLE=3'd0, S_CLR=3'd1, S_FEED=3'd2, S_DRAIN=3'd3,
               S_STORE=3'd4, S_RQDRAIN=3'd5, S_NEXT=3'd6;
    reg [2:0] st;
    reg [9:0]  co, hw;
    reg [13:0] ci_a;                            // Cin counter widened (conv3x3 Cin' up to ~4600)
    reg       feed_vld, feed_vld2;
    reg [4:0] si;
    reg [2:0] drain;                            // H3: widened for +1 read-pipeline
    reg [2:0] rqd;

    wire [MAW-1:0] wrd_addr = (co >> LOGN)*Cin + ci_a;
    wire [MAW-1:0] ard_addr = ci_a*WB + (hw >> LOGN);
    // H3 200MHz: 2-stage URAM read (the 16384-deep amem/wmem cascade clock-to-out -> pe DSP was the
    // worst path). The 2nd register lets Vivado pipeline the URAM cascade. +1 latency (feed_vld2/drain).
    reg  [N*AW-1:0] wq128, aq128, wq128_2, aq128_2;
    always @(posedge clk) begin
        wq128   <= wmem[wrd_addr];
        aq128   <= amem[ard_addr];
        wq128_2 <= wq128;
        aq128_2 <= aq128;
    end

    wire [9:0]   rq_co = co + si;
    wire [127:0] pword = pmem[rq_co[9:1]];
    wire [63:0]  phalf = rq_co[0] ? pword[127:64] : pword[63:0];   // params for rq_co

    always @(posedge clk) begin
        if (!rst_n) begin
            st<=S_IDLE; done<=0; pe_clr<=0; pe_en<=0;
            co<=0; hw<=0; ci_a<=0; feed_vld<=0; feed_vld2<=0; si<=0; drain<=0; rqd<=0; rq_v_r<=0;
        end else begin
            done<=0; pe_clr<=0; pe_en<=0; rq_v_r<=0;
            oa_p1<=oa_p0; oa_p2<=oa_p1; oa_p3<=oa_p2; oa_p4<=oa_p3; oa_p5<=oa_p4;
            cv_p1<=cv_p0; cv_p2<=cv_p1; cv_p3<=cv_p2; cv_p4<=cv_p3; cv_p5<=cv_p4;
            // H3: feed pe_array on feed_vld2 (= feed_vld delayed 1) for the 2-stage URAM read, and keep
            // draining the in-flight feeds across S_FEED/S_DRAIN.
            feed_vld2 <= feed_vld;
            if (feed_vld2) begin
                pe_act = wq128_2;          // act_col[i]=W[co+i] (2nd read stage)
                pe_wt  = aq128_2;          // wt_row[j] =A[ci][hw+j]
                pe_en <= 1;
            end
            case (st)
                S_IDLE: if (start) begin co<=0; hw<=0; st<=S_CLR; end
                S_CLR:  begin pe_clr<=1; ci_a<=0; feed_vld<=0; st<=S_FEED; end
                S_FEED: begin
                    if (ci_a == Cin) begin feed_vld<=1'b0; drain<=0; st<=S_DRAIN; end
                    else begin ci_a<=ci_a+14'd1; feed_vld<=1'b1; end
                end
                S_DRAIN: begin
                    feed_vld<=1'b0; drain<=drain+3'd1;
                    if (drain==3'd4) begin si<=0; pe_ri<=0; st<=S_STORE; end  // BOARD FIX: drain=4 (rev76-proven). Sim ok @2 but board (URAM read +1 latency vs sim) saturates -> generous output margin. @100MHz.
                end
                S_STORE: begin
                    for (t=0;t<N;t=t+1)
                        rq_psum_r[t] <= $signed(pe_row[t*ACCW +: ACCW]) + $signed(phalf[55:24]);
                    rq_m0_r   <= phalf[15:0];
                    rq_shift_r<= phalf[16 +: `SHIFT_W];
                    rq_v_r    <= (co+si) < Cout;
                    oa_p0     <= (co+si)*WB + (hw >> LOGN);
                    for (t=0;t<N;t=t+1) cv_p0[t] <= ((hw + t) < HW);
                    if (si==N-1) begin rqd<=0; st<=S_RQDRAIN; end
                    else begin si<=si+5'd1; pe_ri<=pe_ri+1'b1; end
                end
                S_RQDRAIN: begin
                    rqd<=rqd+3'd1;
                    if (rqd==3'd6) st<=S_NEXT;  // H3: rr_unit 5-stage
                end
                S_NEXT: begin
                    if (hw + N < HW)        begin hw<=hw+N; st<=S_CLR; end
                    else if (co + N < Cout) begin hw<=0; co<=co+N; st<=S_CLR; end
                    else begin done<=1; st<=S_IDLE; end
                end
                default: st<=S_IDLE;
            endcase
        end
    end

    // 128-bit O read-back (registered)
    always @(posedge clk) rd_o_data <= omem[rd_o_addr];
    wire _unused = &{1'b0, pe_o};   // pe_o intentionally unused (single-read port)
endmodule
