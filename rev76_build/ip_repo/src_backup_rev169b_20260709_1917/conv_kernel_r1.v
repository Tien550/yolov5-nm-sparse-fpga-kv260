// conv_kernel_r1.v -- R1 kernel: BRAM adapter around conv_tile_r1 (step 4.1b: PIPELINED load).
// Loader streams IN/WT BRAM at 1 byte/cycle: address is combinational from the issue
// counters, and a 2-deep tag pipe (matching RD_LAT=2) carries the destination indices so
// the returning data is written to the right tile slot. ACT (n_ic*54) and WT (n_ic*9*4)
// are pipelined; the small requant tail (28B) stays simple (3 cyc/byte). vs the 4.1
// 3-cyc/byte loader (~4400 cyc) this loads in ~1500 cyc.
//
// BRAM byte layouts (match encode_block / CHW host tile-stream):
//   IN  : act_tile[ic][kh][col] = IN [ic*54 + kh*18 + col]
//   WT  : wt_tile[g][ic][tap]   = WT [(ic*9+tap)*17 + 1 + g]
//   WT requant tail at Cin*9*17 : per oc 7B = [m0_lo,m0_hi,shift,bias0..3]
`timescale 1ns/1ps
`include "cnn_defines.vh"

module conv_kernel_r1 (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [5:0]  n_ic,            // up to 32 (conv1x1 CIN_CHUNK=32)
    input  wire        keep_psum,
    input  wire        no_flush,
    input  wire        apply_relu,
    input  wire        mode_1x1,      // R3: 1 = conv1x1 (1 tap, 1 row, 256B ACT / Cin*17 WT); 0 = conv3x3
    input  wire        dbg_wt,        // DEBUG: store loaded center weights to OUT instead of result
    output reg         done,

    output wire [5:0]  in_rd128_addr, // IN BRAM WIDE read: 16 bytes/word (RD_LAT=2)
    input  wire [127:0] in_rd128_data,
    output wire [11:0] wt_rd_addr,    // WT BRAM read (RD_LAT=2)
    input  wire [7:0]  wt_rd_data,
    output reg  [9:0]  out_wr_addr,   // OUT BRAM write
    output reg  [7:0]  out_wr_data,
    output reg         out_wr_en
);
    localparam RD_LAT = 2;
    localparam K_IDLE=4'd0, K_ACT=4'd1, K_WT=4'd2, K_RQ=4'd3,
               K_COMPUTE=4'd4, K_WAITC=4'd5, K_STORE=4'd6, K_DONE=4'd7;
    reg [3:0] kstate;

    // ── conv_tile_r1 instance + its load/read ports ─────────────────────────
    reg        t_start;
    reg        ld_act_we;  reg [5:0] ld_act_word; reg [127:0] ld_act_d128;
    reg        ld_wt_we;   reg [1:0] ld_wt_g;   reg [4:0] ld_wt_ic;  reg [3:0] ld_wt_tap;  reg signed [7:0] ld_wt_data;
    reg        ld_rq_we;   reg [1:0] ld_rq_oc;  reg [1:0] ld_rq_field; reg [31:0] ld_rq_data;
    wire       t_done;
    reg  [5:0] t_rd_addr;
    wire signed [7:0] t_rd_out;
    wire signed [7:0] t_rd_wt;

    conv_tile_r1 u_core (
        .clk(clk), .rst_n(rst_n), .start(t_start), .n_ic(n_ic),
        .tap_max(mode_1x1 ? 4'd0 : 4'd8),
        .keep_psum(keep_psum), .no_flush(no_flush), .apply_relu(apply_relu), .done(t_done),
        .ld_act_we(ld_act_we), .ld_act_word(ld_act_word), .ld_act_d128(ld_act_d128),
        .ld_wt_we(ld_wt_we), .ld_wt_g(ld_wt_g), .ld_wt_ic(ld_wt_ic),
        .ld_wt_tap(ld_wt_tap), .ld_wt_data(ld_wt_data),
        .ld_rq_we(ld_rq_we), .ld_rq_oc(ld_rq_oc), .ld_rq_field(ld_rq_field), .ld_rq_data(ld_rq_data),
        .rd_addr(t_rd_addr), .rd_psum(), .rd_out(t_rd_out), .rd_wt(t_rd_wt)
    );

    // ── issue counters + 2-deep tag pipe (shared ACT/WT, interpreted by phase) ─
    reg        issuing;
    reg [5:0]  a_word;                                    // ACT: 16-byte word counter
    reg [4:0]  w_ic; reg [3:0] w_tap; reg [1:0] w_g;      // WT (w_ic up to 31, CIN_CHUNK=32)
    reg        v1, v2; reg [10:0] pay1, pay2;
    // requant (simple loader)
    reg [1:0]  rdw; reg [1:0] q_oc; reg [2:0] q_byte; reg [7:0] rqb [0:6];
    reg [6:0]  st_idx;

    // R3: conv1x1 layout = act[ic][col] @ IN[ic*16+col] (1 row, 16 col); wt entry (ic) @ ic*17;
    // requant tail @ Cin*17.  conv3x3 = act[ic][kh][col] @ IN[ic*54+kh*18+col]; wt @ (ic*9+tap)*17.
    wire [3:0]  wtap_max  = mode_1x1 ? 4'd0  : 4'd8;
    // ACT bytes = n_ic*16 (1x1) or n_ic*54 (3x3); wide loader streams ceil(bytes/16) words
    wire [11:0] act_bytes = mode_1x1 ? ({7'd0,n_ic} << 4) : ({7'd0,n_ic} * 12'd54);
    wire [5:0]  n_word    = (act_bytes + 12'd15) >> 4;     // words to cover the ACT tile
    wire [11:0] rq_base = mode_1x1 ? ({4'd0,n_ic} * 12'd17)
                                   : (({4'd0,n_ic} * 12'd9) * 12'd17);
    assign in_rd128_addr = a_word;
    assign wt_rd_addr = (kstate==K_RQ) ? (rq_base + {7'd0,q_oc}*12'd7 + {9'd0,q_byte})
                      : mode_1x1 ? (({7'd0,w_ic}*12'd17) + 12'd1 + {10'd0,w_g})
                                 : ((({7'd0,w_ic}*12'd9 + {8'd0,w_tap})*12'd17) + 12'd1 + {10'd0,w_g});

    // ── combinational tile-load drive (PIPELINED: stage-2 tag valid → write tile;
    //    conv_tile_r1 registers the write 1 cyc later). pay layout:
    //    ACT pay2[10:7]=ic [6:5]=kh [4:0]=col ; WT pay2[9:8]=g [7:4]=ic [3:0]=tap ─
    always @* begin
        ld_act_we=0; ld_act_word=0; ld_act_d128=0;
        ld_wt_we=0;  ld_wt_g=0;  ld_wt_ic=0;  ld_wt_tap=0;  ld_wt_data=0;
        ld_rq_we=0;  ld_rq_oc=0; ld_rq_field=0; ld_rq_data=0;
        if (kstate==K_ACT && v2) begin   // ACT 16-byte word for pay2 returned (RD_LAT=2)
            ld_act_we=1; ld_act_word=pay2[5:0]; ld_act_d128=in_rd128_data;
        end
        if (kstate==K_WT && v2) begin     // WT data for pay2 returned (RD_LAT=2). pay: g[10:9] ic[8:4] tap[3:0]
            ld_wt_we=1; ld_wt_g=pay2[10:9]; ld_wt_ic=pay2[8:4]; ld_wt_tap=pay2[3:0];
            ld_wt_data=wt_rd_data;
        end
        if (kstate==K_RQ && rdw==RD_LAT) begin
            if (q_byte==3'd1) begin ld_rq_we=1; ld_rq_oc=q_oc; ld_rq_field=2'd0;
                                    ld_rq_data={16'd0, wt_rd_data, rqb[0]}; end
            else if (q_byte==3'd2) begin ld_rq_we=1; ld_rq_oc=q_oc; ld_rq_field=2'd1;
                                    ld_rq_data={27'd0, wt_rd_data[4:0]}; end
            else if (q_byte==3'd6) begin ld_rq_we=1; ld_rq_oc=q_oc; ld_rq_field=2'd2;
                                    ld_rq_data={wt_rd_data, rqb[5], rqb[4], rqb[3]}; end
        end
    end

    always @(posedge clk) begin
        if (!rst_n) begin
            kstate<=K_IDLE; done<=0; t_start<=0; out_wr_en<=0;
            issuing<=0; a_word<=0; w_ic<=0; w_tap<=0; w_g<=0;
            v1<=0; v2<=0; pay1<=0; pay2<=0; rdw<=0; q_oc<=0; q_byte<=0; st_idx<=0;
            out_wr_addr<=0; out_wr_data<=0; t_rd_addr<=0;
        end else begin
            done<=0; t_start<=0; out_wr_en<=0;
            case (kstate)
                K_IDLE: if (start) begin
                    a_word<=0; issuing<=1'b1; v1<=1'b0; v2<=1'b0;
                    kstate<=K_ACT;
                end
                // ── WIDE ACT load (16 bytes/cycle: issue 1 word/cyc, drain 2-deep tag pipe;
                //    safe because unified gates kr1_start on dma-engine idle) ──
                K_ACT: begin
                    v1 <= 1'b0;
                    if (issuing) begin
                        v1   <= 1'b1;
                        pay1 <= {5'd0, a_word};
                        if (a_word == n_word - 6'd1) issuing<=1'b0;
                        else a_word <= a_word + 6'd1;
                    end
                    v2 <= v1; pay2 <= pay1;
                    if (!issuing && !v1 && !v2) begin
                        w_ic<=0; w_tap<=0; w_g<=0; issuing<=1'b1; kstate<=K_WT;
                    end
                end
                // ── PIPELINED WT load (1 byte/cycle) ──
                K_WT: begin
                    v1 <= 1'b0;
                    if (issuing) begin
                        v1   <= 1'b1;
                        pay1 <= {w_g, w_ic, w_tap};       // [10:9]=g [8:4]=ic [3:0]=tap (w_ic 5-bit)
                        if (w_g==2'd3) begin
                            w_g<=0;
                            if (w_tap==wtap_max) begin
                                w_tap<=0;
                                if (w_ic==n_ic[4:0]-5'd1) issuing<=1'b0;
                                else w_ic<=w_ic+5'd1;
                            end else w_tap<=w_tap+4'd1;
                        end else w_g<=w_g+2'd1;
                    end
                    v2 <= v1; pay2 <= pay1;
                    if (!issuing && !v1 && !v2) begin
                        // requant only USED on flush (last ic-chunk). Middle chunks (no_flush)
                        // accumulate without requant → skip the 84-cyc K_RQ load. Validated by
                        // tb_b0 (NICB=2: chunk0 no_flush skips, chunk1 flush loads its own rq →
                        // 0/384). Big win: 15/16 conv1x1 sub-kicks skip K_RQ.
                        if (no_flush) kstate<=K_COMPUTE;
                        else begin q_oc<=0; q_byte<=0; rdw<=0; kstate<=K_RQ; end
                    end
                end
                // ── simple requant tail load (3 cyc/byte) ──
                K_RQ: begin
                    if (rdw==RD_LAT) begin
                        rqb[q_byte] <= wt_rd_data;
                        rdw<=0;
                        if (q_byte==3'd6) begin
                            q_byte<=0;
                            if (q_oc==2'd3) kstate<=K_COMPUTE;
                            else q_oc<=q_oc+2'd1;
                        end else q_byte<=q_byte+3'd1;
                    end else rdw<=rdw+2'd1;
                end
                K_COMPUTE: begin t_start<=1; kstate<=K_WAITC; end
                K_WAITC: if (t_done) begin st_idx<=0; t_rd_addr<=6'd0; kstate<=K_STORE; end
                K_STORE: begin
                    if (no_flush) begin done<=1; kstate<=K_IDLE; end
                    else begin
                        out_wr_en<=1; out_wr_addr<={3'b0, st_idx}; out_wr_data<=(dbg_wt ? t_rd_wt : t_rd_out);
                        if (st_idx==7'd63) kstate<=K_DONE;
                        else begin st_idx<=st_idx+7'd1; t_rd_addr<=st_idx[5:0]+6'd1; end
                    end
                end
                K_DONE: begin done<=1; kstate<=K_IDLE; end
                default: kstate<=K_IDLE;
            endcase
        end
    end
endmodule
