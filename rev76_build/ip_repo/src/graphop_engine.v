// graphop_engine.v -- Rev170: CAT / RESID_CAT trên FPGA (đưa graph-op CPU lên PL).
// Kích hoạt khi LAYER_TYPE=4 (CONV_TYPE_GRAPHOP), độc lập hoàn toàn với FSM conv1x1/c3/dw
// hiện có (không đụng config_decoder/swc_en) — engine riêng, đọc/ghi thẳng act_uram.
//
// THIẾT KẾ V2 (viết lại sau khi tự bắt bug timing ở bản đầu — Rev170 review):
// TUẦN TỰ TRIỆT ĐỂ, mỗi độ trễ pipeline có bộ đếm RIÊNG, không chồng lấn, không tái dùng
// counter cho 2 việc khác nhau. Chậm hơn/giá trị (nhiều cycle hơn) nhưng dễ chứng minh đúng.
//
// Toán học (khớp inference_cpu.py rescale_int8 + resid_cat, bit-exact):
//   rescale_int8(x, fs, ts) = round(x * fs/ts) clip [-128,127]
//     -> HW: (x_int8 sign-extend 32b) * M0(uint16) round-shift == rr_unit path (apply_relu=0)
//   cat(a,b)      = concat(rescale(a), rescale(b))
//   resid_cat(cv1,mc2,cv2) = concat( clip(rescale(cv1)+rescale(mc2)), rescale(cv2) )
//
// Rev171: OUTPUT ghi TILED position-major (word = dst_word + pt*Cpad + C_global) để conv1x1
// BI (consumer, vd L05) đọc thẳng act_uram.
// Rev200 (ZCU102 full-flow): INPUT cũng TILED (chuẩn hoá — producer conv1x1 RESOUT scatter
// tiled qua LS deint_tbase). byte(p,c) = base + (p[15:4]*cpad_src + c)*16 + p[3:0].
// cpad_src RIÊNG từng nguồn: A/B = ceil(ch_ab/16)*16 ; C = ceil(ch_c/16)*16.
// n_pos KHÔNG cần chia hết 16 nữa (tiled sp-pad chuẩn): host round n_pos lên bội 16,
// lane pad chứa rác — consumer conv tính trên pad rồi bỏ (chuẩn P_pad có sẵn).
`timescale 1ns/1ps
`include "cnn_defines.vh"

module graphop_engine (
    input  wire        clk, rst_n,
    input  wire        start,        // pulse 1 cycle -> bắt đầu 1 kick graphop
    output reg         done,         // pulse 1 cycle khi xong (giống ap_done kernel)

    input  wire        resid_mode,   // 1=resid_cat (cộng A+B) ; 0=cat thường (không cộng)
    input  wire [20:0] addr_a, addr_b, addr_c, addr_dst,  // byte addr trong act_uram
    input  wire [15:0] n_pos,        // số vị trí không gian (PHẢI chia hết 16 — xem trên)
    input  wire [7:0]  ch_ab,        // số kênh nhóm A/B (= số kênh "bott")
    input  wire [7:0]  ch_c,         // số kênh nhóm C (nửa sau concat)
    input  wire [15:0] m0_a, input wire [4:0] sh_a,
    input  wire [15:0] m0_b, input wire [4:0] sh_b,
    input  wire [15:0] m0_c, input wire [4:0] sh_c,

    // ── act_uram byte-read port (dùng chung — mux ở unified_acc_top, graphop chỉ chạy
    //    khi conv-engine không kick, không tranh chấp) ──
    output reg  [20:0] act_rd_addr,
    input  wire [7:0]  act_rd_data,     // RD_LAT=3 tính từ act_rd_addr (act_uram.v)
    // ── act_uram word-write port (mux ở unified_acc_top) ──
    output reg  [16:0] act_wr_waddr,
    output reg  [127:0] act_wr_wdata,
    output reg          act_wr_wen
);
    localparam RD_LAT = 3;   // act_uram byte-read latency (act_uram.v RD_LAT=3)
    localparam RR_LAT = 5;   // rr_unit pipeline: tmp48_s1,tmp48_2a,tmp48_s2,arelu_3a,out_s3 = 5 tầng

    // ── State: mỗi bước 1 state riêng, KHÔNG chồng lấn 2 việc trong cùng 1 counter ──
    localparam S_IDLE      = 4'd0;
    localparam S_RD_A_ISSUE= 4'd1;  // đặt act_rd_addr cho A (hoặc C khi phase_c)
    localparam S_RD_A_WAIT = 4'd2;  // chờ RD_LAT, bắt act_rd_data -> byte_a
    localparam S_RS_A_ISSUE= 4'd3;  // feed rr_unit cho A
    localparam S_RS_A_WAIT = 4'd4;  // chờ RR_LAT, bắt rr_out -> val_a
    localparam S_RS_A_CAP  = 4'd5;  // bắt rr_out (=rescale A) — 1 cycle SAU khi wcnt chạm RR_LAT-1
    localparam S_RD_B_ISSUE= 4'd6;  // (chỉ AB+resid) đặt act_rd_addr cho B
    localparam S_RD_B_WAIT = 4'd7;
    localparam S_RS_B_ISSUE= 4'd8;
    localparam S_RS_B_WAIT = 4'd9;  // -> S_COMBINE THẲNG (S_COMBINE đọc rr_out trực tiếp/tổ hợp,
                                    // KHÔNG bắt vào thanh ghi cùng nhịp check -> an toàn, không
                                    // cần state CAP riêng như nhánh A (A phải giữ val_a qua nhiều
                                    // state sau nên bắt buộc có 1 state CAP để lệch đúng 1 cycle).
    localparam S_COMBINE   = 4'd10; // val_a (+val_b nếu resid) -> byte_result, clip
    localparam S_STORE     = 4'd11; // ghi byte_result vào pack_buf; flush word nếu đủ/hết kênh
    localparam S_ADVANCE   = 4'd12; // tăng pos/ch, chuyển pha AB->C hoặc DONE
    localparam S_DONE      = 4'd13;

    reg [3:0]  st;
    reg [3:0]  wcnt;                  // bộ đếm DÙNG RIÊNG cho state đang chờ (RD_LAT hoặc RR_LAT)
    reg        phase_c;               // 0=nhóm AB, 1=nhóm C
    reg [15:0] pos_i;
    reg [7:0]  ch_i, ch_lim;
    reg [20:0] rdbase_1, rdbase_2;     // A/C base ; B base (chỉ AB+resid dùng rdbase_2)
    reg [16:0] dstw;                   // Rev171: dst WORD base (=addr_dst>>4) — output TILED
    reg [9:0]  cpad;                   // Rev171: padded output channels = ceil((ch_ab+ch_c)/16)*16
    reg [9:0]  cpad_ab, cpad_c;        // Rev200: padded channels NGUỒN (A/B ; C) — input TILED
    wire [9:0] cpad_src = phase_c ? cpad_c : cpad_ab;
    // Rev200 input TILED addr: word = p[15:4]*cpad_src + ch_i ; byte = base + {word,4'b0} + p[3:0]
    wire [16:0] rd_word_i = pos_i[15:4] * {7'd0, cpad_src} + {9'd0, ch_i};

    reg signed [31:0] rr_in;
    reg  [15:0]       rr_m0;
    reg  [4:0]        rr_sh;
    reg               rr_valid_in;
    wire signed [7:0] rr_out;
    wire              rr_valid_out;   // không dùng (đếm cycle cố định thay vì chờ valid)
    rr_unit u_rr_go (
        .clk(clk), .rst_n(rst_n),
        .psum_in(rr_in), .rq_scale(rr_m0), .rq_shift(rr_sh),
        .apply_relu(1'b0), .valid_in(rr_valid_in),
        .data_out(rr_out), .valid_out(rr_valid_out)
    );

    reg signed [7:0] val_a;
    reg [127:0]      pack_buf;
    reg [3:0]         pack_cnt;        // 0..15 vị trí byte trong word đang gom

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; done <= 1'b0; act_wr_wen <= 1'b0; rr_valid_in <= 1'b0;
            wcnt <= 4'd0; pack_cnt <= 4'd0;
        end else begin
            done <= 1'b0; act_wr_wen <= 1'b0; rr_valid_in <= 1'b0;

            case (st)
                S_IDLE: if (start) begin
                    phase_c  <= 1'b0;
                    ch_lim   <= ch_ab;
                    rdbase_1 <= addr_a; rdbase_2 <= addr_b;
                    dstw <= addr_dst[20:4];                                    // Rev171: dst WORD base
                    cpad <= ({2'd0,ch_ab} + {2'd0,ch_c} + 10'd15) & 10'h3F0;  // ceil((ab+c)/16)*16
                    cpad_ab <= ({2'd0,ch_ab} + 10'd15) & 10'h3F0;             // Rev200 src pads
                    cpad_c  <= ({2'd0,ch_c}  + 10'd15) & 10'h3F0;
                    pos_i <= 16'd0; ch_i <= 8'd0; pack_cnt <= 4'd0;
                    st <= S_RD_A_ISSUE;
                end

                // ── đọc A (hoặc C ở pha C) — Rev200: TILED addr ──
                S_RD_A_ISSUE: begin
                    act_rd_addr <= rdbase_1 + {rd_word_i[16:0], 4'b0} + {17'd0, pos_i[3:0]};
                    wcnt <= 4'd0;
                    st   <= S_RD_A_WAIT;
                end
                S_RD_A_WAIT: begin
                    wcnt <= wcnt + 4'd1;
                    if (wcnt == RD_LAT[3:0]-4'd1) st <= S_RS_A_ISSUE;  // act_rd_data hợp lệ đúng cycle này
                end

                S_RS_A_ISSUE: begin
                    rr_in       <= {{24{act_rd_data[7]}}, act_rd_data};
                    rr_m0       <= phase_c ? m0_c : m0_a;
                    rr_sh       <= phase_c ? sh_c : sh_a;
                    rr_valid_in <= 1'b1;
                    wcnt        <= 4'd0;
                    st          <= S_RS_A_WAIT;
                end
                S_RS_A_WAIT: begin
                    wcnt <= wcnt + 4'd1;
                    // ⚠️ KHÔNG bắt rr_out ở đây — tại cycle wcnt==RR_LAT-1, rr_out CHƯA hợp lệ
                    // (hợp lệ đúng 1 cycle SAU, khi state đã chuyển). Chỉ chuyển state ở đây;
                    // S_RS_A_CAP (state kế) mới đọc rr_out — bài học bug tự bắt được (Rev170).
                    if (wcnt == RR_LAT[3:0]-4'd1) st <= S_RS_A_CAP;
                end
                S_RS_A_CAP: begin
                    val_a <= rr_out;   // đúng cycle rr_out hợp lệ (1 cycle sau check ở trên)
                    st    <= (!phase_c && resid_mode) ? S_RD_B_ISSUE : S_COMBINE;
                end

                // ── (chỉ nhóm AB + resid_mode) đọc B rồi rescale B — Rev200: TILED ──
                S_RD_B_ISSUE: begin
                    act_rd_addr <= rdbase_2 + {rd_word_i[16:0], 4'b0} + {17'd0, pos_i[3:0]};
                    wcnt <= 4'd0;
                    st   <= S_RD_B_WAIT;
                end
                S_RD_B_WAIT: begin
                    wcnt <= wcnt + 4'd1;
                    if (wcnt == RD_LAT[3:0]-4'd1) st <= S_RS_B_ISSUE;
                end
                S_RS_B_ISSUE: begin
                    rr_in       <= {{24{act_rd_data[7]}}, act_rd_data};
                    rr_m0       <= m0_b;
                    rr_sh       <= sh_b;
                    rr_valid_in <= 1'b1;
                    wcnt        <= 4'd0;
                    st          <= S_RS_B_WAIT;
                end
                S_RS_B_WAIT: begin
                    wcnt <= wcnt + 4'd1;
                    if (wcnt == RR_LAT[3:0]-4'd1) st <= S_COMBINE;  // rr_out = rescale(B)
                end

                // ── cộng (nếu AB+resid) hoặc dùng thẳng val_a (cat thường / nhóm C) ──
                S_COMBINE: begin : comb_blk
                    reg signed [8:0] sum9;
                    reg signed [7:0] byte_result;
                    if (!phase_c && resid_mode) begin
                        sum9 = {val_a[7],val_a} + {rr_out[7],rr_out};
                        byte_result = (sum9 > 9'sd127) ? 8'sd127
                                    : (sum9 < -9'sd128) ? -8'sd128
                                    : sum9[7:0];
                    end else begin
                        byte_result = val_a;   // cat thường / nhóm C: rescale đơn (val_a đã chứa kết quả)
                    end
                    pack_buf[pack_cnt*8 +: 8] <= byte_result;
                    st <= S_STORE;
                end

                // ── ghi word 128-bit theo layout TILED position-major (khớp conv1x1 BI đọc act_uram):
                //    word = dstw + pt*cpad + C_global ; pt = pos_i>>4 (tile vừa gom đủ 16 byte);
                //    C_global = (phase_c? ch_ab:0)+ch_i. (Rev171: đổi CHW→tiled cho consumer 1x1) ──
                S_STORE: begin : store_blk
                    reg [9:0]  cg;
                    reg [23:0] wr_word;
                    cg      = phase_c ? ({2'd0,ch_ab} + {2'd0,ch_i}) : {2'd0,ch_i};
                    wr_word = {7'd0,dstw} + ({12'd0,pos_i[15:4]} * {14'd0,cpad}) + {14'd0,cg};
                    if (pack_cnt == 4'd15) begin
                        act_wr_waddr <= wr_word[16:0];
                        act_wr_wdata <= pack_buf;
                        act_wr_wen   <= 1'b1;
                        pack_cnt     <= 4'd0;
                    end else begin
                        pack_cnt <= pack_cnt + 4'd1;
                    end
                    st <= S_ADVANCE;
                end

                S_ADVANCE: begin
                    if (pos_i == n_pos-16'd1) begin
                        // hết vị trí của kênh này — n_pos chia hết 16 (ràng buộc thiết kế) nên
                        // pack_cnt LUÔN vừa quay về 0 ở S_STORE (không cần flush thừa ở đây).
                        pos_i <= 16'd0;
                        if (ch_i == ch_lim-8'd1) begin
                            if (!phase_c) begin
                                phase_c  <= 1'b1;
                                ch_lim   <= ch_c;
                                rdbase_1 <= addr_c;   // Rev171: chỉ đổi read base; ghi TILED dùng C_global
                                ch_i     <= 8'd0;
                                st       <= S_RD_A_ISSUE;
                            end else st <= S_DONE;
                        end else begin
                            ch_i <= ch_i + 8'd1;
                            st   <= S_RD_A_ISSUE;
                        end
                    end else begin
                        pos_i <= pos_i + 16'd1;
                        st    <= S_RD_A_ISSUE;
                    end
                end

                S_DONE: begin done <= 1'b1; st <= S_IDLE; end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
