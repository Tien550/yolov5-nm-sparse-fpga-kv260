// tb_graphop_engine.v -- Testbench mô phỏng cho graphop_engine.v, dùng act_uram THẬT
// (không mock) để timing RD_LAT khớp chính xác hệ thống thật. So kết quả với golden
// từ gen_graphop_tb.py (rescale_int8/resid_cat bit-exact). Chạy: xvlog+xelab+xsim.
`timescale 1ns/1ps
`include "tb_params.vh"

module tb_graphop_engine;
    reg clk = 0; reg rst_n = 0;
    always #5 clk = ~clk;   // 100MHz mô phỏng (không cần khớp tần số thật, chỉ test logic)

    // Chuyển mạch act_uram giữa host-preload/readback (=0) và DUT chạy thật (=1)
    reg dut_drives_au;

    // ── graphop_engine (DUT) — khai báo dây TRƯỚC (act_uram mux ở dưới cần dùng) ──
    reg         go_start;
    wire        go_done;
    reg         go_resid_mode;
    reg  [20:0] go_addr_a, go_addr_b, go_addr_c, go_addr_dst;
    reg  [15:0] go_n_pos;
    reg  [7:0]  go_ch_ab, go_ch_c;
    wire [20:0] go_act_rd_addr;
    wire [16:0] go_act_wr_waddr; wire [127:0] go_act_wr_wdata; wire go_act_wr_wen;
    wire [7:0]  au_rd_data;   // (act_uram khai báo dưới — forward-ref cho port DUT)

    graphop_engine u_dut (
        .clk(clk), .rst_n(rst_n),
        .start(go_start), .done(go_done),
        .resid_mode(go_resid_mode),
        .addr_a(go_addr_a), .addr_b(go_addr_b), .addr_c(go_addr_c), .addr_dst(go_addr_dst),
        .n_pos(go_n_pos), .ch_ab(go_ch_ab), .ch_c(go_ch_c),
        .m0_a(`TB_M0_A), .sh_a(`TB_SH_A),
        .m0_b(`TB_M0_B), .sh_b(`TB_SH_B),
        .m0_c(`TB_M0_C), .sh_c(`TB_SH_C),
        .act_rd_addr(go_act_rd_addr), .act_rd_data(au_rd_data),
        .act_wr_waddr(go_act_wr_waddr), .act_wr_wdata(go_act_wr_wdata), .act_wr_wen(go_act_wr_wen)
    );

    // Giám sát MỌI lần ghi thật sự (không giới hạn cycle) — debug bắt bug
    always @(posedge clk) if (go_act_wr_wen)
        $display("  WRITE t=%0t waddr=%0d wdata=%h", $time, go_act_wr_waddr, go_act_wr_wdata);

    // ── act_uram (module thật) ──
    // host_* = biến DUY NHẤT host (initial block) được ghi; au_* = wire mux liên tục
    // (KHÔNG dùng reg bị 2 procedural block cùng ghi — bài học bug testbench vừa bắt được).
    reg  [16:0] host_wr_waddr; reg [127:0] host_wr_wdata; reg host_wr_wen;
    reg  [20:0] host_rd_baddr;
    wire [16:0] au_wr_waddr = dut_drives_au ? go_act_wr_waddr : host_wr_waddr;
    wire [127:0] au_wr_wdata = dut_drives_au ? go_act_wr_wdata : host_wr_wdata;
    wire au_wr_wen = dut_drives_au ? go_act_wr_wen : host_wr_wen;
    wire [20:0] au_rd_baddr = dut_drives_au ? go_act_rd_addr : host_rd_baddr;
    act_uram u_au (
        .clk(clk),
        .wr_waddr(au_wr_waddr), .wr_wdata(au_wr_wdata), .wr_wen(au_wr_wen),
        .rd_baddr(au_rd_baddr), .rd_data(au_rd_data),
        .rd_wsel(1'b0), .rd_waddr(17'd0), .rd_wdata()
    );

    // ── địa chỉ vùng nhớ test (byte) ──
    // Rev200: INPUT TILED (hex đã tiled từ gen). A/B 1024B (cpad 32, N_IC=2), C 512B (cpad 16).
    // Offset chia hết 256 (chuẩn region full-flow).
    localparam ADDR_A   = 21'd0;
    localparam ADDR_B   = 21'd1024;
    localparam ADDR_C   = 21'd2048;
    localparam ADDR_DST = 21'd2560;

    integer i, errors;
    reg [7:0] mem_a [0:`TB_A_BYTES-1];
    reg [7:0] mem_b [0:`TB_B_BYTES-1];
    reg [7:0] mem_c [0:`TB_C_BYTES-1];
    reg [7:0] gold_resid [0:(`TB_CH_AB+`TB_CH_C)*`TB_N_POS-1];
    reg [7:0] gold_cat   [0:(`TB_CH_AB+`TB_CH_C)*`TB_N_POS-1];

    // (bỏ task truyền mảng — xelab không cho unpacked array width khác nhau qua task port;
    //  viết thẳng 3 vòng lặp inline trong initial block, tránh vấn đề đó hoàn toàn)
    integer w, b;
    reg [127:0] pre_word;

    initial begin
        $readmemh("tb_a.hex", mem_a);
        $readmemh("tb_b.hex", mem_b);
        $readmemh("tb_c.hex", mem_c);
        $readmemh("tb_gold_resid.hex", gold_resid);
        $readmemh("tb_gold_cat.hex", gold_cat);

        errors = 0;
        dut_drives_au = 1'b0;
        go_start = 1'b0;
        rst_n = 1'b0;
        repeat (5) @(posedge clk); #1;
        rst_n = 1'b1;
        repeat (2) @(posedge clk); #1;

        // ── preload A,B,C vào act_uram (host ghi, dut_drives_au=0) ──
        // ⚠️ #1 SAU MỖI @(posedge clk) trước khi gán tín hiệu — tránh đua tranh
        // blocking-assignment giữa testbench và always@(posedge clk) của DUT/act_uram
        // (bài học bug: thiếu #1 khiến DUT không bao giờ thấy được xung start).
        for (w = 0; w < `TB_A_BYTES/16; w = w+1) begin
            pre_word = 128'd0;
            for (b = 0; b < 16; b = b+1) pre_word[b*8 +: 8] = mem_a[w*16+b];
            @(posedge clk); #1;
            host_wr_waddr = (ADDR_A>>4) + w[16:0]; host_wr_wdata = pre_word; host_wr_wen = 1'b1;
        end
        @(posedge clk); #1; host_wr_wen = 1'b0;
        for (w = 0; w < `TB_B_BYTES/16; w = w+1) begin
            pre_word = 128'd0;
            for (b = 0; b < 16; b = b+1) pre_word[b*8 +: 8] = mem_b[w*16+b];
            @(posedge clk); #1;
            host_wr_waddr = (ADDR_B>>4) + w[16:0]; host_wr_wdata = pre_word; host_wr_wen = 1'b1;
        end
        @(posedge clk); #1; host_wr_wen = 1'b0;
        for (w = 0; w < `TB_C_BYTES/16; w = w+1) begin
            pre_word = 128'd0;
            for (b = 0; b < 16; b = b+1) pre_word[b*8 +: 8] = mem_c[w*16+b];
            @(posedge clk); #1;
            host_wr_waddr = (ADDR_C>>4) + w[16:0]; host_wr_wdata = pre_word; host_wr_wen = 1'b1;
        end
        @(posedge clk); #1; host_wr_wen = 1'b0;
        repeat (3) @(posedge clk); #1;

        // ══ TEST 1: resid_cat (resid_mode=1) ══
        dut_drives_au = 1'b1;
        go_resid_mode = 1'b1;
        go_addr_a = ADDR_A; go_addr_b = ADDR_B; go_addr_c = ADDR_C; go_addr_dst = ADDR_DST;
        go_n_pos  = `TB_N_POS; go_ch_ab = `TB_CH_AB; go_ch_c = `TB_CH_C;
        $display("DEBUG truoc start: ch_ab=%0d ch_c=%0d n_pos=%0d resid=%0d addr_a=%0d addr_dst=%0d",
                  go_ch_ab, go_ch_c, go_n_pos, go_resid_mode, go_addr_a, go_addr_dst);
        @(posedge clk); #1; go_start = 1'b1;
        @(posedge clk); #1; go_start = 1'b0;
        wait (go_done);
        @(posedge clk); #1;
        dut_drives_au = 1'b0;
        repeat (3) @(posedge clk); #1;
        // DEBUG: xem thẳng nội dung mem[] tại DST + vùng A (kiểm tra preload lẫn DUT-write)
        $display("DEBUG mem[ADDR_A>>4]=%h", u_au.mem[ADDR_A>>4]);
        $display("DEBUG mem[ADDR_DST>>4]=%h", u_au.mem[ADDR_DST>>4]);
        $display("DEBUG mem[(ADDR_DST>>4)+1]=%h", u_au.mem[(ADDR_DST>>4)+1]);

        // đọc lại kết quả qua act_uram byte-read (host đọc, dut_drives_au=0)
        for (i = 0; i < (`TB_CH_AB+`TB_CH_C)*`TB_N_POS; i = i+1) begin
            host_rd_baddr = ADDR_DST + i[20:0];
            repeat (4) @(posedge clk); #1;   // RD_LAT=3 + 1 dư an toàn
            if (au_rd_data !== gold_resid[i]) begin
                $display("[FAIL resid_cat] byte %0d: HW=%02x GOLD=%02x", i, au_rd_data, gold_resid[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("[PASS] resid_cat: %0d byte khop bit-exact", (`TB_CH_AB+`TB_CH_C)*`TB_N_POS);

        // ══ TEST 2: cat thường (resid_mode=0, dùng A và C) ══
        dut_drives_au = 1'b1;
        go_resid_mode = 1'b0;
        go_addr_a = ADDR_A; go_addr_b = ADDR_B; go_addr_c = ADDR_C; go_addr_dst = ADDR_DST;
        go_n_pos  = `TB_N_POS; go_ch_ab = `TB_CH_AB; go_ch_c = `TB_CH_C;
        @(posedge clk); #1; go_start = 1'b1;
        @(posedge clk); #1; go_start = 1'b0;
        wait (go_done);
        @(posedge clk); #1;
        dut_drives_au = 1'b0;
        repeat (3) @(posedge clk); #1;

        for (i = 0; i < (`TB_CH_AB+`TB_CH_C)*`TB_N_POS; i = i+1) begin
            host_rd_baddr = ADDR_DST + i[20:0];
            repeat (4) @(posedge clk); #1;
            if (au_rd_data !== gold_cat[i]) begin
                $display("[FAIL cat] byte %0d: HW=%02x GOLD=%02x", i, au_rd_data, gold_cat[i]);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("[PASS] cat thuong: da khop tiep");

        if (errors == 0) $display("=== TB_GRAPHOP_ALL_PASS ===");
        else             $display("=== TB_GRAPHOP_FAILED: %0d loi ===", errors);
        $finish;
    end

    initial begin
        #2000000; $display("=== TB_GRAPHOP_TIMEOUT ==="); $finish;   // Rev171: test lớn hơn (32ch×32pos×2)
    end
endmodule
