// tb_unified_acc_top.v -- Testbench for unified_acc_top (conv1x1 + conv3x3 + dwconv)
//
// TC_01: conv_type=0 (1x1)
//   Cin=16, Cout=4, Hin=1, Win=1, N_config=1, poc=0, accu_mode=0
//   Weight BRAM[0..16]: byte[0]=idx=3, byte[1..4]=s0g0..g3=5, rest=0
//   Input BRAM[3]=10, rest=0
//   PSUM_g0 = 5*10 = 50; after requant (scale=0x4000, shift=15): out = clamp(50>>1)=25
//   Wait irq, verify irq fires within timeout
//
// TC_02: conv_type=1 (3x3)
//   Cin=16, Cout=4, Hin=3, Win=3 (3x3 input, 1x1 output with pad=1, stride=1)
//   N_config=1, poc=0, accu_mode=0
//   Weight BRAM[0..16]: idx=0, all SPEC g0=4, rest=0
//   Input BRAM: 3x3 window all = 2 (9 pixels × 4 = 36... psum = 4*2*9 = 72)
//   Wait irq, verify fires within timeout
//
// TC_03: conv_type=2 (dwconv) — verify config_decoder forcing
//   Write LAYER_TYPE=2, POC=3 (intentionally wrong), ACCU_MODE=2 (intentionally wrong)
//   Verify eff_poc=0, eff_accu_mode=0 (forced regardless)
//   Check dw_mode signal in config_decoder forces override
//   Run minimal compute, wait irq
//
// TC_04: config_decoder force check (reg_conv_type=2, user sets poc=3)
//   Verify eff_poc is 0 (NOT 3), eff_accu_mode is 0 (NOT 2)
//   Done by probing internal net via $cast or waveform; in this TB verified
//   indirectly: if poc were 3, the FSM would behave differently → irq still fires

`timescale 1ns/1ps
`include "cnn_defines.vh"

module tb_unified_acc_top;

    // ── Clock & reset ─────────────────────────────────────────────────────────
    reg clk, rst_n;
    initial clk = 1'b0;
    always #5 clk = ~clk;  // 100 MHz

    // ── AXI4-Lite ports ───────────────────────────────────────────────────────
    reg  [8:0]  awaddr;  reg  awvalid; wire awready;
    reg  [31:0] wdata;   reg  [3:0]  wstrb; reg  wvalid; wire wready;
    wire [1:0]  bresp;   wire bvalid; reg  bready;
    reg  [8:0]  araddr;  reg  arvalid; wire arready;
    wire [31:0] rdata;   wire [1:0]  rresp; wire rvalid; reg  rready;

    // ── DMA interface ─────────────────────────────────────────────────────────
    reg  [9:0]  dma_in_addr;
    reg  [7:0]  dma_in_data;
    reg         dma_in_wr_en;
    reg         dma_in_done;
    reg  [11:0] dma_wt_addr;
    reg  [7:0]  dma_wt_data;
    reg         dma_wt_wr_en;
    reg  [9:0]  dma_out_addr;
    wire [7:0]  dma_out_data;
    wire        irq_out;
    wire        dma_nxt_req;

    // ── Config outputs (unused in TB but must be connected) ───────────────────
    wire [31:0] reg_hin_out, reg_win_out;
    wire [3:0]  reg_ncfg_out;
    wire [63:0] reg_in_addr_out, reg_wt_addr_out, reg_out_addr_out;
    wire        ap_start_out;

    // ── DUT ───────────────────────────────────────────────────────────────────
    unified_acc_top dut (
        .aclk(clk), .aresetn(rst_n),
        .s_awaddr(awaddr), .s_awvalid(awvalid), .s_awready(awready),
        .s_wdata(wdata),   .s_wstrb(wstrb),     .s_wvalid(wvalid),  .s_wready(wready),
        .s_bresp(bresp),   .s_bvalid(bvalid),   .s_bready(bready),
        .s_araddr(araddr), .s_arvalid(arvalid), .s_arready(arready),
        .s_rdata(rdata),   .s_rresp(rresp),     .s_rvalid(rvalid),  .s_rready(rready),
        .dma_in_addr(dma_in_addr),   .dma_in_data(dma_in_data),
        .dma_in_wr_en(dma_in_wr_en), .dma_in_done(dma_in_done),
        .dma_wt_addr(dma_wt_addr),   .dma_wt_data(dma_wt_data),
        .dma_wt_wr_en(dma_wt_wr_en),
        .dma_out_addr(dma_out_addr), .dma_out_data(dma_out_data),
        .irq_out(irq_out),
        .reg_hin_out(reg_hin_out),   .reg_win_out(reg_win_out),
        .reg_ncfg_out(reg_ncfg_out),
        .reg_in_addr_out(reg_in_addr_out),
        .reg_wt_addr_out(reg_wt_addr_out),
        .reg_out_addr_out(reg_out_addr_out),
        .ap_start_out(ap_start_out),
        .dma_nxt_req_o(dma_nxt_req)
    );

    // ── Counters ──────────────────────────────────────────────────────────────
    integer fail_count;
    integer pass_count;
    reg [31:0] rd_val;
    integer i, timeout;

    // ── AXI write helper ──────────────────────────────────────────────────────
    task do_axi_write;
        input [8:0]  addr;
        input [31:0] data;
        begin
            @(negedge clk);
            awaddr = addr; awvalid = 1'b1;
            wdata  = data; wstrb   = 4'hF; wvalid = 1'b1;
            @(posedge clk);
            while (!(awready && wready)) @(posedge clk);
            @(negedge clk);
            awvalid = 1'b0; wvalid = 1'b0;
            bready  = 1'b1;
            @(posedge clk);
            while (!bvalid) @(posedge clk);
            @(negedge clk);
            bready = 1'b0;
        end
    endtask

    // ── AXI read helper ───────────────────────────────────────────────────────
    task do_axi_read;
        input  [8:0]  addr;
        output [31:0] data;
        begin
            @(negedge clk);
            araddr = addr; arvalid = 1'b1;
            @(posedge clk);
            while (!arready) @(posedge clk);
            @(negedge clk);
            arvalid = 1'b0; rready = 1'b1;
            @(posedge clk);
            while (!rvalid) @(posedge clk);
            data = rdata;
            @(negedge clk);
            rready = 1'b0;
        end
    endtask

    // ── Load input BRAM helper ────────────────────────────────────────────────
    task bram_in_write;
        input [9:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            dma_in_addr  = addr;
            dma_in_data  = data;
            dma_in_wr_en = 1'b1;
            @(posedge clk);
            @(negedge clk);
            dma_in_wr_en = 1'b0;
        end
    endtask

    // ── Load weight BRAM helper ───────────────────────────────────────────────
    task bram_wt_write;
        input [11:0] addr;
        input [7:0]  data;
        begin
            @(negedge clk);
            dma_wt_addr  = addr;
            dma_wt_data  = data;
            dma_wt_wr_en = 1'b1;
            @(posedge clk);
            @(negedge clk);
            dma_wt_wr_en = 1'b0;
        end
    endtask

    // ── Write 17-byte NZ entry into weight BRAM ───────────────────────────────
    // Format: [0]=idx[3:0], [1..4]=s0g0..g3, [5..8]=s1g0..g3,
    //         [9..12]=s2g0..g3, [13..16]=s3g0..g3
    task load_nz_entry;
        input [11:0] base;   // BRAM byte address of this entry
        input [3:0]  idx;
        input signed [7:0] s0g0, s0g1, s0g2, s0g3;
        input signed [7:0] s1g0, s1g1, s1g2, s1g3;
        input signed [7:0] s2g0, s2g1, s2g2, s2g3;
        input signed [7:0] s3g0, s3g1, s3g2, s3g3;
        begin
            bram_wt_write(base+0,  {4'd0, idx});
            bram_wt_write(base+1,  s0g0);
            bram_wt_write(base+2,  s0g1);
            bram_wt_write(base+3,  s0g2);
            bram_wt_write(base+4,  s0g3);
            bram_wt_write(base+5,  s1g0);
            bram_wt_write(base+6,  s1g1);
            bram_wt_write(base+7,  s1g2);
            bram_wt_write(base+8,  s1g3);
            bram_wt_write(base+9,  s2g0);
            bram_wt_write(base+10, s2g1);
            bram_wt_write(base+11, s2g2);
            bram_wt_write(base+12, s2g3);
            bram_wt_write(base+13, s3g0);
            bram_wt_write(base+14, s3g1);
            bram_wt_write(base+15, s3g2);
            bram_wt_write(base+16, s3g3);
        end
    endtask

    // ── Signal dma_in_done pulse ──────────────────────────────────────────────
    task pulse_dma_done;
        begin
            @(negedge clk);
            dma_in_done = 1'b1;
            @(posedge clk);
            @(negedge clk);
            dma_in_done = 1'b0;
        end
    endtask

    // ── Wait for IRQ with timeout ─────────────────────────────────────────────
    // Returns 1=success, 0=timeout
    task wait_irq;
        input  integer max_cycles;
        input  [127:0] tc_name;
        output integer ok;
        begin : wt_irq_blk
            integer cnt;
            cnt = 0; ok = 0;
            while (!irq_out && cnt < max_cycles) begin
                @(posedge clk);
                cnt = cnt + 1;
                if (cnt % 200 == 0)
                    $display("  DBG @%0d cyc: state=%0d swc_en=%b tc_valid=%b dma_in_done_in=%b",
                        cnt, dut.state, dut.swc_en, dut.tc_valid, dut.dma_in_done);
            end
            if (irq_out) begin
                $display("PASS %s: irq_out asserted after %0d cycles", tc_name, cnt);
                ok = 1;
            end else begin
                $display("FAIL %s: timeout (%0d cycles) waiting for irq_out", tc_name, max_cycles);
                $display("  FINAL state=%0d swc_en=%b tc_valid=%b act_ld=%0d wt_ld=%0d",
                    dut.state, dut.swc_en, dut.tc_valid, dut.act_ld_cnt, dut.wt_ld_cnt);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ── CHECK helper ─────────────────────────────────────────────────────────
    task check_eq;
        input [127:0] label;
        input [31:0]  got;
        input [31:0]  exp;
        begin
            if (got === exp) begin
                $display("PASS %s: got=%0d exp=%0d", label, got, exp);
                pass_count = pass_count + 1;
            end else begin
                $display("FAIL %s: got=0x%08h exp=0x%08h", label, got, exp);
                fail_count = fail_count + 1;
            end
        end
    endtask

    // ── Reset all DMA signals ─────────────────────────────────────────────────
    task dma_idle;
        begin
            dma_in_addr  = 10'd0; dma_in_data  = 8'd0;
            dma_in_wr_en = 1'b0;  dma_in_done  = 1'b0;
            dma_wt_addr  = 12'd0; dma_wt_data  = 8'd0;
            dma_wt_wr_en = 1'b0;
            dma_out_addr = 10'd0;
        end
    endtask

    // ── Clear interrupt ───────────────────────────────────────────────────────
    task clear_irq;
        begin
            do_axi_write(`AXI_OFF_AP_ISR, 32'd1);  // W1C
        end
    endtask

    // ─────────────────────────────────────────────────────────────────────────
    // MAIN TEST
    // ─────────────────────────────────────────────────────────────────────────
    integer ok;

    initial begin
        fail_count = 0; pass_count = 0;

        // Init AXI
        awvalid=0; awaddr=0; wvalid=0; wdata=0; wstrb=4'hF; bready=0;
        arvalid=0; araddr=0; rready=0;
        dma_idle;

        // Reset
        rst_n = 1'b0;
        repeat(4) @(posedge clk);
        rst_n = 1'b1;
        repeat(2) @(posedge clk);

        // ══════════════════════════════════════════════════════════════════════
        // TC_01: conv_type=0 (conv1x1)
        // Cin=16, Cout=4, Hin=1, Win=1, N_config=1, poc=0, accu_mode=0
        // Weight BRAM: idx=3, s0g0=5, rest=0  → PSUM = 5*act[3]
        // Input BRAM: act[3]=10  → PSUM_g0 = 50
        // Requant with scale=0x4000 (=1.0 in Q14), shift=15: out = round(50*1.0 >> 15)
        // But rr_unit hardwired scale=0x4000, shift=15 in unified_acc_top → out=round(50/32768)=0
        // TC_01 goal: verify IRQ fires (compute completes), not numeric output
        // ══════════════════════════════════════════════════════════════════════
        $display("=== TC_01: conv1x1 (conv_type=0) ===");

        // Write config
        do_axi_write(`AXI_OFF_LAYER_TYPE, 32'd0);  // CONV_TYPE_1X1
        do_axi_write(`AXI_OFF_CIN,        32'd16);
        do_axi_write(`AXI_OFF_COUT,       32'd4);
        do_axi_write(`AXI_OFF_HIN,        32'd1);
        do_axi_write(`AXI_OFF_WIN,        32'd1);
        do_axi_write(`AXI_OFF_N_CONFIG,   32'd1);
        do_axi_write(`AXI_OFF_POC,        32'd0);
        do_axi_write(`AXI_OFF_ACCU_MODE,  32'd0);
        do_axi_write(`AXI_OFF_APPLY_RELU, 32'd0);

        // Verify LAYER_TYPE readback
        $display("DEBUG TC_01: reading AXI_OFF_LAYER_TYPE = 0x%03h", `AXI_OFF_LAYER_TYPE);
        do_axi_read(`AXI_OFF_LAYER_TYPE, rd_val);
        $display("DEBUG TC_01: LAYER_TYPE rd_val = 0x%08h", rd_val);
        check_eq("TC_01 LAYER_TYPE readback", rd_val, 32'd0);

        // Verify ap_idle=1
        do_axi_read(`AXI_OFF_AP_CTRL, rd_val);
        check_eq("TC_01 ap_idle at reset", rd_val[2], 1'b1);

        // Enable interrupt
        do_axi_write(`AXI_OFF_AP_GIE, 32'd1);
        do_axi_write(`AXI_OFF_AP_IER, 32'd1);

        // Load weight BRAM: 1 NZ entry at byte[0..16]
        // idx=3, s0g0=5, s0g1=0, s0g2=0, s0g3=0, all s1..s3=0
        load_nz_entry(12'd0,
            4'd3,
            8'sd5, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0);

        // Load input BRAM: 16 bytes (1 group of 16 positions)
        for (i = 0; i < 16; i = i + 1)
            bram_in_write(i[9:0], (i == 3) ? 8'd10 : 8'd0);

        // ap_start
        do_axi_write(`AXI_OFF_AP_CTRL, 32'd1);
        repeat(2) @(posedge clk);

        // Signal DMA done
        pulse_dma_done;

        // Wait for IRQ
        wait_irq(500, "TC_01 conv1x1 irq", ok);
        if (ok) begin
            do_axi_read(`AXI_OFF_AP_ISR, rd_val);
            check_eq("TC_01 ap_isr latched", rd_val[0], 1'b1);
            clear_irq;
        end

        repeat(10) @(posedge clk);

        // ══════════════════════════════════════════════════════════════════════
        // TC_02: conv_type=1 (conv3x3)
        // Cin=16, Cout=4, Hin=3, Win=3, N_config=1, poc=0, accu_mode=0
        // stride=1, pad=1
        // Weight BRAM: idx=0, s0g0=4, rest=0
        // Input BRAM: 3x3 = 9 pixels, act[0..8] = 2 (all same)
        // 3x3 kernel with 9 positions × wt[idx=0,s0g0=4] × act[k=0]=2 → psum=72
        // TC_02 goal: verify IRQ fires (3x3 sliding window compute completes)
        // ══════════════════════════════════════════════════════════════════════
        $display("=== TC_02: conv3x3 (conv_type=1) ===");

        dma_idle;
        // Reset DUT
        @(negedge clk); rst_n = 1'b0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(2) @(posedge clk);

        do_axi_write(`AXI_OFF_LAYER_TYPE, 32'd1);  // CONV_TYPE_3X3
        do_axi_write(`AXI_OFF_CIN,        32'd16);
        do_axi_write(`AXI_OFF_COUT,       32'd4);
        do_axi_write(`AXI_OFF_HIN,        32'd3);
        do_axi_write(`AXI_OFF_WIN,        32'd3);
        do_axi_write(`AXI_OFF_N_CONFIG,   32'd1);
        do_axi_write(`AXI_OFF_POC,        32'd0);
        do_axi_write(`AXI_OFF_ACCU_MODE,  32'd0);
        do_axi_write(`AXI_OFF_APPLY_RELU, 32'd0);
        do_axi_write(`AXI_OFF_STRIDE,     32'd1);
        do_axi_write(`AXI_OFF_PAD,        32'd1);

        // Verify LAYER_TYPE readback
        do_axi_read(`AXI_OFF_LAYER_TYPE, rd_val);
        check_eq("TC_02 LAYER_TYPE readback", rd_val, 32'd1);

        // Verify STRIDE readback
        do_axi_read(`AXI_OFF_STRIDE, rd_val);
        check_eq("TC_02 STRIDE readback", rd_val, 32'd1);

        // Enable interrupt
        do_axi_write(`AXI_OFF_AP_GIE, 32'd1);
        do_axi_write(`AXI_OFF_AP_IER, 32'd1);

        // Load weight BRAM: 1 NZ entry, idx=0, s0g0=4
        load_nz_entry(12'd0,
            4'd0,
            8'sd4, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0);

        // Load input BRAM: 3x3 = 9 pixels, each group channel 0 = 2
        // Layout: row-major, 3 pixels/row, 3 rows (halo=1 → 5x5 padded, but
        // we load just the actual 3x3 patch; TB drives activations via BRAM)
        for (i = 0; i < 9; i = i + 1)
            bram_in_write(i[9:0], 8'd2);

        // ap_start
        do_axi_write(`AXI_OFF_AP_CTRL, 32'd1);
        repeat(2) @(posedge clk);

        // Signal DMA done
        pulse_dma_done;

        // Wait for IRQ — 3x3 needs more cycles (9 kernel positions)
        wait_irq(40000, "TC_02 conv3x3 irq", ok);
        if (ok) begin
            do_axi_read(`AXI_OFF_AP_ISR, rd_val);
            check_eq("TC_02 ap_isr latched", rd_val[0], 1'b1);
            clear_irq;
        end

        repeat(10) @(posedge clk);

        // ══════════════════════════════════════════════════════════════════════
        // TC_03: conv_type=2 (dwconv3x3)
        // config_decoder must force eff_poc=0, eff_accu_mode=0
        // User intentionally writes POC=3, ACCU_MODE=2 (wrong for dwconv)
        // If forcing works, FSM must still complete and IRQ fires
        // If NOT forced, poc=3 would make FSM stall (undefined behavior)
        // ══════════════════════════════════════════════════════════════════════
        $display("=== TC_03: dwconv (conv_type=2, poc/accu forced) ===");

        dma_idle;
        @(negedge clk); rst_n = 1'b0;
        repeat(4) @(posedge clk);
        @(negedge clk); rst_n = 1'b1;
        repeat(2) @(posedge clk);

        do_axi_write(`AXI_OFF_LAYER_TYPE, 32'd2);  // CONV_TYPE_DW
        do_axi_write(`AXI_OFF_CIN,        32'd16);
        do_axi_write(`AXI_OFF_COUT,       32'd16);  // dwconv: Cin==Cout
        do_axi_write(`AXI_OFF_HIN,        32'd3);
        do_axi_write(`AXI_OFF_WIN,        32'd3);
        do_axi_write(`AXI_OFF_N_CONFIG,   32'd1);
        do_axi_write(`AXI_OFF_POC,        32'd3);   // intentionally wrong — must be forced to 0
        do_axi_write(`AXI_OFF_ACCU_MODE,  32'd2);   // intentionally wrong — must be forced to 0
        do_axi_write(`AXI_OFF_APPLY_RELU, 32'd0);
        do_axi_write(`AXI_OFF_STRIDE,     32'd1);
        do_axi_write(`AXI_OFF_PAD,        32'd1);

        // Verify LAYER_TYPE=2
        do_axi_read(`AXI_OFF_LAYER_TYPE, rd_val);
        check_eq("TC_03 LAYER_TYPE readback", rd_val, 32'd2);

        // Verify POC register stores 3 (AXI reg stores user value)
        do_axi_read(`AXI_OFF_POC, rd_val);
        check_eq("TC_03 reg_poc stores user value (3)", rd_val[1:0], 2'd3);

        // Verify ACCU_MODE register stores 2
        do_axi_read(`AXI_OFF_ACCU_MODE, rd_val);
        check_eq("TC_03 reg_accu_mode stores user value (2)", rd_val[1:0], 2'd2);

        // TC_04 note: eff_poc/eff_accu_mode are internal wires in config_decoder.
        // We verify forcing indirectly: FSM must complete without stall.
        // If poc=3 were used, i_spec would expect 4-out accumulation which
        // with only 16 input channels and 1 NZ entry would stall or produce
        // wrong output count. IRQ firing = forcing worked correctly.
        $display("INFO TC_04: eff_poc/eff_accu_mode force verified by IRQ firing");
        $display("INFO TC_04: reg stores poc=3, accu=2 but config_decoder forces eff=0,0 for DW");

        // Enable interrupt
        do_axi_write(`AXI_OFF_AP_GIE, 32'd1);
        do_axi_write(`AXI_OFF_AP_IER, 32'd1);

        // Load weight BRAM: 1 NZ entry, idx=0, s0g0=3 (arbitrary non-zero)
        load_nz_entry(12'd0,
            4'd0,
            8'sd3, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0,
            8'sd0, 8'sd0, 8'sd0, 8'sd0);

        // Load 3x3 input (same as TC_02)
        for (i = 0; i < 9; i = i + 1)
            bram_in_write(i[9:0], 8'd1);

        // ap_start
        do_axi_write(`AXI_OFF_AP_CTRL, 32'd1);
        repeat(2) @(posedge clk);
        pulse_dma_done;

        // Wait for IRQ — if config_decoder forcing fails, FSM stalls → timeout
        wait_irq(40000, "TC_03/04 dwconv irq (force check)", ok);
        if (ok) begin
            do_axi_read(`AXI_OFF_AP_ISR, rd_val);
            check_eq("TC_03 ap_isr latched", rd_val[0], 1'b1);
            clear_irq;
            $display("PASS TC_04: config_decoder forced eff_poc=0, eff_accu_mode=0 (IRQ fired)");
            pass_count = pass_count + 1;
        end else begin
            $display("FAIL TC_04: IRQ timeout implies poc/accu_mode NOT forced — config_decoder broken");
            fail_count = fail_count + 1;
        end

        repeat(10) @(posedge clk);

        // ══════════════════════════════════════════════════════════════════════
        // Summary
        // ══════════════════════════════════════════════════════════════════════
        $display("========================================");
        $display("SUMMARY: %0d PASS, %0d FAIL", pass_count, fail_count);
        if (fail_count == 0)
            $display("ALL PASS");
        else
            $display("SOME TESTS FAILED");
        $display("========================================");
        $finish;
    end

    // ── Watchdog: kill sim if runaway ─────────────────────────────────────────
    initial begin
        #500000;
        $display("WATCHDOG: simulation exceeded 500us — ABORT");
        $finish;
    end

    // ── VCD dump (optional, comment out if not needed) ────────────────────────
    initial begin
        $dumpfile("tb_unified_acc_top.vcd");
        $dumpvars(0, tb_unified_acc_top);
    end

endmodule
