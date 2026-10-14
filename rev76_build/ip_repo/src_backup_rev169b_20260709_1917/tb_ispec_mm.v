// tb_ispec_mm.v -- Layer-scale verification of i_spec_mm (paper-faithful n_cfg=N COO)
// Conv1x1: Cin=32 (2 M-tiles), Cout=4 (1 SPEC=4 groups), P=16 spatial pos, N=3 NZ/M-tile.
// Golden: out[g][p] = Σ_{mtile} Σ_{e<N} W[g][mtile][e] * A[ idx[mtile][e] ][p]
// (structured: idx shared across all g for a given M-tile)

`timescale 1ns/1ps
`include "cnn_defines.vh"

module tb_ispec_mm;
    localparam CIN     = 512;      // L26 worst-case (32 M-tiles)
    localparam M_TILES = CIN/`M;   // 32
    localparam COUT    = 4;
    localparam N       = 3;

    reg clk=0, rst_n=0;
    always #5 clk = ~clk;

    reg [3:0] n_cfg;
    reg       acc_clr;
    reg [1:0] shuf_mode;
    reg [`M*`P*`DATA_W-1:0] act_load_bus;
    reg       wt_load;
    reg [3:0] idx_0,idx_1,idx_2,idx_3,idx_4,idx_5,idx_6,idx_7,idx_8,idx_9,idx_10,idx_11,idx_12,idx_13,idx_14,idx_15;
    reg signed [`DATA_W-1:0] wv_g0_0,wv_g0_1,wv_g0_2,wv_g0_3;
    reg signed [`DATA_W-1:0] wv_g1_0,wv_g1_1,wv_g1_2,wv_g1_3;
    reg signed [`DATA_W-1:0] wv_g2_0,wv_g2_1,wv_g2_2,wv_g2_3;
    reg signed [`DATA_W-1:0] wv_g3_0,wv_g3_1,wv_g3_2,wv_g3_3;

    wire signed [`ACC_W-1:0]
        ps_g0[0:15], ps_g1[0:15], ps_g2[0:15], ps_g3[0:15];
    wire mac_valid;

    i_spec_mm dut (
        .clk(clk),.rst_n(rst_n),.n_cfg(n_cfg),.acc_clr(acc_clr),
        .shuf_mode(shuf_mode),.act_load_bus(act_load_bus),.wt_load(wt_load),
        .idx_0(idx_0),.idx_1(idx_1),.idx_2(idx_2),.idx_3(idx_3),.idx_4(idx_4),.idx_5(idx_5),.idx_6(idx_6),.idx_7(idx_7),
        .idx_8(idx_8),.idx_9(idx_9),.idx_10(idx_10),.idx_11(idx_11),.idx_12(idx_12),.idx_13(idx_13),.idx_14(idx_14),.idx_15(idx_15),
        .wv_g0_0(wv_g0_0),.wv_g0_1(wv_g0_1),.wv_g0_2(wv_g0_2),.wv_g0_3(wv_g0_3),
        .wv_g1_0(wv_g1_0),.wv_g1_1(wv_g1_1),.wv_g1_2(wv_g1_2),.wv_g1_3(wv_g1_3),
        .wv_g2_0(wv_g2_0),.wv_g2_1(wv_g2_1),.wv_g2_2(wv_g2_2),.wv_g2_3(wv_g2_3),
        .wv_g3_0(wv_g3_0),.wv_g3_1(wv_g3_1),.wv_g3_2(wv_g3_2),.wv_g3_3(wv_g3_3),
        .psum_g0_p0(ps_g0[0]),.psum_g0_p1(ps_g0[1]),.psum_g0_p2(ps_g0[2]),.psum_g0_p3(ps_g0[3]),.psum_g0_p4(ps_g0[4]),.psum_g0_p5(ps_g0[5]),.psum_g0_p6(ps_g0[6]),.psum_g0_p7(ps_g0[7]),
        .psum_g0_p8(ps_g0[8]),.psum_g0_p9(ps_g0[9]),.psum_g0_p10(ps_g0[10]),.psum_g0_p11(ps_g0[11]),.psum_g0_p12(ps_g0[12]),.psum_g0_p13(ps_g0[13]),.psum_g0_p14(ps_g0[14]),.psum_g0_p15(ps_g0[15]),
        .psum_g1_p0(ps_g1[0]),.psum_g1_p1(ps_g1[1]),.psum_g1_p2(ps_g1[2]),.psum_g1_p3(ps_g1[3]),.psum_g1_p4(ps_g1[4]),.psum_g1_p5(ps_g1[5]),.psum_g1_p6(ps_g1[6]),.psum_g1_p7(ps_g1[7]),
        .psum_g1_p8(ps_g1[8]),.psum_g1_p9(ps_g1[9]),.psum_g1_p10(ps_g1[10]),.psum_g1_p11(ps_g1[11]),.psum_g1_p12(ps_g1[12]),.psum_g1_p13(ps_g1[13]),.psum_g1_p14(ps_g1[14]),.psum_g1_p15(ps_g1[15]),
        .psum_g2_p0(ps_g2[0]),.psum_g2_p1(ps_g2[1]),.psum_g2_p2(ps_g2[2]),.psum_g2_p3(ps_g2[3]),.psum_g2_p4(ps_g2[4]),.psum_g2_p5(ps_g2[5]),.psum_g2_p6(ps_g2[6]),.psum_g2_p7(ps_g2[7]),
        .psum_g2_p8(ps_g2[8]),.psum_g2_p9(ps_g2[9]),.psum_g2_p10(ps_g2[10]),.psum_g2_p11(ps_g2[11]),.psum_g2_p12(ps_g2[12]),.psum_g2_p13(ps_g2[13]),.psum_g2_p14(ps_g2[14]),.psum_g2_p15(ps_g2[15]),
        .psum_g3_p0(ps_g3[0]),.psum_g3_p1(ps_g3[1]),.psum_g3_p2(ps_g3[2]),.psum_g3_p3(ps_g3[3]),.psum_g3_p4(ps_g3[4]),.psum_g3_p5(ps_g3[5]),.psum_g3_p6(ps_g3[6]),.psum_g3_p7(ps_g3[7]),
        .psum_g3_p8(ps_g3[8]),.psum_g3_p9(ps_g3[9]),.psum_g3_p10(ps_g3[10]),.psum_g3_p11(ps_g3[11]),.psum_g3_p12(ps_g3[12]),.psum_g3_p13(ps_g3[13]),.psum_g3_p14(ps_g3[14]),.psum_g3_p15(ps_g3[15]),
        .mac_valid(mac_valid)
    );

    // ── Test data ────────────────────────────────────────────────────────────
    integer A   [0:CIN-1][0:`P-1];          // activations [ic][p]
    integer IDX [0:M_TILES-1][0:N-1];       // NZ index within M-tile (0..15), shared all g
    integer W   [0:COUT-1][0:M_TILES-1][0:N-1]; // weight per (oc, mtile, entry)
    integer golden [0:COUT-1][0:`P-1];

    integer mt, e, p, g, ic, gic;
    integer fail, cyc0;

    // load act_load_bus for a given M-tile: act_mem[ic_local][p]=A[mt*M+ic_local][p]
    task load_act_tile(input integer mtile);
        integer iic, pp;
        begin
            for (iic=0; iic<`M; iic=iic+1)
                for (pp=0; pp<`P; pp=pp+1)
                    act_load_bus[(iic*`P+pp)*`DATA_W +: `DATA_W] = A[mtile*`M+iic][pp];
        end
    endtask

    task drive_idx(input integer mtile);
        begin
            idx_0=IDX[mtile][0]; idx_1=IDX[mtile][1]; idx_2=IDX[mtile][2]; idx_3=0;
            idx_4=0;idx_5=0;idx_6=0;idx_7=0;idx_8=0;idx_9=0;idx_10=0;idx_11=0;idx_12=0;idx_13=0;idx_14=0;idx_15=0;
        end
    endtask

    task drive_wt(input integer mtile);
        begin
            wv_g0_0=W[0][mtile][0]; wv_g0_1=W[0][mtile][1]; wv_g0_2=W[0][mtile][2]; wv_g0_3=0;
            wv_g1_0=W[1][mtile][0]; wv_g1_1=W[1][mtile][1]; wv_g1_2=W[1][mtile][2]; wv_g1_3=0;
            wv_g2_0=W[2][mtile][0]; wv_g2_1=W[2][mtile][1]; wv_g2_2=W[2][mtile][2]; wv_g2_3=0;
            wv_g3_0=W[3][mtile][0]; wv_g3_1=W[3][mtile][1]; wv_g3_2=W[3][mtile][2]; wv_g3_3=0;
        end
    endtask

    initial begin
        // randomize data
        for (ic=0; ic<CIN; ic=ic+1)
            for (p=0; p<`P; p=p+1)
                A[ic][p] = ($random % 15);          // -14..14
        for (mt=0; mt<M_TILES; mt=mt+1) begin
            // pick N distinct sorted indices, varied per M-tile (deterministic)
            IDX[mt][0] = (mt*1) % 5;             // 0..4
            IDX[mt][1] = 5 + ((mt*2) % 5);       // 5..9
            IDX[mt][2] = 10 + ((mt*3) % 6);      // 10..15
            for (g=0; g<COUT; g=g+1)
                for (e=0; e<N; e=e+1)
                    W[g][mt][e] = (($random % 11) - 5);  // -5..5
        end
        // golden
        for (g=0; g<COUT; g=g+1)
            for (p=0; p<`P; p=p+1) begin
                golden[g][p] = 0;
                for (mt=0; mt<M_TILES; mt=mt+1)
                    for (e=0; e<N; e=e+1) begin
                        gic = mt*`M + IDX[mt][e];
                        golden[g][p] = golden[g][p] + W[g][mt][e]*A[gic][p];
                    end
            end

        // ── reset ──
        n_cfg=N; acc_clr=0; shuf_mode=`SHUF_HOLD_MODE; wt_load=0;
        drive_idx(0); drive_wt(0); load_act_tile(0);
        repeat(4) @(posedge clk); rst_n=1; @(posedge clk);

        // clear psum
        @(negedge clk); acc_clr=1; @(posedge clk); @(negedge clk); acc_clr=0;

        cyc0 = $time;
        // ── per M-tile: LOAD act → wt_load → drain ──
        for (mt=0; mt<M_TILES; mt=mt+1) begin
            // load activations for this M-tile
            load_act_tile(mt);
            @(negedge clk); shuf_mode=`SHUF_LOAD;
            @(posedge clk); @(negedge clk); shuf_mode=`SHUF_HOLD_MODE;
            // weight load pulse (drains N entries)
            drive_idx(mt); drive_wt(mt);
            @(negedge clk); wt_load=1;
            @(posedge clk); @(negedge clk); wt_load=0;
            // wait drain (N entries) + mac 2-stage + margin
            repeat(N+2) @(posedge clk);
        end
        repeat(4) @(posedge clk);

        // ── check ──
        fail = 0;
        for (p=0; p<`P; p=p+1) begin
            if (ps_g0[p] !== golden[0][p]) begin fail=fail+1; $display("FAIL g0 p%0d: got=%0d exp=%0d",p,ps_g0[p],golden[0][p]); end
            if (ps_g1[p] !== golden[1][p]) begin fail=fail+1; $display("FAIL g1 p%0d: got=%0d exp=%0d",p,ps_g1[p],golden[1][p]); end
            if (ps_g2[p] !== golden[2][p]) begin fail=fail+1; $display("FAIL g2 p%0d: got=%0d exp=%0d",p,ps_g2[p],golden[2][p]); end
            if (ps_g3[p] !== golden[3][p]) begin fail=fail+1; $display("FAIL g3 p%0d: got=%0d exp=%0d",p,ps_g3[p],golden[3][p]); end
        end

        $display("---- sample: g0 p0 got=%0d exp=%0d | g3 p15 got=%0d exp=%0d",
                 ps_g0[0], golden[0][0], ps_g3[15], golden[3][15]);
        if (fail==0) $display("[PASS] tb_ispec_mm: all %0d outputs match (Cin=%0d Cout=%0d N=%0d)", COUT*`P, CIN, COUT, N);
        else         $display("[FAIL] tb_ispec_mm: %0d mismatches", fail);
        $finish;
    end
endmodule
