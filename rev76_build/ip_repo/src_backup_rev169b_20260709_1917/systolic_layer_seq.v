// systolic_layer_seq.v -- B1: on-chip layer loop for the SYSTOLIC engine (conv1x1_systolic_top).
// Replaces the Python per-kick host loop (conv1x1_systolic_tiled): on ONE ap_start (LAYER_MODE=1)
// it sweeps an ENTIRE conv1x1 layer by looping the HW-tiles (hwt) and Cout-tiles (cot) the host
// used to issue as separate kicks. For each tile it DMAs ACT/WT/PAR from DDR into the engine
// memories, kicks the systolic engine, waits done, and DMAs the OUT tile back -- all on-chip,
// so the host pays the per-kick overhead (put/wr/get ~7ms) ONCE per layer instead of per tile.
//
// Loop order: hwt OUTER, cot INNER. ACT (the fmap HW-chunk) is loaded ONCE per hwt and REUSED
// across all Cout-tiles (the engine only READS amem during compute, so it persists across kicks);
// only WT/PAR change per cot. This removes the redundant ACT re-DMA the host loop did.
//
// DDR tile-stream layout (host bakes 1x/layer; uniform/padded tiles so config is constant):
//   ACT : act_base + hwt * act_tile_bytes               (banked HW-chunk, stride 4KB-safe)
//   WT  : wt_base  + cot * wt_tile_bytes                (wmem blob for the Cout-chunk)
//   PAR : par_base + cot * par_tile_bytes               (pmem requant for the Cout-chunk)
//   OUT : out_base + (cot*n_hwt + hwt) * out_tile_bytes (cot-major; host de-interleaves 1x/layer)
//
// Owns dma_engine while LAYER_MODE=1 (muxed against systolic_seq in cnn_acc_top, like dw_layer).
// Mirrors systolic_seq's per-DMA `issued` handshake (advance only on OWN dma_done).
`timescale 1ns/1ps

module systolic_layer_seq (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        layer_mode,     // 1 = this sequencer drives engine + dma_engine
    input  wire        ap_start,       // SW ap_start (shared); acted on only when layer_mode=1
    input  wire        sys_done,       // systolic engine done (per kick)

    // config (latched at ap_start) -- all tiles uniform (host pads HW/Cout to tile multiples)
    input  wire [15:0] n_hwt,          // # HW-tiles
    input  wire [7:0]  n_cot,          // # Cout-tiles
    input  wire [23:0] act_tile_bytes, // ACT bytes per HW-tile  (4KB-safe stride)
    input  wire [23:0] wt_tile_bytes,  // WT  bytes per Cout-tile (4KB-safe stride)
    input  wire [23:0] par_tile_bytes, // PAR bytes per Cout-tile (0 => no param load)
    input  wire [23:0] out_tile_bytes, // OUT bytes per (cot,hwt) tile
    input  wire [63:0] act_base,
    input  wire [63:0] wt_base,
    input  wire [63:0] par_base,
    input  wire [63:0] out_base,

    // dma_engine handshake (muxed owner in cnn_acc_top)
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,    // 0=ACT 1=WT 2=OUT 3=PAR
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    // systolic engine kick
    output reg         sys_start,

    output reg         busy,           // 1 while looping (host ap_idle = ~busy)
    output reg         irq,
    output reg         ls_done_latch,  // set at S_DONE, cleared on ap_start (-> host ap_done)
    output wire        ls_busy,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2, TGT_PAR=2'd3;
    localparam S_IDLE=4'd0, S_LDA=4'd1, S_LDW=4'd2, S_LDP=4'd3, S_KICK=4'd4,
               S_RUN=4'd5, S_OUT=4'd6, S_NEXT=4'd7, S_DONE=4'd8;
    reg [3:0] st;
    reg       ap_seen, issued;
    reg [15:0] hwt;  reg [7:0] cot;                 // loop counters
    reg [15:0] l_nhwt; reg [7:0] l_ncot;            // latched config
    reg [23:0] l_actc, l_wtc, l_parc, l_outc;
    reg [63:0] l_actb, l_wtb, l_parb, l_outb;
    reg [23:0] out_rem, out_off;                    // OUT chunking (dma_out_addr 10-bit => <=1024B)

    assign ls_busy   = (st != S_IDLE);
    assign dbg_state = st;

    // tile addresses (combinational; multiplies map to DSP). ACT idx=hwt, WT/PAR idx=cot,
    // OUT idx = cot*n_hwt + hwt (cot-major).
    wire [31:0] out_idx  = ({16'd0, cot} * {16'd0, l_nhwt}) + {16'd0, hwt};
    wire [63:0] act_addr = l_actb + ({40'd0, hwt} * {40'd0, l_actc});
    wire [63:0] wt_addr  = l_wtb  + ({40'd0, cot} * {40'd0, l_wtc});
    wire [63:0] par_addr = l_parb + ({40'd0, cot} * {40'd0, l_parc});
    wire [63:0] out_addr = l_outb + ({32'd0, out_idx} * {40'd0, l_outc});

    always @(posedge aclk) begin
        if (!aresetn) begin
            st<=S_IDLE; dma_req<=0; dma_req_tgt<=0; dma_req_addr<=0; dma_req_bytes<=0;
            sys_start<=0; busy<=0; irq<=0; ls_done_latch<=0; ap_seen<=0; issued<=0;
            hwt<=0; cot<=0; out_rem<=0; out_off<=0;
        end else begin
            dma_req<=1'b0; sys_start<=1'b0; irq<=1'b0;   // 1-cyc pulses default low
            if (ap_start && layer_mode) ls_done_latch<=1'b0;   // clear on new ap_start
            case (st)
                S_IDLE: begin
                    busy<=1'b0; issued<=1'b0;
                    if (ap_start && layer_mode && !ap_seen) begin
                        ap_seen<=1'b1; busy<=1'b1;
                        hwt<=16'd0; cot<=8'd0;
                        l_nhwt<=n_hwt; l_ncot<=n_cot;
                        l_actc<=act_tile_bytes; l_wtc<=wt_tile_bytes;
                        l_parc<=par_tile_bytes; l_outc<=out_tile_bytes;
                        l_actb<=act_base; l_wtb<=wt_base; l_parb<=par_base; l_outb<=out_base;
                        st<=S_LDA;
                    end
                    if (!ap_start) ap_seen<=1'b0;
                end
                // ── load ACT for current hwt (once per HW-tile; reused across Cout-tiles) ──
                S_LDA: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_ACT;
                            dma_req_addr<=act_addr; dma_req_bytes<=l_actc; issued<=1'b1;
                        end
                    end else if (dma_done) begin issued<=1'b0; st<=S_LDW; end
                end
                // ── load WT for current cot ──
                S_LDW: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_WT;
                            dma_req_addr<=wt_addr; dma_req_bytes<=l_wtc; issued<=1'b1;
                        end
                    end else if (dma_done) begin issued<=1'b0; st<=S_LDP; end
                end
                // ── load PAR (requant) for current cot ──
                S_LDP: begin
                    if (l_parc==24'd0) st<=S_KICK;
                    else if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req<=1'b1; dma_req_tgt<=TGT_PAR;
                            dma_req_addr<=par_addr; dma_req_bytes<=l_parc; issued<=1'b1;
                        end
                    end else if (dma_done) begin issued<=1'b0; st<=S_KICK; end
                end
                S_KICK: begin sys_start<=1'b1; out_rem<=l_outc; out_off<=24'd0; st<=S_RUN; end
                S_RUN:  if (sys_done) st<=S_OUT;
                // ── store OUT for (cot,hwt), chunked <=1024B (dma_out_addr is 10-bit) ──
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
                // ── advance: cot inner (reuse ACT, load next WT), then hwt outer ──
                S_NEXT: begin
                    if (cot + 8'd1 < l_ncot) begin cot<=cot+8'd1; st<=S_LDW; end
                    else begin
                        cot<=8'd0;
                        if (hwt + 16'd1 < l_nhwt) begin hwt<=hwt+16'd1; st<=S_LDA; end
                        else st<=S_DONE;
                    end
                end
                S_DONE: begin irq<=1'b1; busy<=1'b0; ls_done_latch<=1'b1; ap_seen<=1'b0; st<=S_IDLE; end
                default: st<=S_IDLE;
            endcase
        end
    end
endmodule
