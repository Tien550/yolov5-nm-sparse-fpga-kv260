// conv3x3_layer_seq.v -- On-chip layer loop for the conv3x3 engine (LAYER_TYPE=5).
// Replaces the Python (band × Cout-chunk) nested host kick loop with a single ap_start.
// Host pre-stages ALL padded-act bands and ALL weight+param chunks in DDR before kicking.
// FSM: for each band b: DMA act[b]→amem; for each Cout-chunk c: DMA wt[c]→wmem, kick engine,
//      DMA omem→out[b*n_cout+c]; then next c, next b.
//
// Register mapping (matches existing AXI offsets so no new regs needed):
//   n_bands  ← reg_n_pos[7:0]     (REG_N_POS  0x1B0)
//   n_cout   ← reg_n_ocb[7:0]     (REG_N_OCB  0x1A0)
//   act_band_bytes ← reg_act_chunk (REG_ACT_CHUNK 0x1E0) — bytes per band; also band stride (tight pack)
//   wt_chunk_bytes ← reg_wt_bytes  (REG_WT_BYTES  0x160) — bytes per wt+param chunk; also stride
//   out_chunk_bytes← reg_out_bytes (REG_OUT_BYTES 0x150) — bytes per (band,cout) output chunk; also stride
// DDR layout:
//   act:  act_base + b * act_bytes                        (tight, ≥4KB each → safe)
//   wt:   wt_base  + c * wt_bytes                        (tight, reused across bands)
//   out:  out_base + (b*n_cout + c) * out_bytes          (tight)
// Owned dma_engine while is_c3 & LAYER_MODE=1. Mirrors spec_layer_seq's `issued` handshake.
`timescale 1ns/1ps

module conv3x3_layer_seq (
    input  wire        aclk,
    input  wire        aresetn,
    input  wire        layer_mode,      // = reg_layer_mode (1 = this sequencer drives c3 + dma_engine)
    input  wire        ap_start,        // gated in cnn_acc_top: ap_start & is_c3
    input  wire        c3_done,         // conv3x3 engine compute done (c3_done_w)

    // Config latched at ap_start
    input  wire [7:0]  n_bands,         // row-band count  (reg_n_pos[7:0])
    input  wire [7:0]  n_cout,          // Cout-chunk count (reg_n_ocb[7:0])
    input  wire [23:0] act_bytes,       // padded-act DMA bytes per band = Cin*WBin*16 (reg_act_chunk)
    input  wire [23:0] wt_bytes,        // wt+param DMA bytes per Cout-chunk (reg_wt_bytes)
    input  wire [23:0] out_bytes,       // output bytes per (band,Cout-chunk) (reg_out_bytes)
    input  wire [63:0] act_base,        // DDR base for act bands (reg_in_addr)
    input  wire [63:0] wt_base,         // DDR base for wt chunks (reg_wt_addr)
    input  wire [63:0] out_base,        // DDR base for output (reg_out_addr)

    // dma_engine handshake
    output reg         dma_req,
    output reg  [1:0]  dma_req_tgt,    // 0=ACT 1=WT 2=OUT
    output reg  [63:0] dma_req_addr,
    output reg  [23:0] dma_req_bytes,
    input  wire        dma_active,
    input  wire        dma_done,

    output reg         c3seq_start,    // kick c3 engine compute
    output reg         busy,
    output reg         c3seq_done_latch,
    output wire [3:0]  dbg_state
);
    localparam TGT_ACT=2'd0, TGT_WT=2'd1, TGT_OUT=2'd2;
    localparam S_IDLE=4'd0, S_LDA=4'd1, S_LDW=4'd2, S_KICK=4'd3,
               S_RUN=4'd4, S_OUT=4'd5, S_NEXT=4'd6, S_DONE=4'd7;

    reg [3:0] st;
    reg       ap_seen, issued;
    reg [7:0] band_i, cout_i;
    reg [7:0] l_nbands, l_ncout;
    reg [23:0] l_actb, l_wtb, l_outb;
    reg [63:0] l_actbase, l_wtbase, l_outbase;
    reg [23:0] out_rem, out_off;

    assign dbg_state = st;

    // Compute DDR addresses for current (band_i, cout_i)
    wire [63:0] cur_act_addr = l_actbase + ({56'd0, band_i} * {40'd0, l_actb});
    wire [63:0] cur_wt_addr  = l_wtbase  + ({56'd0, cout_i} * {40'd0, l_wtb});
    // out index = band_i * n_cout + cout_i (16-bit max: 255*255+255=65535)
    wire [15:0] out_idx = ({8'd0, band_i} * {8'd0, l_ncout}) + {8'd0, cout_i};
    wire [63:0] cur_out_addr = l_outbase + ({48'd0, out_idx} * {40'd0, l_outb});

    always @(posedge aclk) begin
        if (!aresetn) begin
            st <= S_IDLE; dma_req <= 0; c3seq_start <= 0; busy <= 0;
            issued <= 0; ap_seen <= 0; c3seq_done_latch <= 0;
            band_i <= 0; cout_i <= 0; out_rem <= 0; out_off <= 0;
        end else begin
            dma_req <= 1'b0; c3seq_start <= 1'b0;
            if (ap_start && layer_mode) c3seq_done_latch <= 1'b0;
            case (st)
                S_IDLE: begin
                    busy <= 1'b0; issued <= 1'b0;
                    if (ap_start && layer_mode && !ap_seen) begin
                        ap_seen <= 1'b1; busy <= 1'b1; band_i <= 8'd0; cout_i <= 8'd0;
                        l_nbands <= n_bands; l_ncout <= n_cout;
                        l_actb <= act_bytes; l_wtb <= wt_bytes; l_outb <= out_bytes;
                        l_actbase <= act_base; l_wtbase <= wt_base; l_outbase <= out_base;
                        st <= S_LDA;
                    end
                    if (!ap_start) ap_seen <= 1'b0;
                end

                // Load padded-act band → amem (once per band)
                S_LDA: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req <= 1'b1; dma_req_tgt <= TGT_ACT;
                            dma_req_addr <= cur_act_addr; dma_req_bytes <= l_actb;
                            issued <= 1'b1;
                        end
                    end else if (dma_done) begin issued <= 1'b0; st <= S_LDW; end
                end

                // Load wt+param chunk → wmem (once per Cout-chunk)
                S_LDW: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req <= 1'b1; dma_req_tgt <= TGT_WT;
                            dma_req_addr <= cur_wt_addr; dma_req_bytes <= l_wtb;
                            issued <= 1'b1;
                        end
                    end else if (dma_done) begin issued <= 1'b0; st <= S_KICK; end
                end

                // Start conv3x3 engine compute
                S_KICK: begin
                    c3seq_start <= 1'b1; out_rem <= l_outb; out_off <= 24'd0; st <= S_RUN;
                end

                // Wait for compute done
                S_RUN: if (c3_done) st <= S_OUT;

                // DMA output chunk → DDR (chunked ≤1024B to fit dma_out_addr 10-bit counter)
                S_OUT: begin
                    if (!issued) begin
                        if (!dma_active && !dma_req) begin
                            dma_req <= 1'b1; dma_req_tgt <= TGT_OUT;
                            dma_req_addr <= cur_out_addr + {40'd0, out_off};
                            dma_req_bytes <= (out_rem > 24'd1024) ? 24'd1024 : out_rem;
                            issued <= 1'b1;
                        end
                    end else if (dma_done) begin
                        issued <= 1'b0;
                        if (out_rem > 24'd1024) begin
                            out_rem <= out_rem - 24'd1024; out_off <= out_off + 24'd1024;
                        end else st <= S_NEXT;
                    end
                end

                // Advance cout_i / band_i
                S_NEXT: begin
                    if (cout_i + 8'd1 < l_ncout) begin
                        cout_i <= cout_i + 8'd1; st <= S_LDW;   // next Cout-chunk (same band)
                    end else begin
                        cout_i <= 8'd0;
                        if (band_i + 8'd1 < l_nbands) begin
                            band_i <= band_i + 8'd1; st <= S_LDA;  // next band
                        end else st <= S_DONE;
                    end
                end

                S_DONE: begin
                    busy <= 1'b0; c3seq_done_latch <= 1'b1; ap_seen <= 1'b0; st <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
