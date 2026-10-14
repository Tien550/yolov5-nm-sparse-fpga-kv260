// c1_copy.v -- On-chip copy: c1 omem → c1 amem (LAYER_TYPE=13, is_c1c).
// Sequential read from c1 omem (via sys_o_rd_data) and write to c1 amem.
// Duration: ~n_words = Cout × WB clock cycles (µs range, much faster than DDR DMA).
// After this: LT7 (is_c1na) makes the next conv1x1 skip ACT DMA and read from amem directly.
`timescale 1ns/1ps

module c1_copy (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,          // 1-cycle pulse: ap_start & is_c1c
    input  wire [19:0] n_words,        // Cout × WB (pre-registered in cnn_acc_top)

    // c1 omem read (sys_o_rd_addr → sys_o_rd_data, 1-cycle read latency)
    output reg  [13:0] rd_addr,
    input  wire [127:0] rd_data,

    // c1 amem write (c1_ld_a port)
    output reg         wr_en,
    output reg  [13:0] wr_addr,
    output wire [127:0] wr_data,       // = rd_data delayed 1 cycle for latency

    output reg         busy,
    output reg         done_latch,
    output wire [3:0]  dbg_state
);
    localparam S_IDLE=2'd0, S_COPY=2'd1, S_DRAIN=2'd2;
    reg [1:0]  st;
    reg [19:0] cnt;         // current read address counter (write = cnt-1, latency-adjusted)
    reg [127:0] rd_latch;   // 1-cycle read latency register

    assign dbg_state = {2'b0, st};
    assign wr_data   = rd_latch;     // write the previously-latched read word

    always @(posedge clk) rd_latch <= rd_data;   // pipeline register for omem read latency

    always @(posedge clk) begin
        if (!rst_n) begin
            st <= S_IDLE; busy <= 1'b0; done_latch <= 1'b0;
            wr_en <= 1'b0; rd_addr <= 14'd0; wr_addr <= 14'd0; cnt <= 20'd0;
        end else begin
            wr_en <= 1'b0;
            // done_latch is a LATCH (stays high until next start), NOT a 1-cycle pulse.
            // This matches spec_layer_seq/sys_done_latch_r pattern so Python polling sees AP_DONE.
            case (st)
                S_IDLE: begin
                    if (start && n_words != 20'd0) begin
                        busy <= 1'b1; cnt <= 20'd0; rd_addr <= 14'd0;
                        done_latch <= 1'b0;   // clear latch on new kick
                        st <= S_COPY;
                    end
                end
                S_COPY: begin
                    // Issue read at cnt; write cnt-1 (latched data from previous cycle)
                    rd_addr <= cnt[13:0];
                    if (cnt >= 20'd1) begin
                        wr_en   <= 1'b1;
                        wr_addr <= cnt[13:0] - 14'd1;   // write is one cycle behind read
                    end
                    cnt <= cnt + 20'd1;
                    if (cnt + 20'd1 >= n_words) st <= S_DRAIN;  // last read in flight → drain
                end
                S_DRAIN: begin
                    // Last read (addr = n_words-1) is now in rd_latch. Write it.
                    wr_en       <= 1'b1;
                    wr_addr     <= n_words[13:0] - 14'd1;
                    done_latch  <= 1'b1;   // SET latch — stays high until next S_IDLE clear
                    busy        <= 1'b0;
                    st          <= S_IDLE;
                end
                default: st <= S_IDLE;
            endcase
        end
    end
endmodule
