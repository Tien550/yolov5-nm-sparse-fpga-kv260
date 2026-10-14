// deint_bram.v — HW-offload Nấc B (deint-on-HW): staging BRAM lật (pos, oc_local) → CHW.
//
// Mục đích: bỏ ARM-deint (numpy transpose). Engine ghi output tile-order [pos][oc_local][16];
// module này giữ 1 ob-block, rồi CHO ĐỌC channel-order [oc_local][pos][16] → DMA DDR = CHW.
// TRANSPOSE nằm ở ĐỊA CHỈ (write addr ≠ read iteration), không logic phức tạp.
//
// Kích thước: 128-bit × DEPTH word. DEPTH=8192 → 128KB → cover per-ob ≤ 52² (N_POS≤169, OCK=48:
// 169*48=8112 ≤ 8192). Layer 104² (N_POS=676) KHÔNG fit → host giữ ARM-deint (gate ngoài).
// ram_style=block (dùng BRAM TRỐNG 100%, KHÔNG đụng URAM 87.5% đầy).
//
// Port: write (từ out_bram copy, mỗi kick 48 word tại pos*OCK) + read (DMA đọc, iterate oc_local
// ngoài pos trong). Dual-port BRAM (write-phase accumulate, read-phase DMA — có thể trùng ob kế
// nếu ping-pong sau; hiện single ob, write xong mới read).
module deint_bram #(
    parameter WORD_W = 128,
    parameter DEPTH  = 32768,         // 512KB — per-ob ≤ 104² (676*48=32448<=32768)
    parameter AW     = 15             // ceil(log2(32768))
) (
    input  wire              clk,
    input  wire [AW-1:0]     wr_addr,     // = pos*OCK + oc_local (tile order)
    input  wire [WORD_W-1:0] wr_data,
    input  wire              wr_en,
    input  wire [AW-1:0]     rd_addr,     // = pos*OCK + oc_local, iterate (oc_local outer, pos inner)
    output reg  [WORD_W-1:0] rd_data      // RD_LAT=1
);
    (* ram_style = "block" *) reg [WORD_W-1:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule
