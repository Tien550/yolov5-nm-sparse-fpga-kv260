// lag_buf.v -- Rev169 IN-PLACE LAG SCATTER: buffer trễ 32 slot × 64 word × 16B (32KB, 8 RAMB36).
// Kick pos p (hàng r) đọc input hàng r-1..r+1 → output pos p KHÔNG được đè plane ngay.
// Fill: oc_deint (fill) ghi out_bram của pos p vào slot p[4:0] (base={p[4:0],6'b0}, stride=1).
// Drain: oc_deint (drain) đọc slot dpos[4:0], ghi act_uram tại base=dpos, stride=npos —
// dpos = p - LAG_D (LAG_D = 2*ncolg = 2 hàng) → chỉ đè hàng đã tiêu thụ xong.
// RD_LAT = 1 (addr reg → data reg) — khớp latency out_bram_wide rd128 mà oc_deint kỳ vọng.
`timescale 1ns/1ps

module lag_buf (
    input  wire         clk,
    input  wire [10:0]  wr_addr,
    input  wire [127:0] wr_data,
    input  wire         wr_en,
    input  wire [10:0]  rd_addr,
    output reg  [127:0] rd_data
);
    (* ram_style = "block" *) reg [127:0] mem [0:2047];
    always @(posedge clk) begin
        if (wr_en) mem[wr_addr] <= wr_data;
        rd_data <= mem[rd_addr];
    end
endmodule
