// oc_deint.v — scatter 1 kick: out_bram[oc] → đích[base + oc*stride]. Rev167: tổng quát hoá
// (base,stride) cho 2 mode: CHW (base=pos, stride=n_pos — deint cũ) và M-BLOCK resident-act
// (base=pos*cpad do LS tính TĂNG DẦN, stride=1 → addr = pos*cpad + ch, đúng chunk [16ch][16sp]).
// ob_rd_addr registered (+1) + out_bram RD_LAT (+1) → data trễ 2 cyc → write pipe 2-deep.
module oc_deint #(parameter AW = 15) (
    input  wire          clk, rst_n, start,
    input  wire [7:0]    ock,
    input  wire [AW-1:0] base, stride,
    output reg  [7:0]    ob_rd_addr,
    input  wire [127:0]  ob_rd_data,
    output reg  [AW-1:0] db_wr_addr,
    output reg  [127:0]  db_wr_data,
    output reg           db_wr_en,
    output reg           done
);
    reg           run;
    reg  [8:0]    cnt;
    reg  [AW-1:0] acc;
    reg  [AW-1:0] wa1, wa2;   // write-addr pipe 2-deep
    reg           wv1, wv2;
    always @(posedge clk) begin
        if (!rst_n) begin
            run<=0; cnt<=0; acc<=0; wv1<=0; wv2<=0; db_wr_en<=0; done<=0; ob_rd_addr<=0;
        end else begin
            db_wr_en<=0; done<=0;
            if (start) begin
                run<=1; cnt<=0; acc<=base; ob_rd_addr<=0; wv1<=0; wv2<=0;
            end else if (run) begin
                // stage-1: issue read oc=cnt
                if (cnt < ock) begin
                    ob_rd_addr <= cnt[7:0];
                    wa1 <= acc; wv1 <= 1'b1;
                    acc <= acc + stride; cnt <= cnt + 9'd1;
                end else begin
                    wv1 <= 1'b0;
                end
                // pipe
                wa2 <= wa1; wv2 <= wv1;
                // stage-3: data cho addr đã về (trễ 2) → ghi
                if (wv2) begin
                    db_wr_addr <= wa2; db_wr_data <= ob_rd_data; db_wr_en <= 1'b1;
                end
                if (cnt >= ock && !wv1 && !wv2) begin done<=1'b1; run<=1'b0; end
            end
        end
    end
endmodule
