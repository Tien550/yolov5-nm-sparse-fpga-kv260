`timescale 1ns/1ps
module tb;
  localparam AW=13, OCK=48, NPOS=169;   // 52² worst-fit (169*48=8112<=8192)
  reg clk=0, rst_n=0, start=0;
  always #5 clk=~clk;
  reg [7:0] ock = OCK; reg [AW-1:0] pos=0, n_pos=NPOS;
  wire [7:0] ob_rd_addr; wire [AW-1:0] db_wr_addr; wire [127:0] db_wr_data; wire db_wr_en, done;
  // model out_bram cho 1 kick (OCK word), nội dung = f(pos,oc)
  reg [127:0] obram [0:255]; reg [127:0] ob_rd_data;
  reg [AW-1:0] cur_pos;
  always @(posedge clk) ob_rd_data <= obram[ob_rd_addr];   // RD_LAT=1
  // deint_bram
  reg [127:0] dbram [0:8191];
  always @(posedge clk) if (db_wr_en) dbram[db_wr_addr] <= db_wr_data;

  oc_deint #(.AW(AW)) u (.clk(clk),.rst_n(rst_n),.start(start),.ock(ock),
     .pos(cur_pos),.n_pos(n_pos),.ob_rd_addr(ob_rd_addr),.ob_rd_data(ob_rd_data),
     .db_wr_addr(db_wr_addr),.db_wr_data(db_wr_data),.db_wr_en(db_wr_en),.done(done));

  integer p, oc, errs, a; reg [127:0] exp, got;
  function [127:0] gen; input [15:0] pp; input [15:0] oo;
     gen = {8'hAB, pp[7:0], oo[7:0], 8'h11, 96'h0} ^ {pp,oo,96'hDEAD}; endfunction

  initial begin
    rst_n=0; #20; rst_n=1; #10;
    for (p=0;p<NPOS;p=p+1) begin
      // nạp out_bram cho kick pos=p: word oc = gen(p,oc)
      for (oc=0;oc<OCK;oc=oc+1) obram[oc] = gen(p,oc);
      cur_pos = p[AW-1:0];
      @(posedge clk); start<=1; @(posedge clk); start<=0;
      wait(done); @(posedge clk);
    end
    // verify: deint_bram[oc*NPOS + p] == gen(p,oc)  (CHW: oc outer, pos inner)
    errs=0;
    for (oc=0;oc<OCK;oc=oc+1) for (p=0;p<NPOS;p=p+1) begin
      a = oc*NPOS + p; exp = gen(p,oc); got = dbram[a];
      if (got !== exp) begin errs=errs+1;
        if (errs<=5) $display("  MISMATCH oc=%0d p=%0d addr=%0d exp=%h got=%h",oc,p,a,exp,got); end
    end
    $display("DEINT_SIM: OCK=%0d NPOS=%0d total=%0d errors=%0d", OCK, NPOS, OCK*NPOS, errs);
    if (errs==0) $display("RESULT_PASS"); else $display("RESULT_FAIL");
    $finish;
  end
  initial begin #2000000; $display("TIMEOUT"); $finish; end
endmodule
