`timescale 1ns/1ps
module tb_activity;
  logic clk=0;always #5 clk=~clk;
  logic rst_n=0,sr=0;logic [1:0] mode=0;
  logic valid=0,ready,last=0;logic signed [7:0] data;
  logic fv=0,fr,mv,ml,seq,fd,rfd,le,pe,ov;logic signed [15:0] md;logic [5:0] mi;
  logic [31:0] cycles,low,total;
  reg [7:0] inputs[0:12287];reg [15:0] gold0[0:415],gold1[0:415],gold2[0:415];
  integer m,t,j,outidx=0,errors=0;reg [15:0] want;
  h144_decoder_core dut(.clk(clk),.rst_n(rst_n),.sequence_reset(sr),.session_id(4'd0),
    .precision_mode(mode),.low_precision_elements(low),.total_precision_elements(total),
    .s_axis_tvalid(valid),.s_axis_tready(ready),.s_axis_tdata(data),.s_axis_tlast(last),
    .flush_valid(fv),.flush_ready(fr),.m_axis_tvalid(mv),.m_axis_tready(1'b1),
    .m_axis_tdata(md),.m_axis_tindex(mi),.m_axis_tlast(ml),.sequence_ready(seq),
    .frame_done(fd),.result_frame_done(rfd),.logits_emitted(le),.protocol_error(pe),
    .accumulator_overflow(ov),.last_frame_cycles(cycles));
  always @(posedge clk) if(mv) begin
    want=(mode==0)?gold0[outidx]:((mode==1)?gold1[outidx]:gold2[outidx]);
    if(md!==want)begin if(errors<8)$display("MISMATCH mode=%0d idx=%0d want=%0d got=%0d",mode,outidx,$signed(want),md);errors=errors+1;end
    outidx=outidx+1;
  end
  initial begin
    $readmemh("input.mem",inputs);$readmemh("logits0.mem",gold0);$readmemh("logits1.mem",gold1);$readmemh("logits2.mem",gold2);
    for(m=0;m<3;m=m+1)begin
      @(negedge clk);rst_n=0;mode=m;repeat(4)@(negedge clk);rst_n=1;outidx=0;
      wait(seq);
      for(t=0;t<64;t=t+1)begin
        wait(ready);
        for(j=0;j<192;j=j+1)begin @(negedge clk);valid=1;data=inputs[t*192+j];last=(j==191);end
        @(negedge clk);valid=0;last=0;wait(fd);@(negedge clk);
        if(pe||ov)$fatal(1,"protocol or overflow");
      end
      for(t=0;t<6;t=t+1)begin wait(fr);@(negedge clk);fv=1;@(negedge clk);fv=0;wait(fd);@(negedge clk);end
      if(outidx!=416)$fatal(1,"output count %0d",outidx);
      if(total!=11088)$fatal(1,"element count %0d",total);
      if((m==0 && low!=0)||(m==1 && low!=11088)||(m==2 && low!=6048))$fatal(1,"low count %0d",low);
      $display("MODE_PASS mode=%0d low=%0d total=%0d",mode,low,total);
    end
    if(errors)$fatal(1,"errors=%0d",errors);
    $display("PASS: activity full core all modes 1248 logits and policy counters");$finish;
  end
  initial begin #500000000;$fatal(1,"timeout");end
endmodule
