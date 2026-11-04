from pathlib import Path
import shutil,subprocess,json,time,os,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src/ps/model"))
from activity_golden import Golden,ROOT,np
R=ROOT
W=R/'build/activity_sim';W.mkdir(exist_ok=True,parents=True)
shutil.copytree(R/'data/golden_ref/rtl_vectors/accelerator',W/'accelerator',dirs_exist_ok=True)
model=Golden();x=np.zeros((64,192),np.int64)
# 第一段平稳、随后小扰动、再大扰动，验证高低精度切换与回退，不用人体数据。
x[24:40,::3]=2;x[40:48]=np.arange(192)%31-15
def mem(name,values,bits):
    (W/name).write_text('\n'.join(f'{int(v)&((1<<bits)-1):0{bits//4}x}' for v in values.flatten())+'\n')
mem('input.mem',x,8);expected=[]
for mode in [0,1,2]:
    y,d=model.forward(x,0,mode,True);mem(f'logits{mode}.mem',y,16);expected.append(int(d['low_elements']))
    np.savez(ROOT/'sim/results'/f'stress_mode{mode}.npz',input=x,logits_down=y,**d)
tb='''`timescale 1ns/1ps
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
      if((m==0 && low!=LOW0)||(m==1 && low!=LOW1)||(m==2 && low!=LOW2))$fatal(1,"low count %0d",low);
      $display("MODE_PASS mode=%0d low=%0d total=%0d",mode,low,total);
    end
    if(errors)$fatal(1,"errors=%0d",errors);
    $display("PASS: activity full core all modes 1248 logits and policy counters");$finish;
  end
  initial begin #500000000;$fatal(1,"timeout");end
endmodule
'''
for mode,val in enumerate(expected):tb=tb.replace(f'LOW{mode}',str(val))
(W/'tb_activity.sv').write_text(tb)
sources=[R/'src/pl/rtl'/n for n in ['h144_fixed_pkg.sv','h144_layout_pkg.sv','h144_fixed_math_pkg.sv','h144_activation_lut.sv','h144_dense_mac8.sv','h144_gru_element.sv','h144_decoder_core.sv']]+[W/'tb_activity.sv']
start=time.time()
for tool,args in [('xvlog',['--sv',*map(str,sources)]),('xelab',['tb_activity','-s','activity_sim']),('xsim',['activity_sim','-runall'])]:
    run=subprocess.run([str(Path(os.environ.get('VIVADO_BIN','D:/2025.2/Vivado/bin'))/(tool+'.bat' if os.name=='nt' else tool)),*args],cwd=W,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=900)
    output=run.stdout.decode('utf-8','replace');(ROOT/'sim/results'/f'activity_{tool}.log').write_text(output,encoding='utf-8')
    print(tool,run.returncode,output[-1400:],flush=True)
    if run.returncode or 'FATAL:' in output or 'ERROR:' in output:raise RuntimeError(tool)
    if tool=='xsim':assert 'PASS: activity' in output
(ROOT/'sim/results/rtl_simulation.json').write_text(json.dumps(dict(passed=True,synthetic_frames=64,
    modes=[0,1,2],compared_logits=1248,expected_low_elements=expected,seconds=time.time()-start),indent=2))
