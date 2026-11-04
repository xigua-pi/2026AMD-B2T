"""从冻结整数NPZ执行，无PyTorch依赖；动态策略只影响GRU门输出。

决策仅依赖当前输入差和上一轮已经完成的hidden差，首步强制高精度。
固定阈值在读取官方结果前登记，不用参考文本、onset标签或WER调参。
"""
from pathlib import Path
import os,sys,json
os.environ.setdefault('NUMBA_NUM_THREADS','4')
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'vendor'))
import numpy as np
from numba import njit
REPO=ROOT

@njit(cache=True)
def rshift(x,shift):
    if shift<=0:return x<<(-shift)
    a=abs(x);q=a>>shift;r=a- (q<<shift);half=1<<(shift-1)
    q+=int(r>half or (r==half and q%2==1))
    return -q if x<0 else q

@njit(cache=True)
def sat(x,maximum):return min(max(x,-maximum),maximum)

@njit(cache=True)
def gate_reduce(value,bits,unsigned=True):
    shift=16-bits
    q=rshift(value,shift)
    hi=(1<<bits)-1 if unsigned else (1<<(bits-1))-1
    q=min(max(q,0 if unsigned else -hi),hi)
    return q<<shift

@njit(cache=True)
def lut_index(x):
    return min(511,rshift((min(16384,max(-16384,x))+16384)*511,15))

@njit(cache=True)
def linear(x,w,b,exponents,input_exp,target_exp,maximum):
    y=np.empty((len(x),len(w)),np.int64)
    for t in range(len(x)):
        for o in range(len(w)):
            a=b[o]
            for j in range(x.shape[1]):a+=x[t,j]*w[o,j]
            y[t,o]=sat(rshift(sat(a,2147483647),target_exp-input_exp-exponents[o]),maximum)
    return y

@njit(cache=True)
def requant_matrix(x,shifts,maximum):
    out=np.empty_like(x)
    for t in range(len(x)):
        for j in range(x.shape[1]):out[t,j]=sat(rshift(x[t,j],shifts[j]),maximum)
    return out

@njit(cache=True)
def gru(x,wi,wh,bi,bh,ei,eh,start,input_exp,sigmoid,tanh,mode,trace):
    """mode0全16；mode1静态R8/Z12/N16；mode2活动感知R8/Z12/N16。

    GRU1 输入Q-5阈值8，GRU2 Q-6阈值16，对应相同物理幅度0.25。
    hidden变化阈值2048(Q-14=0.125)。不跳过状态更新或矩阵计算。
    """
    gates=linear(x,wi,bi,ei,input_exp,-11,32767)
    h=start.copy();y=np.empty((len(x),144),np.int64)
    details=np.zeros((len(x),144,10),np.int64) if trace else np.empty((0,0,0),np.int64)
    previous_input=np.zeros(x.shape[1],np.int64)
    previous_delta=65535;low_count=0;bit_flips=0;previous_gate=np.zeros((3,144),np.int64)
    input_threshold=8 if input_exp==-5 else 16
    for t in range(len(x)):
        input_delta=0
        for j in range(x.shape[1]):input_delta=max(input_delta,abs(x[t,j]-previous_input[j]))
        low=mode==1 or (mode==2 and t>0 and input_delta<=input_threshold and previous_delta<=2048)
        if low:low_count+=144
        hh=linear(h.reshape(1,144),wh,bh,eh,-14,-11,32767)[0]
        max_delta=0
        for j in range(144):
            r=sigmoid[lut_index(sat(gates[t,j]+hh[j],32767))]
            z=sigmoid[lut_index(sat(gates[t,j+144]+hh[j+144],32767))]
            if low:r=gate_reduce(r,8,True);z=gate_reduce(z,12,True)
            rh=sat(rshift(r*hh[j+288],16),32767)
            n=tanh[lut_index(sat(gates[t,j+288]+rh,32767))]
            new=sat(sat(rshift((65536-z)*n,17),32767)+sat(rshift(z*h[j],16),32767),32767)
            max_delta=max(max_delta,abs(new-h[j]))
            for g,v in enumerate((r,z,n)):
                xor=(v^previous_gate[g,j])&65535
                for bit in range(16):bit_flips+=(xor>>bit)&1
                previous_gate[g,j]=v
            if trace:
                details[t,j,0]=gates[t,j];details[t,j,1]=gates[t,j+144];details[t,j,2]=gates[t,j+288]
                details[t,j,3]=hh[j];details[t,j,4]=hh[j+144];details[t,j,5]=hh[j+288]
                details[t,j,6]=h[j];details[t,j,7]=int(low);details[t,j,8]=new;details[t,j,9]=r
            y[t,j]=new
        h=y[t].copy();previous_input=x[t].copy();previous_delta=max_delta
    return y,details,low_count,bit_flips

class Golden:
    def __init__(self):
        a=np.load(REPO/'data/golden_ref/rtl_integer_parameters.npz',allow_pickle=False)
        self.a={k:a[k].astype(np.int64) for k in a.files}
    def dense(self,x,key,bias_key,ie,oe,bits,session=None):
        w=self.a[key+'.integer'];e=self.a[key+'.scale_exponent']
        if session is not None:w=w[session];e=e[session]
        w=w.reshape(w.shape[0],-1)
        b=np.zeros(len(w),np.int64) if bias_key is None else self.a[bias_key]
        return linear(x,w,b,e,ie,oe,(1<<(bits-1))-1)
    def forward(self,x,session,mode=0,trace=False):
        x=x.astype(np.int64);a=self.a
        mid=self.dense(x,'session_adapter.v',None,-4,-3,8,session)
        lr=self.dense(mid,'session_adapter.u',None,-3,-4,8,session)
        adapted=np.clip(x+lr+a['session_adapter.bias.bias_i8'][session],-127,127)
        padded=np.pad(adapted,((6,6),(0,0)))
        accum=np.zeros_like(x)
        dw=a['temporal_bottleneck.depthwise.weight.integer'][:,0,:]
        for k in range(13):accum+=padded[k:k+len(x)]*dw[:,k]
        e=a['temporal_bottleneck.depthwise.weight.scale_exponent']
        depth=requant_matrix(accum,-1-e,127)
        bottleneck=self.dense(depth,'temporal_bottleneck.pointwise.weight','temporal_bottleneck.pointwise.bias_acc_i32',-5,-5,8)
        def run_g(inp,num,exp):
            key='gru'+str(num)
            return gru(inp,a[key+'.weight_ih_l0.integer'],a[key+'.weight_hh_l0.integer'],
                a[key+'.ih.bias_acc_i32'],a[key+'.hh.bias_acc_i32'],a[key+'.weight_ih_l0.scale_exponent'],
                a[key+'.weight_hh_l0.scale_exponent'],a['start_state_'+str(num)+'.start_i16'].reshape(-1),
                exp,a['sigmoid_lut_u16'],a['tanh_lut_i16'],mode,trace)
        h1,t1,l1,f1=run_g(bottleneck,1,-5)
        # Q-14 -> Q-6，round-even后对称INT8饱和。
        down=requant_matrix(h1[::5],np.full(144,8,np.int64),127)
        h2,t2,l2,f2=run_g(down,2,-6)
        logits=self.dense(h2,'readout.weight','readout.bias_acc_i32',-14,-8,16)
        return logits.astype(np.int16),dict(low_elements=l1+l2,total_elements=(len(h1)+len(h2))*144,
            low_gru1=l1,low_gru2=l2,gate_register_bit_flips=f1+f2,
            gru1=t1,gru2=t2,hidden1=h1,hidden2=h2,bottleneck=bottleneck)

if __name__=='__main__':
    model=Golden();fixture=REPO/'data/golden_ref/synthetic_fixture'
    m=json.loads((fixture/'manifest.json').read_text())
    def read(name):
        info=m['tensors'][name];v=np.array([int(s,16) for s in (fixture/info['file']).read_text().split()],np.int64)
        if info['signed']:v=np.where(v>=(1<<(info['bits']-1)),v-(1<<info['bits']),v)
        return v.reshape(info['shape'])
    x=read('input');ref=read('logits_down')
    y,d=model.forward(x,0,0,True)
    assert np.array_equal(y,ref),(np.max(np.abs(y.astype(int)-ref)),y.shape,ref.shape)
    for mode in [0,1,2]:
        y,d=model.forward(x,0,mode,True)
        np.savez(ROOT/'results'/f'synthetic_mode{mode}.npz',input=x,logits_down=y,**d)
    print('PASS numpy integer baseline agrees with frozen synthetic golden')
