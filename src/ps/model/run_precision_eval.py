"""冻结单一动态策略后做官方228句一次性对照；不扫描阈值、不挑配置。"""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
from pathlib import Path
import json,time,sys,hashlib
from activity_golden import Golden,ROOT,REPO,np
sys.path.insert(0,str(REPO/'src/ps/model'))
# 不导入依赖torch的原函数；下面复写同一20ms onset crossing规则并核对历史计数。
LETTERS="abcdefghijklmnopqrstuvwxyz>,'~?"
def render(s):
    s=s.replace('>',' ').replace('~','.')
    for c in ",?' .":
        if c!=' ':s=s.replace(' '+c,c)
    return ' '.join(s.split())
def decode(logits,length):
    x=np.repeat(logits,5,axis=0)[:length]
    # logits为Q7.8；等价于sigmoid>0.3，用浮点只做最终离线文本判决。
    onset=1/(1+np.exp(-np.clip(x[:,-1].astype(float)/256,-40,40)))
    starts=np.flatnonzero((onset[:-1]<.3)&(onset[1:]>.3))
    ix=np.minimum(starts+15,length-1)
    return ''.join(LETTERS[i] for i in x[ix,:31].argmax(1))
def edit(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        prev=row;row=[i]+[0]*len(b)
        for j,y in enumerate(b,1):row[j]=min(prev[j]+1,row[j-1]+1,prev[j-1]+(x!=y))
    return row[-1]
def metrics(rows,mode):
    ce=ct=we=wt=0
    for r in rows:
        a=render(r['reference']);b=render(r['modes'][str(mode)]['hypothesis'])
        ce+=edit(a,b);ct+=max(1,len(a));we+=edit(a.split(),b.split());wt+=max(1,len(a.split()))
    return dict(char_errors=ce,char_total=ct,word_errors=we,word_total=wt,cer=ce/ct,wer=we/wt)
def main():
    model=Golden();out=ROOT/'board/results/software_official';out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'data/eeg_dataset/official_traces/manifest.json').read_text())
    config=dict(strategy='max current layer-input difference and previous hidden difference',
                input_threshold_gru1=8,input_threshold_gru2=16,hidden_threshold=2048,
                low_bits=[8,12,16],high_bits=[16,16,16],first_step_high=True,
                hidden_feedback_bits=16,decode_threshold=.3,decode_delay=15,
                threshold_selection='fixed before evaluating official references; no sweep',qat_performed=False)
    (ROOT/'board/results/frozen_activity_policy.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    rows=[]
    for item in manifest['records']:
        file=out/f"record_{item['record_index']:03d}.json"
        if file.exists():rows.append(json.loads(file.read_text()));continue
        t=time.time();raw=np.load(ROOT/'data/eeg_dataset/official_traces'/item['trace_file']);x=raw['input']
        row=dict(record_index=item['record_index'],session_id=item['session_id'],reference=item['reference'],modes={})
        for mode in [0,1,2]:
            y,d=model.forward(x,item['session_id'],mode,False)
            np.savez(out/f"record_{item['record_index']:03d}_mode{mode}.npz",input=x,logits_down=y)
            row['modes'][str(mode)]=dict(hypothesis=decode(y,len(x)),low_elements=int(d['low_elements']),
                total_elements=int(d['total_elements']),gate_register_bit_flips=int(d['gate_register_bit_flips']),
                mismatches_vs_frozen=int(np.count_nonzero(y!=raw['logits_down'])))
            if mode==0:assert np.array_equal(y,raw['logits_down']),item['record_index']
        row['seconds']=time.time()-t;file.write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8');rows.append(row)
        print('RECORD',item['record_index'],'seconds',round(row['seconds'],2),'low',row['modes']['2']['low_elements']/row['modes']['2']['total_elements'],flush=True)
    summary=dict(records=len(rows),metrics={str(m):metrics(rows,m) for m in [0,1,2]},totals={})
    for mode in ['0','1','2']:
        summary['totals'][mode]={k:sum(r['modes'][mode][k] for r in rows) for k in ['low_elements','total_elements','gate_register_bit_flips','mismatches_vs_frozen']}
    assert summary['metrics']['0']['char_errors']==1081 and summary['metrics']['0']['word_errors']==817,summary['metrics']['0']
    (ROOT/'board/results/software_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
