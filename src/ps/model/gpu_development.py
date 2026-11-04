"""工作站只评估99开发句，先验证活动QAT接口high模式与原模型相同。"""
from pathlib import Path
import sys,json,time,os
import numpy as np
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[3]
PROJECT=Path(os.environ.get('B2T_TRAIN_PROJECT',str(ROOT)));sys.path.insert(0,str(PROJECT))
from recurrent_fixed_point_model import RecurrentFixedPointH144 as Original
from activity_qat_model import ActivityAwareH144
from train_compact_bci import load_frozen_label_cache
from spatiotemporal_discriminability_atlas import split_train_internal
from paper_v5_pipeline import PaperV5EvalDataset,collate_variable_paper_batch,decode_paper_v5_outputs
from evaluate_h144_official_heldout import error_counts,split_fingerprint

def main():
    torch.set_num_threads(4);torch.cuda.set_device(1);torch.backends.cuda.matmul.allow_tf32=False
    out=ROOT/'board/results/gpu_development';out.mkdir(exist_ok=True,parents=True)
    ckpt=ROOT/'data/golden_ref/checkpoints/fpga_candidate.pt'
    checkpoint=torch.load(ckpt,map_location='cpu',weights_only=False)
    old=Original(**checkpoint['model_config']);old.load_state_dict(checkpoint['model_state'])
    model=ActivityAwareH144(**checkpoint['model_config']);model.load_state_dict(checkpoint['model_state'])
    for m in [old,model]:
        m.activation_mode='lut';m.lut_entries=512;m.lut_range=8.;m.quantization_enabled=True
        m.set_observer_mode(False,False);m.to('cuda:1').eval()
    model.set_gate_precision(16,16,16)
    data=load_frozen_label_cache(str(PROJECT/'paper_v5_label_cache.pt'),os.environ['B2T_DATASETS_ROOT'])
    fit,dev,blocks=split_train_internal([r for r in data['records'] if r.split=='train'])
    assert len(fit)==403 and len(dev)==99
    loader=DataLoader(PaperV5EvalDataset(str(data['datasets_root']),dev,bin_size=2,training_steps_root=None),batch_size=4,
        shuffle=False,num_workers=0,collate_fn=collate_variable_paper_batch)
    results={};first=True
    with torch.no_grad():
        for mode in ['high','static','dynamic']:
            model.activity_mode=mode;rows=[];started=time.time()
            for batch_idx,b in enumerate(loader):
                x=b['inputs'].to('cuda:1');lens=b['input_lengths'].to('cuda:1');sessions=b['session_ids'].to('cuda:1')
                y,_=model(x,lens,sessions)
                if first:
                    ref,_=old(x,lens,sessions);err=float((ref-y).abs().max());assert err==0,err
                    (out/'high_equivalence.json').write_text(json.dumps(dict(max_abs_error=err,shape=list(y.shape),passed=True)))
                    first=False
                y=y.float().cpu()
                for k,length in enumerate(b['input_lengths'].tolist()):
                    hyp=decode_paper_v5_outputs(y[k,:length],length,start_threshold=.3,start_delay=15)
                    rows.append(dict(session=b['session_names'][k],sentence_index=int(b['sentence_indices'][k]),
                                     reference=b['transcripts'][k],hypothesis=hyp))
                print('DEV',mode,batch_idx,len(rows),'seconds',round(time.time()-started,1),flush=True)
            results[mode]=dict(records=len(rows),metrics=error_counts([r['reference'] for r in rows],[r['hypothesis'] for r in rows]),
                              rows=rows,seconds=time.time()-started)
            (out/f'development_{mode}.json').write_text(json.dumps(results[mode],ensure_ascii=False,indent=2))
    summary=dict(complete=True,device='cuda:1',checkpoint=str(ckpt),split_fingerprint=split_fingerprint(dev),
        scope='PyTorch recurrent fake-quant graph on development only; different from pure-integer board benchmark',
        training_performed=False,results={k:{kk:vv for kk,vv in v.items() if kk!='rows'} for k,v in results.items()})
    (out/'gpu_development_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
