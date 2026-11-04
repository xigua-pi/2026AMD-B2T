"""将活动感知门精度接入附件QAT模型；未声称本次已在GPU完成QAT。

调用前须把附件recurrent_fixed_point_model_mp.py与原项目模块加入PYTHONPATH。
forward中的决策与RTL相同：输入最大变化<=0.25，上一更新hidden最大变化<=0.125。
保留每步H16反馈，不跳过矩阵运算；低精度R8/Z12/N16，其余R16/Z16/N16。
"""
import torch
from recurrent_fixed_point_model_mp import RecurrentFixedPointH144 as Base,round_ste

class ActivityAwareH144(Base):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.set_gate_precision(16,16,16)
        self.activity_mode='dynamic'

    def _gru_sequence(self,gru,inputs,lengths,start_state,input_quantizer,hidden_quantizer,preact_quantizer,return_trace):
        # 复用附件的严格递归计算，通过门量化器包装保留其原梯度和数值路径。
        # forward hook仅替换量化后门值，下一时刻反馈仍来自替换后的hidden。
        history={'step':0,'previous_input':None,'previous_hidden':None,'last_delta':None,'low':None}
        masks=[];layer_previous=None
        def low_mask():
            nonlocal layer_previous
            t=history['step'];batch=inputs.shape[0]
            if t==0:
                low=torch.zeros((batch,1),device=inputs.device,dtype=torch.bool)
            else:
                delta=(inputs[:,t]-inputs[:,t-1]).detach().abs().amax(-1,keepdim=True)
                # hidden_quantizer的最后一次调用是上一时刻最终反馈。
                low=(delta<=.25)&(history['last_delta']<=.125)
            if self.activity_mode=='static':low=torch.ones_like(low)
            if self.activity_mode=='high':low=torch.zeros_like(low)
            history['low']=low;masks.append(low)
            return low
        def reset_hook(module,args,value):
            low=low_mask();q=round_ste((value*256).clamp(0,255))/256
            return torch.where(low,q,value)
        def update_hook(module,args,value):
            q=round_ste((value*4096).clamp(0,4095))/4096
            return torch.where(history['low'],q,value)
        # hidden量化器被调用：初始化1次；每步两个乘积分支+总和反馈共3次。
        state={'calls':0,'last':None}
        def hidden_hook(module,args,value):
            if state['calls']==0:state['last']=value.detach()
            elif state['calls']%3==0:
                history['last_delta']=(value.detach()-state['last']).abs().amax(-1,keepdim=True)
                state['last']=value.detach();history['step']+=1
            state['calls']+=1
            return value
        hooks=[self.reset_q.register_forward_hook(reset_hook),self.update_q.register_forward_hook(update_hook),hidden_quantizer.register_forward_hook(hidden_hook)]
        try:output,trace=super()._gru_sequence(gru,inputs,lengths,start_state,input_quantizer,hidden_quantizer,preact_quantizer,return_trace)
        finally:
            for hook in hooks:hook.remove()
        if return_trace:trace['activity_low_precision']=torch.stack(masks,dim=1)
        return output,trace

    def fixed_point_specification(self):
        spec=super().fixed_point_specification()
        spec['activity_precision']=dict(mode=self.activity_mode,input_delta_threshold=.25,previous_hidden_delta_threshold=.125,
            low_gate_bits=[8,12,16],high_gate_bits=[16,16,16],matrix_mac_skipping=False)
        return spec
