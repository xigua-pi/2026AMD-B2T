# 第三方来源与授权

## Willett数据与方法

Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M., Shenoy, K. V. High-performance brain-to-text communication via handwriting. Nature 593, 249–254 (2021).

- 论文：https://doi.org/10.1038/s41586-021-03506-2
- 数据：https://doi.org/10.5061/dryad.wh70rxwmv
- 原复现代码：https://github.com/fwillett/handwritingBCI
- 原代码许可证：https://github.com/fwillett/handwritingBCI/blob/main/LICENSE.txt

原代码不是MIT/BSD通用开源许可。其条款涉及内部非商业研究使用、衍生作品、保留权利及附带许可证等要求。随附`third_party/stanford_handwritingBCI_LICENSE.txt`保留原文；公开发布或商业使用前需作者/课题组核实适用范围。本说明不对每个本地文件是否构成衍生作品作法律认定，也不重新授权上游材料。

当前代码包含本地PyTorch复现、紧凑网络、定点和RTL实现及配套工具，不应将其描述为斯坦福官方权重。本仓库不分发原始Dryad神经数据或人体派生输入特征trace。按用户提供的最终交付包保留定点输出golden及逐句预测，作为逐位验证证据；它们不是原始采集波形。权重和历史聚合指标保留来源、格式和哈希。

## AMD/Xilinx与PYNQ

Vivado、PS7、AXI DMA、互连等属于相应权利人。`build/scripts/pynq_3_0_1_base_reference.tcl`保留其已有版权头。冻结bitstream/HWH用于对应板卡的研究复现，不是商业IP独立源码或工具安装包。



