---
name: pynq-neural-decoder
description: Reproduce and demonstrate a PYNQ AXI-DMA fixed-point neural decoder, including cache-safe output streaming and bit-exact verification, using existing exported feature traces.
---
# PYNQ神经解码器复现

先读仓库README、冻结数值规范与板端版本号，区分静态模型和活动感知变体。当前发布bitstream固定mode=2，0x40只读，不能宣称支持写寄存器切换模式。

DMA先启动S2MM再启动MM2S；输入flush后发送，输出运行中只invalidate，不能flush覆盖PL已写数据。只消费实际DDR中完整的32-logit行。停止/异常时先终止硬件访问再释放CMA缓冲。

每句重置GRU与历史状态，句内保留递归；6帧尾部flush用于13点非因果滤波。核对数据哈希、session_id、输入字节、输出字数、错误标志和策略计数，逐位比较对应策略golden。成功信息不可代替输出检查。

UI文本只能来自本次PL返回值，不能从参考文本、排序结果或缓存预测直接发射。按准确率排序只是演示选择；保留完整228句总体评测，不能将前若干句指标替代整体指标。

功耗代理、Vivado vectorless估计与实测功率分开；本活动策略MAC/权重不变，新增控制增加LUT/FF，不能声称跳算、面积或整机功耗下降。

执行验证入口：`python skill/verify_scripts/verify_repository.py`。板端演示入口见`board/scripts/README.md`。所有地址与文件位置由参数传入，不修改无关网卡、VPN或远程任务。
