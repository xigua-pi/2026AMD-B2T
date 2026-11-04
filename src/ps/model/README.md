# 模型入口

主阅读顺序：`compact_bci_model.py` → `recurrent_fixed_point_model.py` → `bit_exact_h144_golden.py`。

`train_compact_bci.py`、`train_recurrent_fixed_qat.py`实现训练和 QAT；`evaluate_h144_official_heldout.py`执行官方划分评估。辅助模块提供数据读取、预处理和实验分析支持。

运行时传入数据、标签、checkpoint 及输出路径。输入为完成预处理的 192 通道植入式神经活动特征；冻结配置见仓库 `configs/` 目录。

