# 模型入口

主阅读顺序：`compact_bci_model.py` → `recurrent_fixed_point_model.py` → `bit_exact_h144_golden.py`。

`train_compact_bci.py`、`train_recurrent_fixed_qat.py`保存训练和QAT方法；`evaluate_h144_official_heldout.py`执行官方划分评估。其余文件是本地复现/数据处理的导入依赖或历史辅助模块，保留是为了代码可运行，**不表示科研仓库同时推进几何、CTC、MWER等多条创新主线**。

默认路径和启动超参数不是冻结实验命令，运行时显式传入数据、标签、checkpoint及输出路径。输入是预处理后的神经活动特征，不能将历史注释中的raw/EEG目录命名理解为原始头皮电压。

