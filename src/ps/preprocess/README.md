# 数据准备

`export_h144_official_board_traces.py`位于`../model/`。已有输入已完成标准化、平滑、20ms合帧和INT8量化，板端不再次执行原始波形阈值提取。此处的export_vivado_mem.py生成ROM初始化，不是采集前端。
