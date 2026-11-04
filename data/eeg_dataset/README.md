# 输入数据

输入为植入式阈值越界计数衍生的 192 通道神经活动特征，目录名称遵循竞赛结构。

从 [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv) 获取数据，按导出脚本完成标准化、平滑、10 ms → 20 ms 合帧与量化。将导出的 228 句 `official_traces/`（manifest.json、official_000.npz…official_227.npz）置于本目录，用于整数评估及板端解码。
