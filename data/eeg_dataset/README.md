# 输入数据

输入是植入式阈值越界计数衍生的192通道神经活动特征，不是原始头皮EEG。目录名称按竞赛要求保留。

原数据：https://doi.org/10.5061/dryad.wh70rxwmv 。原始人体神经数据不随仓库发布。将已有导出的228句`official_traces/`（manifest.json、official_000.npz…official_227.npz）置于本目录。先按原项目导出脚本从合法取得的数据生成，不能用随机输入代替正式准确率。数据需标准化、平滑和10ms→20ms合帧，再按冻结scale量化。
