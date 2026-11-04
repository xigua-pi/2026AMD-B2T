# B2Tnet 最终交付说明

发布标识：`b2tnet-activity-fpga-20261104`。

交付目录遵循 `src/`、`sim/`、`build/`、`board/`、`data/`、`skill/`、`report/` 的结构；`configs/` 保存冻结数值配置，`third_party/` 保存上游许可。

## 模型与硬件

- B2Tnet 活动感知门精度模型，278,416 个参数。
- PYNQ-Z2 / XC7Z020-CLG400-1；50 MHz PL 时钟。
- W8A8 / H16 / ACC32，Sigmoid / Tanh LUT512。
- 官方 HeldOutBlocks 228 句：CER 9.6988%，WER 37.6260%。
- 3,457,568 个 INT16 板端输出与对应整数参考逐位一致。

模型、RTL、冻结权重、比特流与板测结果保持一致。文档中的作者身份元数据已清除，原始对话、认证材料及个人主目录不属于交付内容；`report/llm_trace/` 仅保留协作说明。

## 复现与校验

在仓库根目录运行：

```bash
python -m pip install -r requirements.txt
python skill/verify_scripts/verify_repository.py
```

该检查同时支持 Git 克隆和不含 `.git` 的源码 ZIP。原始神经活动输入从官方数据源另行准备，步骤见 README。

`file_manifest_sha256.json` 列出交付文件的内容哈希。交付压缩包只包含登记文件及清单，不含 Git 配置、私有备份、工作目录缓存或本地审计原始材料。
