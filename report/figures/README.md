# 配图说明

b2tnet_architecture_final.png 为作者提供的最终网络结构与量化精度图，README 使用该图展示。板端实现中 PL 输出 logits，字符起始检测与文本判决由 PS / ARM 完成。图像按原文件保存；对应的最终实验报告见 ../design_report.docx 与 ../design_report.pdf。

paper_vs_b2tnet_activity.png 对比原论文方法与 B2Tnet 活动感知模型的参数量、CER 和 WER。原论文识别指标引用 Willett et al., Nature 2021 Table 1 的 Raw online output；B2Tnet 指标取自 software_summary.json 的模式 2，并有完整 228 句板测对拍。原网络参数量为本地同结构统计值。数据及来源在 paper_comparison.json 中登记，安装 matplotlib 后运行 generate_paper_comparison.py 可重新生成图片；Linux 可用 --font 指定中文字体。

system/policy/accuracy为真实实验数据与流程。B2Tnet_editable_figures.pptx为冻结主干两页框图，活动选择支路见设计报告。demo_ui_partial_view_20261103.png是浏览器窄窗口的局部截图，不是完整产品界面；运行与逐位证据以board/results/demo_validation_20261103.json、demo_timeline及demo_sessions为准。
