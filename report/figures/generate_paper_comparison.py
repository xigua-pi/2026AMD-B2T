"""按已登记的文献指标与板测结果生成参数量、CER 和 WER 对比图。"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font", type=Path, help="可选的中文字体文件")
    args = parser.parse_args()

    candidates = [args.font, Path("C:/Windows/Fonts/msyh.ttc")]
    for path in candidates:
        if path is not None and path.is_file():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=str(path)
            ).get_name()
            break
    plt.rcParams.update({"font.size": 13, "axes.unicode_minus": False})

    data = json.loads((root / "paper_comparison.json").read_text(encoding="utf-8"))
    paper, model = data["paper"], data["b2tnet_activity"]
    # 原论文未直接报告这个参数总数，图中以星号标注同结构本地统计值。
    parameters = [paper["local_same_architecture_parameter_count"], model["parameter_count"]]
    panels = [
        ("参数量（M） ↓", [x / 1e6 for x in parameters], ["3.048*", "0.278"], 3.75),
        ("字符错误率 CER（%） ↓", [paper["cer_percent"], model["cer_percent"]], ["5.90%", "9.70%"], 12),
        ("词错误率 WER（%） ↓", [paper["wer_percent"], model["wer_percent"]], ["25.10%", "37.63%"], 46),
    ]
    colors = ["#A0A0A0", "#AA1B38"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (title, values, labels, upper) in zip(axes, panels):
        ax.bar([0, 1], values, width=0.52, color=colors, zorder=3)
        ax.set_title(title, fontsize=15, pad=16)
        ax.set_xticks([0, 1], ["原论文方法", "本方法"])
        ax.set_ylim(0, upper)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E1E1E1", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#999999")
        ax.tick_params(axis="both", length=0, labelcolor="#444444", pad=8)
        for x, value, label in zip([0, 1], values, labels):
            ax.text(x, value + upper * 0.025, label, ha="center", va="bottom", fontsize=15, fontweight="bold")

    fig.suptitle("原论文方法与 B2Tnet 活动感知模型", fontsize=19, y=0.98)
    fig.legend(
        handles=[Patch(facecolor=colors[0], label="原论文方法"), Patch(facecolor=colors[1], label="本方法（活动感知）")],
        loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False,
    )
    # 明确文献在线实验与本方法离线回放的区别，避免绘图暗示实验设置完全相同。
    fig.text(0.5, 0.05, "原论文：Raw online；本方法：官方 HeldOutBlocks 228 句 FPGA 回放，均不使用语言模型。", ha="center", fontsize=10, color="#555555")
    fig.text(0.5, 0.005, "* 原网络参数量按本地同结构模型统计，含 10 个会话校准及初始状态；非论文直接报告值。", ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.73, bottom=0.21, wspace=0.35)
    fig.savefig(root / "paper_vs_b2tnet_activity.png", dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


if __name__ == "__main__":
    main()
