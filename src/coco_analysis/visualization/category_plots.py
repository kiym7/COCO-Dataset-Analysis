"""类别分布相关的可视化函数。"""

from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .theme import (
    get_color_palette,
    get_supercategory_color,
    save_figure,
    SUPERCATEGORY_COLORS,
)


def plot_category_distribution(
    freq_df: pd.DataFrame,
    top_k: int = 80,
    title: str = "COCO 2017 类别实例分布",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制类别实例分布的横向条形图。

    Args:
        freq_df: category_frequency() 的输出
        top_k: 显示前 k 个类别
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    data = freq_df.head(top_k).copy()
    data = data.sort_values("count", ascending=True)  # 从小到大便于横向显示

    fig, ax = plt.subplots(figsize=(14, 18))

    # 按超类别着色
    colors = [get_supercategory_color(sc) for sc in data["supercategory"]]

    bars = ax.barh(data["category_name"], data["count"], color=colors, edgecolor="white", linewidth=0.3)

    # 标注 Top5 和 Bottom5
    for i, (_, row) in enumerate(data.iterrows()):
        if row["rank"] <= 5:
            ax.text(
                row["count"] + max(data["count"]) * 0.01,
                i,
                f"#{int(row['rank'])}",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#333",
            )

    # 图例
    legend_handles = []
    for sc, color in SUPERCATEGORY_COLORS.items():
        if sc in data["supercategory"].values:
            legend_handles.append(
                plt.Rectangle((0, 0), 1, 1, facecolor=color, label=sc)
            )
    ax.legend(
        handles=legend_handles,
        title="超类别",
        loc="lower right",
        fontsize=8,
        title_fontsize=9,
    )

    ax.set_xlabel("实例数量", fontsize=13)
    ax.set_ylabel("类别", fontsize=13)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)

    # 格式化 x 轴
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.tick_params(axis="y", labelsize=9)

    # 添加网格
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return save_figure(fig, "01_category_distribution", output_dir)[0]


def plot_long_tail_curve(
    freq_df: pd.DataFrame,
    head_ratio: float = 0.20,
    tail_ratio: float = 0.50,
    gini: Optional[float] = None,
    title: str = "COCO 2017 长尾分布曲线",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制双对数坐标下的长尾分布曲线。

    Args:
        freq_df: category_frequency() 的输出
        head_ratio: Head 区域的累积占比阈值
        tail_ratio: Tail 区域的累积占比阈值
        gini: Gini 系数（可选，用于标注）
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    data = freq_df.sort_values("count", ascending=False).copy()
    total = data["count"].sum()
    data["cum_pct"] = data["count"].cumsum() / total

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左图: 排名 vs 实例数 (双对数)
    ax1 = axes[0]
    ax1.loglog(
        data["rank"],
        data["count"],
        "o-",
        markersize=4,
        linewidth=1.5,
        color="#1f77b4",
        alpha=0.8,
    )

    # Head/Tail 区域着色
    head_threshold_rank = int(len(data) * head_ratio)
    tail_threshold_rank = int(len(data) * (1 - tail_ratio))

    ax1.axvspan(1, head_threshold_rank, alpha=0.1, color="green", label=f"Head ({head_ratio:.0%})")
    ax1.axvspan(tail_threshold_rank, len(data), alpha=0.1, color="red", label=f"Tail ({tail_ratio:.0%})")

    ax1.set_xlabel("类别排名 (对数)", fontsize=12)
    ax1.set_ylabel("实例数量 (对数)", fontsize=12)
    ax1.set_title("实例数量分布 (双对数)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3, linestyle="--")

    if gini is not None:
        ax1.text(
            0.95, 0.95,
            f"Gini 系数: {gini:.3f}",
            transform=ax1.transAxes,
            fontsize=11,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # 右图: 累积分布曲线
    ax2 = axes[1]
    ax2.plot(data["rank"], data["cum_pct"] * 100, linewidth=2, color="#2ca02c")
    ax2.axhline(y=head_ratio * 100, color="green", linestyle="--", alpha=0.6, label=f"Head 阈值 ({head_ratio:.0%})")
    ax2.axhline(y=(1 - tail_ratio) * 100, color="red", linestyle="--", alpha=0.6, label=f"Tail 阈值 ({tail_ratio:.0%})")

    # 标注 Head 类别数
    head_count = head_threshold_rank
    ax2.axvline(x=head_count, color="green", linestyle=":", alpha=0.5)
    ax2.annotate(
        f"{head_count} 类 = {head_ratio:.0%} 实例",
        xy=(head_count, head_ratio * 100),
        xytext=(head_count + 5, head_ratio * 100 + 10),
        arrowprops=dict(arrowstyle="->", alpha=0.6),
        fontsize=10,
    )

    ax2.set_xlabel("类别排名", fontsize=12)
    ax2.set_ylabel("累积实例占比 (%)", fontsize=12)
    ax2.set_title("实例累积分布", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, linestyle="--")
    ax2.set_ylim(0, 105)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return save_figure(fig, "02_long_tail_curve", output_dir)[0]


def plot_supercategory_treemap(
    supercat_df: pd.DataFrame,
    output_dir: str = "outputs/figures",
) -> str:
    """使用 plotly 绘制超类别 treemap（交互式）。

    Args:
        supercat_df: supercategory_distribution() 的输出
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    import plotly.express as px

    # 构建层次数据: 超类别 → 类别
    rows = []
    for _, sc_row in supercat_df.iterrows():
        sc_name = sc_row["supercategory"]
        categories = sc_row["categories"]
        for cat in categories:
            rows.append({
                "supercategory": sc_name,
                "category": cat,
            })

    tree_df = pd.DataFrame(rows)

    # 计算每个节点的实例数
    cat_counts = pd.DataFrame()  # placeholder
    # 简化版: 直接用超类别的实例数
    fig = px.treemap(
        supercat_df,
        path=["supercategory"],
        values="instance_count",
        color="instance_count",
        color_continuous_scale="Viridis",
        title="COCO 2017 超类别分布 Treemap",
    )

    fig.update_layout(
        width=900,
        height=600,
        margin=dict(t=50, l=25, r=25, b=25),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "03_supercategory_treemap.html"
    fig.write_html(str(filepath))
    return str(filepath)


def plot_category_coverage(
    df: pd.DataFrame,
    title: str = "COCO 2017 类别-图像覆盖分析",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制类别覆盖率分析图。

    Args:
        df: 标注 DataFrame
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    total_images = df["image_id"].nunique()

    cat_img_counts = (
        df.groupby("category_name")["image_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    x = range(1, len(cat_img_counts) + 1)
    y = cat_img_counts.values

    ax.bar(x, y, color="#3498db", alpha=0.8, edgecolor="white")
    ax.axhline(
        y=total_images * 0.5,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"50% 图像覆盖线 ({total_images * 0.5:.0f})",
    )

    # 标注覆盖 50% 图像的类别数
    over_half = (cat_img_counts >= total_images * 0.5).sum()
    ax.annotate(
        f"{over_half} 个类别覆盖了\n超过 50% 的图像",
        xy=(over_half, total_images * 0.5),
        xytext=(over_half + 5, total_images * 0.65),
        arrowprops=dict(arrowstyle="->", color="red", alpha=0.7),
        fontsize=11,
        color="darkred",
    )

    ax.set_xlabel("类别排名", fontsize=12)
    ax.set_ylabel("出现的图像数", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return save_figure(fig, "03_category_coverage", output_dir)[0]
