"""边界框相关的可视化函数。"""

from pathlib import Path
from typing import Dict, Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

from .theme import (
    get_supercategory_color,
    save_figure,
    SUPERCATEGORY_COLORS,
)


def plot_bbox_size_pie(
    size_df: pd.DataFrame,
    title: str = "COCO 2017 目标尺寸分布 (Small / Medium / Large)",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制目标尺寸分布的环形饼图。

    Args:
        size_df: bbox_size_classification() 的输出
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    labels_map = {"small": "Small\n(≤32×32 px)", "medium": "Medium\n(32²-96² px)", "large": "Large\n(>96² px)"}

    sizes = size_df.set_index("bbox_size_class")["count"]
    labels = [labels_map.get(k, k) for k in sizes.index]

    wedges, texts, autotexts = ax.pie(
        sizes.values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        pctdistance=0.75,
        explode=(0.02, 0.02, 0.02),
    )

    # 中心添加总实例数
    total = sizes.sum()
    ax.text(
        0, 0,
        f"{total:,}\n实例总数",
        ha="center", va="center",
        fontsize=14,
        fontweight="bold",
    )

    # 美化
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    for text in texts:
        text.set_fontsize(10)

    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    return save_figure(fig, "04_bbox_size_pie", output_dir)[0]


def plot_aspect_ratio_distribution(
    df: pd.DataFrame,
    title: str = "COCO 2017 目标宽高比分布",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制按超类别分层的宽高比分布图。

    Args:
        df: 标注 DataFrame (需包含 aspect_ratio 和 supercategory 列)
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    # 过滤极端值
    ar_data = df[np.abs(np.log2(df["aspect_ratio"])) < 4].copy()

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # 左图: 分层直方图
    ax1 = axes[0]
    supercategories = ar_data["supercategory"].value_counts().index[:8]  # Top 8 超类别

    for sc in supercategories:
        subset = ar_data[ar_data["supercategory"] == sc]
        log_ar = np.log2(subset["aspect_ratio"])
        ax1.hist(
            log_ar,
            bins=60,
            alpha=0.4,
            label=sc,
            color=get_supercategory_color(sc),
            density=True,
        )

    # 标注关键宽高比
    for val, label in [(-1, "1:2"), (0, "1:1"), (0.7, "4:3"), (1.25, "16:9")]:
        ax1.axvline(x=val, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.text(val, ax1.get_ylim()[1] * 0.95, label, ha="center", fontsize=8, color="gray")

    ax1.set_xlabel("log₂(宽高比)", fontsize=12)
    ax1.set_ylabel("密度", fontsize=12)
    ax1.set_title("宽高比分布 (按超类别)", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=8, ncol=2)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.set_xlim(-3, 3)

    # 右图: 按尺寸分类的 boxplot
    ax2 = axes[1]
    size_order = ["small", "medium", "large"]
    data_for_box = ar_data[ar_data["bbox_size_class"].isin(size_order)].copy()
    data_for_box["log_ar"] = np.log2(data_for_box["aspect_ratio"])

    bp = data_for_box.boxplot(
        column="log_ar",
        by="bbox_size_class",
        ax=ax2,
        grid=True,
        patch_artist=True,
        boxprops=dict(facecolor="#aec7e8", alpha=0.7),
        medianprops=dict(color="darkblue", linewidth=2),
    )

    ax2.set_xlabel("目标尺寸类别", fontsize=12)
    ax2.set_ylabel("log₂(宽高比)", fontsize=12)
    ax2.set_title("宽高比 vs 目标尺寸", fontsize=13, fontweight="bold")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax2.grid(alpha=0.3, linestyle="--")

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()
    return save_figure(fig, "05_aspect_ratio_distribution", output_dir)[0]


def plot_position_heatmap(
    df: pd.DataFrame,
    grid_size: int = 30,
    title: str = "COCO 2017 目标空间位置热力图 (归一化坐标)",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制目标空间位置 2D 热力图。

    Args:
        df: 标注 DataFrame (需包含 center_x, center_y 列)
        grid_size: 热力图分辨率
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    heatmap, xedges, yedges = np.histogram2d(
        df["center_x"].clip(0, 1),
        df["center_y"].clip(0, 1),
        bins=grid_size,
        range=[[0, 1], [0, 1]],
    )

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(
        heatmap.T,
        origin="upper",
        extent=[0, 1, 1, 0],  # 翻转 y 轴
        cmap="YlOrRd",
        aspect="equal",
        interpolation="bilinear",
    )

    # 标注中心偏置方向
    cx_mean = df["center_x"].mean()
    cy_mean = df["center_y"].mean()
    ax.scatter([cx_mean], [cy_mean], marker="+", color="blue", s=200, linewidths=2, label=f"中心均值 ({cx_mean:.3f}, {cy_mean:.3f})")
    ax.scatter([0.5], [0.5], marker="x", color="white", s=100, linewidths=1.5, label="图像中心")

    # 颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("目标密度", fontsize=11)

    ax.set_xlabel("归一化 X 坐标", fontsize=12)
    ax.set_ylabel("归一化 Y 坐标", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")

    return save_figure(fig, "06_position_heatmap", output_dir)[0]


def plot_bbox_area_violin(
    df: pd.DataFrame,
    title: str = "COCO 2017 目标面积分布 (按超类别)",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制按超类别分组的目标面积小提琴图。

    Args:
        df: 标注 DataFrame (需包含 log_area 和 supercategory 列)
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    # 按大小排序超类别
    order = df.groupby("supercategory")["area"].median().sort_values(ascending=False).index.tolist()

    fig, ax = plt.subplots(figsize=(14, 8))

    parts = ax.violinplot(
        [df[df["supercategory"] == sc]["log_area"].dropna().values for sc in order],
        positions=range(len(order)),
        showmeans=True,
        showmedians=True,
        widths=0.7,
    )

    # 着色
    for i, sc in enumerate(order):
        color = get_supercategory_color(sc)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=10)
    ax.set_xlabel("超类别", fontsize=12)
    ax.set_ylabel("log₁₀(目标面积)", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return save_figure(fig, "07_bbox_area_violin", output_dir)[0]


def plot_size_by_category_heatmap(
    df: pd.DataFrame,
    title: str = "COCO 2017 各类别目标尺寸分布热力图",
    output_dir: str = "outputs/figures",
    top_n: int = 30,
) -> str:
    """绘制各类别在不同目标密度下的平均面积热力图。

    Args:
        df: 标注 DataFrame
        title: 图表标题
        output_dir: 输出目录
        top_n: 显示前 n 个类别

    Returns:
        保存的文件路径
    """
    top_cats = df["category_name"].value_counts().head(top_n).index.tolist()
    data = df[df["category_name"].isin(top_cats)].copy()

    # 按目标密度分箱
    data["density_bin"] = pd.cut(
        data["objs_per_image"],
        bins=[0, 3, 7, 15, 30, 100],
        labels=["1-3", "4-7", "8-15", "16-30", "31+"],
    )

    pivot = data.pivot_table(
        values="area",
        index="category_name",
        columns="density_bin",
        aggfunc="median",
    )

    # 按总数排序
    pivot = pivot.loc[top_cats]

    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".0f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "中位面积 (px²)"},
        linewidths=0.5,
    )

    ax.set_xlabel("每图目标数 (密度分箱)", fontsize=12)
    ax.set_ylabel("类别", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    return save_figure(fig, "08_size_density_heatmap", output_dir)[0]
