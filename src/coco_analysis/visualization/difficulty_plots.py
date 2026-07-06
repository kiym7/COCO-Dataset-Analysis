"""检测难度相关的可视化函数。"""

from pathlib import Path
from typing import Dict, Any, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .theme import (
    get_supercategory_color,
    save_figure,
    SUPERCATEGORY_COLORS,
)


def plot_difficulty_radar(
    difficulty_by_supercat: pd.DataFrame,
    dimensions: List[str] | None = None,
    title: str = "COCO 2017 检测难度雷达图 (按超类别)",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制检测难度雷达图，每条线代表一个超类别。

    Args:
        difficulty_by_supercat: 按超类别聚合的难度数据
        dimensions: 雷达图的维度名称列表
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    if dimensions is None:
        dimensions = ["小目标比例", "目标密集度", "面积方差", "宽高比方差", "Crowd比例"]

    # 确保数据包含所需维度
    data_cols = ["small_ratio_norm", "density_norm", "area_std_norm", "ar_std_norm", "crowd_ratio_norm"]
    available_cols = [c for c in data_cols if c in difficulty_by_supercat.columns]

    if len(available_cols) < 3:
        raise ValueError("数据不足以绘制雷达图")

    n_dims = len(available_cols)
    angles = np.linspace(0, 2 * np.pi, n_dims, endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    for _, row in difficulty_by_supercat.iterrows():
        values = [row.get(c, 0) for c in available_cols]
        values += values[:1]  # 闭合

        sc = row.get("supercategory", "Unknown")
        color = get_supercategory_color(sc)

        ax.fill(angles, values, alpha=0.1, color=color)
        ax.plot(angles, values, "o-", linewidth=2, label=sc, color=color, markersize=4)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions[:len(available_cols)], fontsize=10)
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.1)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=25)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    return save_figure(fig, "12_difficulty_radar", output_dir)[0]


def plot_small_object_scatter(
    df: pd.DataFrame,
    title: str = "COCO 2017 小目标分布分析",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制小目标占比 vs 类别频率的散点气泡图。

    Args:
        df: 标注 DataFrame (需包含 category_name, bbox_size_class, supercategory 列)
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    # 聚合
    stats = df.groupby(["category_name", "supercategory"]).agg(
        total=("annotation_id", "count"),
        small_count=("bbox_size_class", lambda x: (x == "small").sum()),
        avg_size=("area", "median"),
    ).reset_index()

    stats["small_ratio"] = stats["small_count"] / stats["total"] * 100

    fig, ax = plt.subplots(figsize=(14, 9))

    for sc in stats["supercategory"].unique():
        subset = stats[stats["supercategory"] == sc]
        ax.scatter(
            subset["total"],
            subset["small_ratio"],
            s=subset["avg_size"] / 10,
            alpha=0.6,
            color=get_supercategory_color(sc),
            label=sc,
            edgecolors="white",
            linewidth=0.5,
        )

    # 标注极端类别
    for _, row in stats.iterrows():
        if row["small_ratio"] > 50 or row["total"] < 1000:
            ax.annotate(
                row["category_name"],
                (row["total"], row["small_ratio"]),
                fontsize=7,
                alpha=0.8,
                xytext=(5, 5),
                textcoords="offset points",
            )

    ax.set_xscale("log")
    ax.set_xlabel("总实例数 (对数)", fontsize=12)
    ax.set_ylabel("小目标占比 (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(alpha=0.3, linestyle="--")
    ax.axhline(y=30, color="red", linestyle="--", alpha=0.3, label="30% 小目标线")

    plt.tight_layout()
    return save_figure(fig, "13_small_object_scatter", output_dir)[0]


def plot_objects_per_image_histogram(
    df: pd.DataFrame,
    title: str = "COCO 2017 每张图像目标数量分布",
    output_dir: str = "outputs/figures",
    max_bin: int = 40,
) -> str:
    """绘制每张图像目标数量的直方图。

    Args:
        df: 标注 DataFrame
        title: 图表标题
        output_dir: 输出目录
        max_bin: 直方图最大显示值（超过此值合并到 "40+" 分箱）

    Returns:
        保存的文件路径
    """
    if "objs_per_image" in df.columns:
        obj_counts = df.groupby("image_id")["objs_per_image"].first()
    else:
        obj_counts = df.groupby("image_id").size()

    mean_val = obj_counts.mean()
    median_val = obj_counts.median()
    p95 = obj_counts.quantile(0.95)

    fig, ax = plt.subplots(figsize=(14, 7))

    # 裁剪到 max_bin
    display_counts = obj_counts.clip(upper=max_bin)

    ax.hist(
        display_counts,
        bins=50,
        color="#3498db",
        alpha=0.7,
        edgecolor="white",
        density=True,
    )

    # 统计标注
    ax.axvline(x=mean_val, color="red", linestyle="--", linewidth=2, label=f"均值: {mean_val:.1f}")
    ax.axvline(x=median_val, color="green", linestyle="--", linewidth=2, label=f"中位数: {median_val:.0f}")
    ax.axvline(x=p95, color="orange", linestyle="--", linewidth=2, label=f"95分位: {p95:.0f}")

    ax.set_xlabel("每张图像目标数", fontsize=12)
    ax.set_ylabel("密度", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return save_figure(fig, "14_objects_per_image", output_dir)[0]


def plot_difficulty_ranking(
    difficulty_df: pd.DataFrame,
    top_n: int = 30,
    title: str = "COCO 2017 类别检测难度排名",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制类别检测难度的横向条形图。

    Args:
        difficulty_df: compute_difficulty_score() 的输出 (by="category")
        top_n: 显示前 n 个最难类别
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    data = difficulty_df.head(top_n).copy()
    data = data.sort_values("difficulty_score", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 12))

    # 渐变颜色：难度越高颜色越暖
    norm = plt.Normalize(data["difficulty_score"].min(), data["difficulty_score"].max())
    colors = plt.cm.YlOrRd(norm(data["difficulty_score"]))

    bars = ax.barh(data["category_name"], data["difficulty_score"], color=colors, edgecolor="white")

    # 标注分数
    for bar, score in zip(bars, data["difficulty_score"]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.1f}",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("综合难度得分 (0-100)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    return save_figure(fig, "15_difficulty_ranking", output_dir)[0]


def plot_augmentation_recommendations(
    mosaic_recs: list,
    oversample_recs: list,
    title: str = "COCO 2017 数据增强策略建议",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制增强策略建议的综合图表。

    Args:
        mosaic_recs: recommend_mosaic_augmentation() 的返回
        oversample_recs: recommend_class_balanced_sampling() 中 oversample 部分
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 左图: Mosaic 增强优先级
    ax1 = axes[0]
    if mosaic_recs:
        mosaic_data = pd.DataFrame(mosaic_recs).head(15)
        priority_colors = {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}
        colors = [priority_colors.get(p, "#999") for p in mosaic_data["priority"]]

        ax1.barh(
            mosaic_data["category_name"],
            mosaic_data["small_ratio"],
            color=colors,
            edgecolor="white",
        )
        ax1.set_xlabel("小目标占比 (%)", fontsize=11)
        ax1.set_title("Mosaic 增强推荐 (小目标占比 > 30%)", fontsize=12, fontweight="bold")
        ax1.axvline(x=30, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax1.grid(axis="x", alpha=0.3, linestyle="--")
        ax1.invert_yaxis()

        # 图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#e74c3c", label="高优先级"),
            Patch(facecolor="#f39c12", label="中优先级"),
            Patch(facecolor="#3498db", label="低优先级"),
        ]
        ax1.legend(handles=legend_elements, fontsize=9, loc="lower right")

    # 右图: 类别过采样倍率
    ax2 = axes[1]
    if oversample_recs:
        os_data = pd.DataFrame(oversample_recs).head(15)
        ax2.barh(
            os_data["category"],
            os_data["suggested_oversample_ratio"],
            color="#9b59b6",
            alpha=0.7,
            edgecolor="white",
        )
        ax2.axvline(x=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="原始比例 (1×)")
        ax2.set_xlabel("建议过采样倍率", fontsize=11)
        ax2.set_title("Tail 类别过采样策略", fontsize=12, fontweight="bold")
        ax2.legend(fontsize=9)
        ax2.grid(axis="x", alpha=0.3, linestyle="--")
        ax2.invert_yaxis()

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    return save_figure(fig, "16_augmentation_recommendations", output_dir)[0]
