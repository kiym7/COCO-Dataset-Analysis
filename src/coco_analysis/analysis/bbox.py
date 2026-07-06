"""边界框统计分析模块。

分析 COCO 数据集中目标边界框的尺寸、宽高比和空间分布特征。
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def bbox_size_classification(df: pd.DataFrame) -> pd.DataFrame:
    """按 COCO 标准统计 Small/Medium/Large 目标的分布。

    Small:  area <= 32² = 1024
    Medium: 1024 < area <= 96² = 9216
    Large:  area > 96²

    Args:
        df: 标注 DataFrame（需包含 bbox_size_class 列）

    Returns:
        DataFrame，包含:
        - bbox_size_class: small / medium / large
        - count: 实例数量
        - percentage: 占比 (%)
    """
    if "bbox_size_class" not in df.columns:
        raise ValueError("DataFrame 缺少 bbox_size_class 列, 请先运行 preprocessor")

    stats = (
        df.groupby("bbox_size_class")
        .size()
        .reset_index(name="count")
    )

    total = stats["count"].sum()
    stats["percentage"] = stats["count"] / total * 100

    # 确保顺序
    order = pd.CategoricalDtype(["small", "medium", "large"], ordered=True)
    stats["bbox_size_class"] = stats["bbox_size_class"].astype(order)
    stats = stats.sort_values("bbox_size_class").reset_index(drop=True)

    return stats


def bbox_area_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """计算边界框面积的分布统计量。

    Args:
        df: 标注 DataFrame

    Returns:
        DataFrame，包含按 supercategory 分组的面积统计:
        - supercategory, count, mean_area, median_area,
          std_area, min_area, max_area, p25_area, p75_area
    """
    stats = (
        df.groupby("supercategory")["area"]
        .agg(
            count="count",
            mean_area="mean",
            median_area="median",
            std_area="std",
            min_area="min",
            max_area="max",
            p25_area=lambda x: x.quantile(0.25),
            p75_area=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    return stats.sort_values("mean_area", ascending=False)


def aspect_ratio_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """分析宽高比的统计特征。

    Args:
        df: 标注 DataFrame（需包含 aspect_ratio 列）

    Returns:
        字典，包含:
        - overall_stats: 全局统计 (mean, median, std, skew, q1, q3)
        - by_supercategory: 按超类别分组的统计
        - extreme_categories: 宽高比最极端的类别
    """
    ar = df["aspect_ratio"].dropna()

    overall = {
        "mean": float(ar.mean()),
        "median": float(ar.median()),
        "std": float(ar.std()),
        "skew": float(ar.skew()),
        "q1": float(ar.quantile(0.25)),
        "q3": float(ar.quantile(0.75)),
    }

    # 按超类别统计
    by_sc = (
        df.groupby("supercategory")["aspect_ratio"]
        .agg(mean="mean", median="median", std="std", count="count")
        .reset_index()
    )

    # 最极端的类别 (宽高比最偏离 1.0)
    extreme = (
        df.groupby("category_name")["aspect_ratio"]
        .agg(mean="mean", median="median", count="count")
        .reset_index()
    )
    extreme["deviation_from_square"] = np.abs(np.log2(extreme["median"]))
    extreme = extreme.sort_values("deviation_from_square", ascending=False).head(10)

    return {
        "overall_stats": overall,
        "by_supercategory": by_sc.to_dict("records"),
        "extreme_categories": extreme[["category_name", "median", "deviation_from_square"]].to_dict(
            "records"
        ),
    }


def position_heatmap_data(
    df: pd.DataFrame, grid_size: int = 20
) -> np.ndarray:
    """生成归一化中心点的 2D 密度矩阵，用于位置热力图。

    Args:
        df: 标注 DataFrame（需包含 center_x, center_y 列）
        grid_size: 网格分辨率（默认 20x20）

    Returns:
        (grid_size, grid_size) 形状的 numpy 数组
    """
    heatmap, _, _ = np.histogram2d(
        df["center_x"].clip(0, 1),
        df["center_y"].clip(0, 1),
        bins=grid_size,
        range=[[0, 1], [0, 1]],
    )
    return heatmap


def position_bias_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    """分析目标空间位置的偏置。

    检测目标是否倾向于出现在图像的某些区域
    (如中心偏置、底部偏置等)。

    Args:
        df: 标注 DataFrame（需包含 center_x, center_y 列）

    Returns:
        字典，包含:
        - center_x_mean: x 中心均值
        - center_y_mean: y 中心均值
        - center_bias_description: 偏置描述
        - quadrant_distribution: 四个象限的分布
    """
    cx = df["center_x"]
    cy = df["center_y"]

    cx_mean = float(cx.mean())
    cy_mean = float(cy.mean())

    # 象限分布
    quadrants = {
        "top_left": int(((cx < 0.5) & (cy < 0.5)).sum()),
        "top_right": int(((cx >= 0.5) & (cy < 0.5)).sum()),
        "bottom_left": int(((cx < 0.5) & (cy >= 0.5)).sum()),
        "bottom_right": int(((cx >= 0.5) & (cy >= 0.5)).sum()),
    }

    # 偏置描述
    bias_desc = "中心偏置" if abs(cx_mean - 0.5) < 0.05 and abs(cy_mean - 0.5) < 0.05 else ""
    if not bias_desc:
        h_dir = "左" if cx_mean < 0.48 else ("右" if cx_mean > 0.52 else "中间")
        v_dir = "上" if cy_mean < 0.48 else ("下" if cy_mean > 0.52 else "中间")
        bias_desc = f"{v_dir}{h_dir}偏置"

    return {
        "center_x_mean": cx_mean,
        "center_y_mean": cy_mean,
        "center_bias_description": bias_desc,
        "quadrant_distribution": quadrants,
    }


def per_category_bbox_stats(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个类别的边界框统计信息。

    Args:
        df: 标注 DataFrame

    Returns:
        DataFrame，每个类别一行，包含:
        - category_name
        - count, mean_area, median_area, std_area
        - mean_aspect_ratio, median_aspect_ratio
        - small_ratio: 小目标占比
        - mean_center_x, mean_center_y
    """
    stats = (
        df.groupby("category_name")
        .agg(
            count=("annotation_id", "count"),
            mean_area=("area", "mean"),
            median_area=("area", "median"),
            std_area=("area", "std"),
            mean_aspect_ratio=("aspect_ratio", "mean"),
            median_aspect_ratio=("aspect_ratio", "median"),
            small_ratio=("bbox_size_class", lambda x: (x == "small").mean() * 100),
            mean_center_x=("center_x", "mean"),
            mean_center_y=("center_y", "mean"),
        )
        .reset_index()
    )

    return stats.sort_values("count", ascending=False).reset_index(drop=True)


def generate_bbox_summary(df: pd.DataFrame, config: dict | None = None) -> Dict[str, Any]:
    """生成边界框分析的完整摘要。

    Args:
        df: 预处理后的标注 DataFrame
        config: 分析配置（可选）

    Returns:
        包含所有 bbox 分析结果的字典
    """
    return {
        "size_classification": bbox_size_classification(df),
        "area_distribution": bbox_area_distribution(df),
        "aspect_ratio": aspect_ratio_stats(df),
        "position_bias": position_bias_analysis(df),
        "per_category_stats": per_category_bbox_stats(df),
        "heatmap_data": position_heatmap_data(df),
    }
