"""检测难度量化模块。

从多个维度评估目标检测的难度:
- 小目标比例（面积 < 32² 的目标越多越难）
- 目标密集度（每张图像目标越多越难）
- 尺度多样性（同一图像内面积方差越大越难）
- 宽高比极端程度（非正方形目标更难检测）
- Crowd 标注比例（密集人群等场景更难）
"""

from typing import Any, Dict

import numpy as np
import pandas as pd


def small_object_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个类别中小目标的占比。

    Args:
        df: 标注 DataFrame（需包含 bbox_size_class 和 category_name 列）

    Returns:
        DataFrame，包含:
        - category_name: 类别名称
        - total_count: 该类别的总实例数
        - small_count: 小目标数量
        - small_ratio: 小目标占比 (%)
    """
    per_cat = df.groupby("category_name").agg(
        total_count=("annotation_id", "count"),
        small_count=("bbox_size_class", lambda x: (x == "small").sum()),
    ).reset_index()

    per_cat["small_ratio"] = per_cat["small_count"] / per_cat["total_count"] * 100
    return per_cat.sort_values("small_ratio", ascending=False).reset_index(drop=True)


def compute_objects_per_image_distribution(df: pd.DataFrame) -> pd.Series:
    """计算每张图像的目标数量分布。

    Args:
        df: 标注 DataFrame（需包含 objs_per_image 列）

    Returns:
        Series，包含描述性统计
    """
    if "objs_per_image" not in df.columns:
        obj_counts = df.groupby("image_id").size()
    else:
        obj_counts = df.groupby("image_id")["objs_per_image"].first()

    return obj_counts.describe()


def compute_area_variance_per_image(df: pd.DataFrame) -> pd.DataFrame:
    """计算同一图像内边界框面积的方差（衡量尺度多样性）。

    Args:
        df: 标注 DataFrame

    Returns:
        DataFrame，包含:
        - image_id
        - area_variance: 该图像内面积的方差
        - area_cv: 变异系数 (std/mean)
        - num_objects: 该图像的目标数
    """
    stats = (
        df.groupby("image_id")["area"]
        .agg(area_variance=("var",), area_mean=("mean",), num_objects=("count",))
        .reset_index()
    )

    # 变异系数
    stats["area_cv"] = np.sqrt(stats["area_variance"].clip(lower=0)) / stats[
        "area_mean"
    ].replace(0, np.nan)

    return stats


def compute_crowd_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """统计 Crowd 标注的情况。

    Args:
        df: 标注 DataFrame（需包含 iscrowd 列，且未过滤 crowd）

    Returns:
        字典，包含 crowd 统计信息
    """
    if "iscrowd" not in df.columns:
        return {"total_crowd": 0, "crowd_ratio": 0.0, "crowd_by_category": []}

    total = len(df)
    crowd = df[df["iscrowd"] == 1]
    crowd_count = len(crowd)

    crowd_by_cat = (
        crowd.groupby("category_name")
        .size()
        .reset_index(name="crowd_count")
        .sort_values("crowd_count", ascending=False)
    )

    return {
        "total_crowd": crowd_count,
        "crowd_ratio": crowd_count / total * 100 if total > 0 else 0.0,
        "crowd_by_category": crowd_by_cat.to_dict("records"),
    }


def compute_difficulty_score(
    df: pd.DataFrame,
    weights: Dict[str, float] | None = None,
    by: str = "category",
) -> pd.DataFrame:
    """计算综合检测难度评分 (0-100)。

    评分维度与权重:
    - small_object_ratio (默认 35%): 小目标越多越难
    - density (默认 25%): 目标越密集越难
    - area_variance (默认 20%): 尺度越多样越难
    - aspect_ratio_variance (默认 10%): 宽高比越极端越难
    - crowd_ratio (默认 10%): crowd 越多越难

    Args:
        df: 标注 DataFrame
        weights: 各维度的权重字典
        by: 分组维度 'category' (按类别) 或 'image' (按图像)

    Returns:
        DataFrame，包含各维度的得分和综合得分
    """
    if weights is None:
        weights = {
            "small_object_ratio": 0.35,
            "density": 0.25,
            "area_variance": 0.20,
            "aspect_ratio_variance": 0.10,
            "crowd_ratio": 0.10,
        }

    group_col = "category_name" if by == "category" else "image_id"

    # 各维度计算
    # Build aggregations compatible with pandas 2.x
    scores = df.groupby(group_col).agg(
        total_count=("annotation_id", "count"),
    )

    if "bbox_size_class" in df.columns:
        small_stats = df.groupby(group_col)["bbox_size_class"].apply(
            lambda x: (x == "small").mean()
        ).reset_index(name="small_ratio")
        scores = scores.merge(small_stats, on=group_col, how="left")

    if "objs_per_image" in df.columns:
        density_stats = df.groupby(group_col)["objs_per_image"].first().reset_index()
        density_stats.columns = [group_col, "density"]
        scores = scores.merge(density_stats, on=group_col, how="left")

    if "area" in df.columns:
        area_stats = df.groupby(group_col)["area"].agg(
            area_std="std", area_mean="mean"
        ).reset_index()
        scores = scores.merge(area_stats, on=group_col, how="left")

    if "aspect_ratio" in df.columns:
        ar_stats = df.groupby(group_col)["aspect_ratio"].std().reset_index(name="ar_std")
        scores = scores.merge(ar_stats, on=group_col, how="left")

    if "iscrowd" in df.columns:
        crowd_stats = df.groupby(group_col)["iscrowd"].apply(
            lambda x: (x == 1).mean()
        ).reset_index(name="crowd_ratio")
        scores = scores.merge(crowd_stats, on=group_col, how="left")

    # 每列归一化到 [0, 1]
    normalize_cols = []
    for col in ["small_ratio", "density", "area_std", "ar_std", "crowd_ratio"]:
        if col in scores.columns and scores[col].notna().any():
            min_val = scores[col].min()
            max_val = scores[col].max()
            if max_val > min_val:
                scores[f"{col}_norm"] = (scores[col] - min_val) / (max_val - min_val)
                normalize_cols.append(f"{col}_norm")
            else:
                scores[f"{col}_norm"] = 0.0
                normalize_cols.append(f"{col}_norm")

    # 加权综合得分
    scores["difficulty_score"] = 0.0
    weight_map = {
        "small_ratio_norm": weights.get("small_object_ratio", 0.35),
        "density_norm": weights.get("density", 0.25),
        "area_std_norm": weights.get("area_variance", 0.20),
        "ar_std_norm": weights.get("aspect_ratio_variance", 0.10),
        "crowd_ratio_norm": weights.get("crowd_ratio", 0.10),
    }

    for col, w in weight_map.items():
        if col in scores.columns:
            scores["difficulty_score"] += scores[col].fillna(0) * w

    # 缩放到 0-100
    scores["difficulty_score"] = scores["difficulty_score"] * 100

    return scores.sort_values("difficulty_score", ascending=False).reset_index(drop=True)


def rank_categories_by_difficulty(df: pd.DataFrame) -> pd.DataFrame:
    """按综合难度对类别排名。

    Args:
        df: 标注 DataFrame

    Returns:
        按 difficulty_score 降序排列的 DataFrame
    """
    return compute_difficulty_score(df, by="category")


def generate_difficulty_summary(
    df: pd.DataFrame, config: dict | None = None
) -> Dict[str, Any]:
    """生成检测难度分析的完整摘要。

    Args:
        df: 预处理后的标注 DataFrame
        config: 分析配置

    Returns:
        包含所有难度分析结果的字典
    """
    cfg = config or {}
    diff_cfg = cfg.get("difficulty", {})
    weights = diff_cfg.get("weights", None)

    difficulty_by_category = compute_difficulty_score(df, weights=weights, by="category")
    difficulty_by_image = compute_difficulty_score(df, weights=weights, by="image")

    return {
        "small_object_by_category": small_object_ratio(df),
        "objects_per_image_stats": compute_objects_per_image_distribution(df),
        "crowd_stats": compute_crowd_stats(df),
        "difficulty_by_category": difficulty_by_category,
        "difficulty_by_image": difficulty_by_image,
        "hardest_categories": difficulty_by_category.head(10)[
            ["category_name", "difficulty_score", "total_count"]
        ].to_dict("records"),
        "easiest_categories": difficulty_by_category.tail(10)[
            ["category_name", "difficulty_score", "total_count"]
        ].to_dict("records"),
    }
