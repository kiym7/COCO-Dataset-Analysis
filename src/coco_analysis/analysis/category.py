"""类别分布与长尾分析模块。

分析 COCO 数据集中 80 个类别的分布特征，
包括频率统计、长尾效应量化和超类别分析。
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd


def category_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个类别的实例数量和占比。

    Args:
        df: 标注 DataFrame（需包含 category_id, category_name, supercategory 列）

    Returns:
        DataFrame，包含列:
        - category_id, category_name, supercategory
        - count: 实例数量
        - percentage: 占比 (%)
        - rank: 频率排名 (1 = 最多)
    """
    freq = (
        df.groupby(["category_id", "category_name", "supercategory"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    total = freq["count"].sum()
    freq["percentage"] = freq["count"] / total * 100
    freq["rank"] = range(1, len(freq) + 1)

    return freq


def compute_gini_coefficient(counts: np.ndarray) -> float:
    """计算类别分布的 Gini 系数。

    Gini 系数衡量分布的不均匀程度:
    - 0: 完全均匀（所有类别实例数相同）
    - 1: 完全不均（所有实例都属于一个类别）

    Args:
        counts: 各类别的实例数量（已排序或未排序均可）

    Returns:
        Gini 系数 (0-1)
    """
    if len(counts) <= 1:
        return 0.0

    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    index = np.arange(1, n + 1)

    gini = (2 * np.sum(index * sorted_counts)) / (n * np.sum(sorted_counts)) - (n + 1) / n
    return float(gini)


def compute_head_tail_split(
    freq_df: pd.DataFrame,
    head_ratio: float = 0.20,
    tail_ratio: float = 0.50,
) -> Dict[str, Any]:
    """将类别划分为 Head、Body 和 Tail 三部分。

    基于累积实例占比:
    - Head: 累积占比前 head_ratio (默认 20%) 的类别
    - Tail: 累积占比后 tail_ratio (默认 50%) 的类别
    - Body: 剩余的中间类别

    Args:
        freq_df: category_frequency() 的输出
        head_ratio: Head 类别的累积占比阈值
        tail_ratio: Tail 类别的累积占比阈值

    Returns:
        字典，包含:
        - head_categories: head 类别名称列表
        - body_categories: body 类别名称列表
        - tail_categories: tail 类别名称列表
        - head_count: head 类别数
        - body_count: body 类别数
        - tail_count: tail 类别数
        - head_total_pct: head 类别占总体实例的百分比
        - tail_total_pct: tail 类别占总体实例的百分比
    """
    df = freq_df.sort_values("count", ascending=False).copy()
    total = df["count"].sum()
    df["cum_pct"] = df["count"].cumsum() / total

    head_mask = df["cum_pct"] <= head_ratio
    tail_mask = df["cum_pct"] >= (1 - tail_ratio)

    head_cats = df.loc[head_mask, "category_name"].tolist()
    tail_cats = df.loc[tail_mask, "category_name"].tolist()
    body_mask = ~(head_mask | tail_mask)
    body_cats = df.loc[body_mask, "category_name"].tolist()

    head_total = df.loc[head_mask, "count"].sum() / total * 100
    tail_total = df.loc[tail_mask, "count"].sum() / total * 100

    return {
        "head_categories": head_cats,
        "body_categories": body_cats,
        "tail_categories": tail_cats,
        "head_count": len(head_cats),
        "body_count": len(body_cats),
        "tail_count": len(tail_cats),
        "head_total_pct": head_total,
        "tail_total_pct": tail_total,
    }


def images_per_category(df: pd.DataFrame) -> pd.DataFrame:
    """计算每个类别出现的图像数（即覆盖了多少张图像）。

    Args:
        df: 标注 DataFrame

    Returns:
        DataFrame，包含 columns:
        - category_id, category_name
        - image_count: 包含该类别的图像数
        - avg_instances_per_image: 在包含该类别的图像中，平均每张有几个该类别实例
    """
    # 每类出现的图像数
    img_count = (
        df.groupby(["category_id", "category_name"])["image_id"]
        .nunique()
        .reset_index(name="image_count")
    )

    # 每类在包含它的图像中的平均实例数
    inst_per_img = (
        df.groupby(["category_id"])["annotation_id"]
        .count()
        / df.groupby(["category_id"])["image_id"].nunique()
    ).reset_index(name="avg_instances_per_image")

    result = img_count.merge(inst_per_img, on="category_id")
    return result.sort_values("image_count", ascending=False).reset_index(drop=True)


def supercategory_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """分析 12 个超类别的分布情况。

    Args:
        df: 标注 DataFrame

    Returns:
        DataFrame，包含:
        - supercategory: 超类别名称
        - category_count: 包含的子类别数
        - instance_count: 总实例数
        - instance_pct: 实例占比 (%)
        - categories: 子类别名称列表
    """
    sc = df.groupby("supercategory")

    result = pd.DataFrame({
        "instance_count": sc.size(),
        "category_count": sc["category_id"].nunique(),
    }).reset_index()

    total = result["instance_count"].sum()
    result["instance_pct"] = result["instance_count"] / total * 100

    # 添加子类别列表
    cat_list = (
        df.groupby("supercategory")["category_name"]
        .apply(lambda x: sorted(x.unique()))
        .reset_index(name="categories")
    )
    result = result.merge(cat_list, on="supercategory")

    return result.sort_values("instance_count", ascending=False).reset_index(drop=True)


def long_tail_curve_data(freq_df: pd.DataFrame) -> pd.DataFrame:
    """生成长尾分布曲线的数据点。

    Args:
        freq_df: category_frequency() 的输出

    Returns:
        DataFrame，包含:
        - rank: 类别排名 (1-based)
        - count: 实例数量
        - cum_pct: 累积占比
        - category_name: 类别名称
    """
    df = freq_df.sort_values("count", ascending=False).copy()
    total = df["count"].sum()
    df["cum_pct"] = df["count"].cumsum() / total * 100
    df["rank"] = range(1, len(df) + 1)
    return df[["rank", "count", "cum_pct", "category_name"]]


def generate_category_summary(df: pd.DataFrame, config: dict | None = None) -> Dict[str, Any]:
    """生成类别分析的完整摘要。

    Args:
        df: 标注 DataFrame
        config: 分析配置（可选）

    Returns:
        包含所有类别分析结果的字典
    """
    cfg = config or {}
    lt_cfg = cfg.get("long_tail", {})
    head_ratio = lt_cfg.get("head_ratio", 0.20)
    tail_ratio = lt_cfg.get("tail_ratio", 0.50)

    freq = category_frequency(df)
    gini = compute_gini_coefficient(freq["count"].values)
    head_tail = compute_head_tail_split(freq, head_ratio=head_ratio, tail_ratio=tail_ratio)
    img_per_cat = images_per_category(df)
    supercat = supercategory_distribution(df)
    curve_data = long_tail_curve_data(freq)

    return {
        "category_frequency": freq,
        "gini_coefficient": gini,
        "total_instances": int(freq["count"].sum()),
        "total_categories": len(freq),
        "head_tail": head_tail,
        "images_per_category": img_per_cat,
        "supercategory_distribution": supercat,
        "long_tail_curve": curve_data,
        "top5_categories": freq.head(5)[["category_name", "count", "percentage"]].to_dict(
            "records"
        ),
        "bottom5_categories": freq.tail(5)[["category_name", "count", "percentage"]].to_dict(
            "records"
        ),
    }
