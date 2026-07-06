"""数据预处理器。

在原始 COCO DataFrame 上添加派生列，供后续分析模块使用。
"""

import numpy as np
import pandas as pd


def add_bbox_derived_columns(
    df: pd.DataFrame,
    small_area_max: float = 1024.0,
    medium_area_max: float = 9216.0,
) -> pd.DataFrame:
    """为 DataFrame 添加边界框相关的派生列。

    添加的列:
    - aspect_ratio: 宽高比 (w/h)
    - center_x: 边界框中心 x 坐标（归一化到图像宽度）
    - center_y: 边界框中心 y 坐标（归一化到图像高度）
    - bbox_size_class: 尺寸分类 (small / medium / large)
    - normalized_area: 归一化面积 (bbox_area / image_area)
    - log_area: 面积的对数值 (便于可视化)
    - is_square: 是否接近正方形 (0.8 < aspect_ratio < 1.25)

    Args:
        df: 标注 DataFrame（需包含 bbox_w, bbox_h, area, image_width, image_height 列）
        small_area_max: 小目标面积阈值（默认 32*32=1024）
        medium_area_max: 中目标面积阈值（默认 96*96=9216）

    Returns:
        添加了派生列的 DataFrame
    """
    result = df.copy()

    # 宽高比 (避免除零)
    result["aspect_ratio"] = result["bbox_w"] / result["bbox_h"].replace(0, np.nan)

    # 归一化中心坐标
    result["center_x"] = (result["bbox_x"] + result["bbox_w"] / 2) / result[
        "image_width"
    ]
    result["center_y"] = (result["bbox_y"] + result["bbox_h"] / 2) / result[
        "image_height"
    ]

    # 尺寸分类 (按 COCO 标准)
    area = result["area"]
    conditions = [
        area <= small_area_max,
        (area > small_area_max) & (area <= medium_area_max),
        area > medium_area_max,
    ]
    choices = ["small", "medium", "large"]
    result["bbox_size_class"] = np.select(conditions, choices, default="unknown")

    # 归一化面积
    result["normalized_area"] = result["area"] / (
        result["image_width"] * result["image_height"]
    )

    # 对数面积
    result["log_area"] = np.log10(result["area"].clip(lower=1))

    # 是否接近正方形
    ar = result["aspect_ratio"]
    result["is_square"] = (ar > 0.8) & (ar < 1.25)

    return result


def add_image_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """为 DataFrame 添加图像级别的派生统计列。

    添加的列:
    - objs_per_image: 每张图像的标注实例总数
    - cats_per_image: 每张图像的独有类别数

    Args:
        df: 标注 DataFrame（需包含 image_id 和 category_id 列）

    Returns:
        添加了图像统计列的 DataFrame
    """
    result = df.copy()

    # 每张图像的目标数量
    obj_counts = result.groupby("image_id").size().reset_index(name="objs_per_image")
    result = result.merge(obj_counts, on="image_id", how="left")

    # 每张图像的独有类别数
    cat_counts = (
        result.groupby("image_id")["category_id"]
        .nunique()
        .reset_index(name="cats_per_image")
    )
    result = result.merge(cat_counts, on="image_id", how="left")

    return result


def filter_annotations(
    df: pd.DataFrame,
    remove_crowd: bool = True,
    min_area: float | None = None,
) -> pd.DataFrame:
    """过滤标注数据。

    Args:
        df: 标注 DataFrame
        remove_crowd: 是否移除 crowd 标注（默认 True）
        min_area: 最小面积阈值，为 None 时不筛选

    Returns:
        过滤后的 DataFrame
    """
    result = df.copy()

    if remove_crowd and "iscrowd" in result.columns:
        result = result[result["iscrowd"] == 0]

    if min_area is not None:
        result = result[result["area"] >= min_area]

    return result


def pipeline(
    df: pd.DataFrame,
    config: dict | None = None,
    remove_crowd: bool = True,
) -> pd.DataFrame:
    """执行完整的数据预处理管线。

    依次执行: 过滤 → 添加 bbox 派生列 → 添加图像派生列

    Args:
        df: 原始标注 DataFrame
        config: 分析配置字典（可选，用于阈值参数）
        remove_crowd: 是否移除 crowd 标注

    Returns:
        预处理完成的 DataFrame
    """
    cfg = config or {}

    bbox_cfg = cfg.get("bbox", {})
    small_max = bbox_cfg.get("small_area_max", 1024.0)
    medium_max = bbox_cfg.get("medium_area_max", 9216.0)

    df = filter_annotations(df, remove_crowd=remove_crowd)
    df = add_bbox_derived_columns(df, small_area_max=small_max, medium_area_max=medium_max)
    df = add_image_derived_columns(df)

    return df
