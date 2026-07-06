"""COCO 数据集加载器。

将 COCO JSON 标注文件解析为 pandas DataFrame，
提供高效的数据访问接口。
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from pycocotools.coco import COCO


def load_coco(annotation_path: str) -> COCO:
    """加载 COCO 标注文件。

    Args:
        annotation_path: COCO JSON 标注文件路径
            (如 instances_train2017.json)

    Returns:
        pycocotools.coco.COCO 对象

    Raises:
        FileNotFoundError: 标注文件不存在时抛出
    """
    path = Path(annotation_path)
    if not path.exists():
        raise FileNotFoundError(f"标注文件不存在: {annotation_path}")

    coco = COCO(str(path))
    return coco


def get_category_df(coco: COCO) -> pd.DataFrame:
    """从 COCO 对象中提取类别信息为 DataFrame。

    Args:
        coco: COCO 对象

    Returns:
        DataFrame，包含列:
        - category_id: 类别 ID (1-90, 有间隔)
        - category_name: 类别名称 (如 "person")
        - supercategory: 超类别名称 (如 "animal")
    """
    categories = coco.loadCats(coco.getCatIds())
    df = pd.DataFrame(categories)
    df = df.rename(columns={"id": "category_id", "name": "category_name"})
    return df[["category_id", "category_name", "supercategory"]]


def get_image_df(coco: COCO) -> pd.DataFrame:
    """从 COCO 对象中提取图像信息为 DataFrame。

    Args:
        coco: COCO 对象

    Returns:
        DataFrame，包含列:
        - image_id: 图像 ID
        - file_name: 图像文件名
        - width: 图像宽度 (像素)
        - height: 图像高度 (像素)
    """
    images = coco.loadImgs(coco.getImgIds())
    df = pd.DataFrame(images)
    df = df.rename(columns={"id": "image_id", "width": "image_width", "height": "image_height"})
    return df[["image_id", "file_name", "image_width", "image_height"]]


def coco_to_dataframe(coco: COCO) -> pd.DataFrame:
    """将 COCO 标注转换为扁平的 DataFrame。

    每行代表一个标注实例，包含标注信息、类别信息和图像信息。

    Args:
        coco: COCO 对象

    Returns:
        DataFrame，核心列包括:
        - annotation_id: 标注 ID
        - image_id: 所属图像 ID
        - category_id: 类别 ID
        - category_name: 类别名称
        - supercategory: 超类别名称
        - bbox_x, bbox_y, bbox_w, bbox_h: 边界框坐标 (左上角 x,y 和宽高)
        - area: 边界框面积 (像素²)
        - iscrowd: 是否为 crowd 标注
        - image_width, image_height: 所属图像的尺寸
        - file_name: 所属图像的文件名
    """
    # 加载所有标注
    ann_ids = coco.getAnnIds()
    annotations = coco.loadAnns(ann_ids)

    if not annotations:
        return pd.DataFrame()

    # 构建标注 DataFrame
    ann_df = pd.DataFrame(annotations)
    ann_df = ann_df.rename(columns={"id": "annotation_id", "image_id": "image_id"})

    # 提取 bbox 数组为独立列
    bbox_array = np.array(ann_df["bbox"].tolist())
    ann_df["bbox_x"] = bbox_array[:, 0]
    ann_df["bbox_y"] = bbox_array[:, 1]
    ann_df["bbox_w"] = bbox_array[:, 2]
    ann_df["bbox_h"] = bbox_array[:, 3]

    # 合并类别信息
    cat_df = get_category_df(coco)
    ann_df = ann_df.merge(cat_df, on="category_id", how="left")

    # 合并图像信息
    img_df = get_image_df(coco)
    ann_df = ann_df.merge(img_df, on="image_id", how="left", suffixes=("", "_img"))

    return ann_df


def build_combined_df(coco: COCO) -> pd.DataFrame:
    """构建完整的合并 DataFrame，包含标注、类别和图像信息。

    等同于 coco_to_dataframe()，函数名更语义化。

    Args:
        coco: COCO 对象

    Returns:
        合并后的完整 DataFrame
    """
    return coco_to_dataframe(coco)


def get_dataset_summary(coco: COCO) -> dict:
    """获取数据集的基本摘要信息。

    Args:
        coco: COCO 对象

    Returns:
        包含以下键的字典:
        - num_images: 图像总数
        - num_annotations: 标注总数
        - num_categories: 类别总数
        - category_names: 类别名称列表
        - supercategories: 超类别名称列表
    """
    return {
        "num_images": len(coco.getImgIds()),
        "num_annotations": len(coco.getAnnIds()),
        "num_categories": len(coco.getCatIds()),
        "category_names": [cat["name"] for cat in coco.loadCats(coco.getCatIds())],
        "supercategories": list(
            set(cat["supercategory"] for cat in coco.loadCats(coco.getCatIds()))
        ),
    }
