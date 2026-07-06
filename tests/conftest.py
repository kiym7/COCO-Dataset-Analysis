"""pytest 测试夹具。

生成合成 COCO 数据集用于单元测试，无需下载真实数据。
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest


# ---- 合成数据定义 ----

SYNTHETIC_CATEGORIES = [
    {"id": 1, "name": "person", "supercategory": "person"},
    {"id": 2, "name": "bicycle", "supercategory": "vehicle"},
    {"id": 3, "name": "car", "supercategory": "vehicle"},
    {"id": 4, "name": "dog", "supercategory": "animal"},
    {"id": 5, "name": "cat", "supercategory": "animal"},
]

SYNTHETIC_IMAGES = [
    {"id": 101, "file_name": "img_001.jpg", "width": 640, "height": 480},
    {"id": 102, "file_name": "img_002.jpg", "width": 800, "height": 600},
    {"id": 103, "file_name": "img_003.jpg", "width": 1024, "height": 768},
]

SYNTHETIC_ANNOTATIONS = [
    # img_001: person + dog + car
    {
        "id": 1001, "image_id": 101, "category_id": 1,
        "bbox": [100, 80, 200, 350], "area": 70000, "iscrowd": 0,
    },
    {
        "id": 1002, "image_id": 101, "category_id": 4,
        "bbox": [400, 300, 80, 60], "area": 4800, "iscrowd": 0,
    },
    {
        "id": 1003, "image_id": 101, "category_id": 3,
        "bbox": [50, 200, 150, 100], "area": 15000, "iscrowd": 0,
    },
    # img_002: person + bicycle + person (crowd)
    {
        "id": 1004, "image_id": 102, "category_id": 1,
        "bbox": [300, 100, 50, 150], "area": 7500, "iscrowd": 0,
    },
    {
        "id": 1005, "image_id": 102, "category_id": 2,
        "bbox": [200, 250, 120, 80], "area": 9600, "iscrowd": 0,
    },
    {
        "id": 1006, "image_id": 102, "category_id": 1,
        "bbox": [500, 200, 180, 120], "area": 21600, "iscrowd": 0,
    },
    {
        "id": 1007, "image_id": 102, "category_id": 1,
        "bbox": [0, 0, 800, 600], "area": 480000, "iscrowd": 1,  # crowd
    },
    # img_003: car + dog (small) + cat
    {
        "id": 1008, "image_id": 103, "category_id": 3,
        "bbox": [200, 300, 300, 200], "area": 60000, "iscrowd": 0,
    },
    {
        "id": 1009, "image_id": 103, "category_id": 4,
        "bbox": [100, 500, 20, 15], "area": 300, "iscrowd": 0,  # small
    },
    {
        "id": 1010, "image_id": 103, "category_id": 5,
        "bbox": [700, 100, 40, 35], "area": 1400, "iscrowd": 0,
    },
]


def build_synthetic_coco_json() -> Dict[str, Any]:
    """构建合成 COCO JSON 数据。

    Returns:
        符合 COCO 格式的字典
    """
    return {
        "info": {
            "description": "Synthetic COCO Dataset for Testing",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "images": SYNTHETIC_IMAGES,
        "annotations": SYNTHETIC_ANNOTATIONS,
        "categories": SYNTHETIC_CATEGORIES,
    }


@pytest.fixture
def synthetic_coco_json() -> Dict[str, Any]:
    """返回合成 COCO JSON 数据字典。"""
    return build_synthetic_coco_json()


@pytest.fixture
def synthetic_coco_path(tmp_path: Path) -> Path:
    """将合成 COCO 数据写入临时文件，返回文件路径。

    Args:
        tmp_path: pytest 内置临时目录 fixture

    Returns:
        临时 JSON 文件的 Path
    """
    data = build_synthetic_coco_json()
    file_path = tmp_path / "instances_synthetic.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    return file_path


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """返回一个手动构建的、包含派生列的示例 DataFrame，
    供分析模块测试使用（不依赖 pycocotools）。
    """
    np.random.seed(42)
    n = 50

    df = pd.DataFrame({
        "annotation_id": range(1, n + 1),
        "image_id": np.repeat([101, 102, 103, 104, 105], 10),
        "category_id": np.tile([1, 2, 3, 4, 5], 10),
        "category_name": np.tile(
            ["person", "bicycle", "car", "dog", "cat"], 10
        ),
        "supercategory": np.tile(
            ["person", "vehicle", "vehicle", "animal", "animal"], 10
        ),
        "bbox_x": np.random.uniform(0, 500, n),
        "bbox_y": np.random.uniform(0, 400, n),
        "bbox_w": np.random.uniform(20, 300, n),
        "bbox_h": np.random.uniform(20, 300, n),
        "area": np.random.uniform(400, 90000, n),
        "iscrowd": np.zeros(n, dtype=int),
        "image_width": np.repeat([640, 800, 1024, 800, 640], 10),
        "image_height": np.repeat([480, 600, 768, 600, 480], 10),
        "file_name": np.repeat(
            [f"img_{i:03d}.jpg" for i in range(1, 6)], 10
        ),
    })

    # 添加派生列
    df["aspect_ratio"] = df["bbox_w"] / df["bbox_h"]
    df["center_x"] = (df["bbox_x"] + df["bbox_w"] / 2) / df["image_width"]
    df["center_y"] = (df["bbox_y"] + df["bbox_h"] / 2) / df["image_height"]

    area = df["area"]
    conditions = [
        area <= 1024,
        (area > 1024) & (area <= 9216),
        area > 9216,
    ]
    choices = ["small", "medium", "large"]
    df["bbox_size_class"] = np.select(conditions, choices, default="unknown")
    df["normalized_area"] = df["area"] / (df["image_width"] * df["image_height"])
    df["log_area"] = np.log10(df["area"].clip(lower=1))
    df["is_square"] = (df["aspect_ratio"] > 0.8) & (df["aspect_ratio"] < 1.25)

    # 图像级统计
    df["objs_per_image"] = df.groupby("image_id")["annotation_id"].transform("count")
    df["cats_per_image"] = df.groupby("image_id")["category_id"].transform("nunique")

    return df


@pytest.fixture
def analysis_config() -> Dict[str, Any]:
    """返回测试用的分析配置。"""
    return {
        "bbox": {
            "small_area_max": 1024.0,
            "medium_area_max": 9216.0,
        },
        "cooccurrence": {
            "min_cooccurrence": 3,
            "network_edge_threshold": 5,
            "top_k_pairs": 10,
        },
        "long_tail": {
            "head_ratio": 0.20,
            "tail_ratio": 0.50,
        },
        "difficulty": {
            "weights": {
                "small_object_ratio": 0.35,
                "density": 0.25,
                "area_variance": 0.20,
                "aspect_ratio_variance": 0.10,
                "crowd_ratio": 0.10,
            },
        },
    }
