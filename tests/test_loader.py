"""测试 COCO 数据加载器。"""

import pytest
from pycocotools.coco import COCO

from src.coco_analysis.data.loader import (
    load_coco,
    coco_to_dataframe,
    get_category_df,
    get_image_df,
    get_dataset_summary,
    build_combined_df,
)


class TestLoadCoco:
    """测试 COCO 加载函数。"""

    def test_load_valid_file(self, synthetic_coco_path):
        """测试加载有效的 COCO JSON 文件。"""
        coco = load_coco(str(synthetic_coco_path))
        assert isinstance(coco, COCO)

    def test_load_nonexistent_file(self):
        """测试加载不存在的文件。"""
        with pytest.raises(FileNotFoundError):
            load_coco("nonexistent_file.json")


class TestCategoryDf:
    """测试类别 DataFrame 提取。"""

    def test_get_category_df(self, synthetic_coco_path):
        """测试类别 DataFrame 结构。"""
        coco = load_coco(str(synthetic_coco_path))
        cat_df = get_category_df(coco)

        assert len(cat_df) == 5  # 合成数据有 5 个类别
        assert "category_id" in cat_df.columns
        assert "category_name" in cat_df.columns
        assert "supercategory" in cat_df.columns
        assert "person" in cat_df["category_name"].values


class TestImageDf:
    """测试图像 DataFrame 提取。"""

    def test_get_image_df(self, synthetic_coco_path):
        """测试图像 DataFrame 结构。"""
        coco = load_coco(str(synthetic_coco_path))
        img_df = get_image_df(coco)

        assert len(img_df) == 3  # 合成数据有 3 张图像
        assert "image_id" in img_df.columns
        assert "file_name" in img_df.columns
        assert "image_width" in img_df.columns
        assert "image_height" in img_df.columns


class TestCocoToDataframe:
    """测试主数据转换函数。"""

    def test_basic_conversion(self, synthetic_coco_path):
        """测试基本转换。"""
        coco = load_coco(str(synthetic_coco_path))
        df = coco_to_dataframe(coco)

        assert len(df) == 10  # 合成数据有 10 个标注
        assert "bbox_x" in df.columns
        assert "bbox_y" in df.columns
        assert "bbox_w" in df.columns
        assert "bbox_h" in df.columns
        assert "category_name" in df.columns
        assert "image_width" in df.columns

    def test_build_combined_df(self, synthetic_coco_path):
        """测试 build_combined_df 等同于 coco_to_dataframe。"""
        coco = load_coco(str(synthetic_coco_path))
        df1 = coco_to_dataframe(coco)
        df2 = build_combined_df(coco)
        assert len(df1) == len(df2)


class TestDatasetSummary:
    """测试数据集摘要函数。"""

    def test_get_dataset_summary(self, synthetic_coco_path):
        """测试摘要统计。"""
        coco = load_coco(str(synthetic_coco_path))
        summary = get_dataset_summary(coco)

        assert summary["num_images"] == 3
        assert summary["num_annotations"] == 10
        assert summary["num_categories"] == 5
        assert "person" in summary["category_names"]
