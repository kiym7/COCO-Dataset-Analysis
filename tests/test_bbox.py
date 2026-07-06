"""测试边界框分析模块。"""

import numpy as np
import pytest

from src.coco_analysis.analysis.bbox import (
    bbox_size_classification,
    bbox_area_distribution,
    aspect_ratio_stats,
    position_heatmap_data,
    position_bias_analysis,
    per_category_bbox_stats,
    generate_bbox_summary,
)


class TestBboxSizeClassification:
    """测试尺寸分类统计。"""

    def test_classification(self, sample_df):
        """测试基本分类统计。"""
        result = bbox_size_classification(sample_df)
        assert len(result) == 3
        assert "bbox_size_class" in result.columns
        assert "count" in result.columns
        assert "percentage" in result.columns
        assert result["percentage"].sum() == pytest.approx(100.0, rel=1e-9)

    def test_order(self, sample_df):
        """测试顺序: small -> medium -> large。"""
        result = bbox_size_classification(sample_df)
        assert list(result["bbox_size_class"]) == ["small", "medium", "large"]


class TestBboxAreaDistribution:
    """测试面积分布统计。"""

    def test_distribution(self, sample_df):
        """测试基本面积统计。"""
        result = bbox_area_distribution(sample_df)
        assert "supercategory" in result.columns
        assert "mean_area" in result.columns
        assert "median_area" in result.columns
        assert "std_area" in result.columns


class TestAspectRatioStats:
    """测试宽高比统计。"""

    def test_stats(self, sample_df):
        """测试宽高比统计结构。"""
        result = aspect_ratio_stats(sample_df)
        assert "overall_stats" in result
        assert "by_supercategory" in result
        assert "extreme_categories" in result
        assert "mean" in result["overall_stats"]
        assert "median" in result["overall_stats"]


class TestPositionHeatmap:
    """测试位置热力图数据生成。"""

    def test_heatmap_data(self, sample_df):
        """测试热力图矩阵。"""
        heatmap = position_heatmap_data(sample_df, grid_size=10)
        assert heatmap.shape == (10, 10)
        assert heatmap.sum() == len(sample_df)


class TestPositionBias:
    """测试位置偏置分析。"""

    def test_position_bias(self, sample_df):
        """测试偏置分析输出。"""
        result = position_bias_analysis(sample_df)
        assert "center_x_mean" in result
        assert "center_y_mean" in result
        assert "center_bias_description" in result
        assert "quadrant_distribution" in result
        # 象限分布之和应等于总实例数
        quad_sum = sum(result["quadrant_distribution"].values())
        assert quad_sum == len(sample_df)


class TestPerCategoryBboxStats:
    """测试每类别 bbox 统计。"""

    def test_stats(self, sample_df):
        """测试统计输出。"""
        result = per_category_bbox_stats(sample_df)
        assert "category_name" in result.columns
        assert "mean_area" in result.columns
        assert "mean_aspect_ratio" in result.columns
        assert "small_ratio" in result.columns
        assert len(result) == sample_df["category_name"].nunique()


class TestBboxSummary:
    """测试 bbox 汇总。"""

    def test_summary(self, sample_df, analysis_config):
        """测试完整汇总。"""
        summary = generate_bbox_summary(sample_df, analysis_config)
        assert "size_classification" in summary
        assert "area_distribution" in summary
        assert "aspect_ratio" in summary
        assert "position_bias" in summary
        assert "per_category_stats" in summary
        assert "heatmap_data" in summary
