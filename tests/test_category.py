"""测试类别分布分析模块。"""

import numpy as np
import pandas as pd
import pytest

from src.coco_analysis.analysis.category import (
    category_frequency,
    compute_gini_coefficient,
    compute_head_tail_split,
    images_per_category,
    supercategory_distribution,
    long_tail_curve_data,
    generate_category_summary,
)


class TestCategoryFrequency:
    """测试类别频率计算。"""

    def test_basic_frequency(self, sample_df):
        """测试基本频率计算。"""
        freq = category_frequency(sample_df)
        assert len(freq) == 5  # 5 个类别
        assert "count" in freq.columns
        assert "percentage" in freq.columns
        assert "rank" in freq.columns
        assert freq["percentage"].sum() == pytest.approx(100.0, rel=1e-9)
        assert freq.loc[0, "rank"] == 1  # 第一个排名为 1


class TestGiniCoefficient:
    """测试 Gini 系数计算。"""

    def test_perfect_equality(self):
        """测试完全均匀分布。"""
        counts = np.array([10, 10, 10, 10])
        gini = compute_gini_coefficient(counts)
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_perfect_inequality(self):
        """测试完全不均匀分布。"""
        counts = np.array([100, 0, 0, 0])
        gini = compute_gini_coefficient(counts)
        assert gini == pytest.approx(0.75, abs=1e-1)  # 接近 1

    def test_single_category(self):
        """测试单个类别。"""
        gini = compute_gini_coefficient(np.array([5]))
        assert gini == 0.0


class TestHeadTailSplit:
    """测试 Head/Tail 划分。"""

    def test_split(self, sample_df):
        """测试基本划分。"""
        freq = category_frequency(sample_df)
        result = compute_head_tail_split(freq, head_ratio=0.50, tail_ratio=0.30)

        assert "head_categories" in result
        assert "body_categories" in result
        assert "tail_categories" in result
        assert result["head_count"] + result["body_count"] + result["tail_count"] == len(freq)


class TestImagesPerCategory:
    """测试每类别图像数。"""

    def test_images_per_category(self, sample_df):
        """测试图像计数的基本属性。"""
        result = images_per_category(sample_df)
        assert len(result) == 5
        assert "image_count" in result.columns
        assert "avg_instances_per_image" in result.columns
        assert result["image_count"].max() <= sample_df["image_id"].nunique()


class TestSupercategoryDistribution:
    """测试超类别分布。"""

    def test_supercategory_distribution(self, sample_df):
        """测试超类别统计。"""
        sc = supercategory_distribution(sample_df)
        assert "supercategory" in sc.columns
        assert "instance_count" in sc.columns
        assert "category_count" in sc.columns
        assert "categories" in sc.columns


class TestLongTailCurve:
    """测试长尾曲线数据。"""

    def test_curve_data(self, sample_df):
        """测试曲线数据结构。"""
        freq = category_frequency(sample_df)
        curve = long_tail_curve_data(freq)
        assert "rank" in curve.columns
        assert "count" in curve.columns
        assert "cum_pct" in curve.columns
        assert curve["cum_pct"].iloc[-1] == pytest.approx(100.0, abs=0.1)


class TestCategorySummary:
    """测试类别汇总。"""

    def test_generate_summary(self, sample_df, analysis_config):
        """测试完整汇总生成。"""
        summary = generate_category_summary(sample_df, analysis_config)
        assert "category_frequency" in summary
        assert "gini_coefficient" in summary
        assert "head_tail" in summary
        assert "supercategory_distribution" in summary
        assert "top5_categories" in summary
        assert "bottom5_categories" in summary
