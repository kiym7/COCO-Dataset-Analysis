"""测试检测难度量化模块。"""

import numpy as np
import pytest

from src.coco_analysis.analysis.difficulty import (
    small_object_ratio,
    compute_objects_per_image_distribution,
    compute_crowd_stats,
    compute_difficulty_score,
    rank_categories_by_difficulty,
    generate_difficulty_summary,
)


class TestSmallObjectRatio:
    """测试小目标比例计算。"""

    def test_ratio(self, sample_df):
        """测试比例计算。"""
        result = small_object_ratio(sample_df)
        assert "category_name" in result.columns
        assert "small_ratio" in result.columns
        assert "total_count" in result.columns
        assert "small_count" in result.columns
        assert result["small_ratio"].max() <= 100
        assert result["small_ratio"].min() >= 0
        assert len(result) == sample_df["category_name"].nunique()


class TestObjectsPerImage:
    """测试每图目标数分布。"""

    def test_distribution(self, sample_df):
        """测试分布统计。"""
        result = compute_objects_per_image_distribution(sample_df)
        assert "mean" in result.index
        assert "std" in result.index
        assert "50%" in result.index


class TestCrowdStats:
    """测试 Crowd 统计。"""

    def test_no_crowd(self, sample_df):
        """测试无 crowd 标注的数据。"""
        # sample_df 中没有 iscrowd=1
        result = compute_crowd_stats(sample_df)
        if "iscrowd" in sample_df.columns:
            assert result["crowd_ratio"] == 0.0


class TestDifficultyScore:
    """测试难度评分。"""

    def test_by_category(self, sample_df):
        """测试按类别评分。"""
        result = compute_difficulty_score(sample_df, by="category")
        assert "category_name" in result.columns
        assert "difficulty_score" in result.columns
        assert result["difficulty_score"].max() <= 100
        assert result["difficulty_score"].min() >= 0

    def test_by_image(self, sample_df):
        """测试按图像评分。"""
        result = compute_difficulty_score(sample_df, by="image")
        assert "image_id" in result.columns
        assert "difficulty_score" in result.columns


class TestRankCategories:
    """测试难度排名。"""

    def test_ranking(self, sample_df):
        """测试排名输出。"""
        result = rank_categories_by_difficulty(sample_df)
        assert len(result) == sample_df["category_name"].nunique()


class TestDifficultySummary:
    """测试难度汇总。"""

    def test_summary(self, sample_df, analysis_config):
        """测试完整汇总。"""
        summary = generate_difficulty_summary(sample_df, analysis_config)
        assert "small_object_by_category" in summary
        assert "objects_per_image_stats" in summary
        assert "crowd_stats" in summary
        assert "difficulty_by_category" in summary
        assert "hardest_categories" in summary
        assert "easiest_categories" in summary
