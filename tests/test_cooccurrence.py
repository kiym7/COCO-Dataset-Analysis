"""测试共现网络分析模块。"""

import numpy as np
import pandas as pd
import pytest

from src.coco_analysis.analysis.cooccurrence import (
    build_cooccurrence_matrix,
    build_cooccurrence_probability,
    compute_lift_matrix,
    find_top_cooccurring_pairs,
    generate_cooccurrence_summary,
)


class TestCooccurrenceMatrix:
    """测试共现矩阵构建。"""

    def test_matrix_shape(self, sample_df):
        """测试矩阵维度。"""
        cooc = build_cooccurrence_matrix(sample_df)
        n_cats = sample_df["category_name"].nunique()
        assert cooc.shape == (n_cats, n_cats)

    def test_matrix_symmetric(self, sample_df):
        """测试矩阵对称性。"""
        cooc = build_cooccurrence_matrix(sample_df)
        assert (cooc.values == cooc.values.T).all()

    def test_diagonal_equals_frequency(self, sample_df):
        """测试对角线值 = 类别出现的图像数。"""
        cooc = build_cooccurrence_matrix(sample_df)
        for cat in cooc.index:
            actual_images = sample_df[sample_df["category_name"] == cat]["image_id"].nunique()
            assert cooc.loc[cat, cat] == actual_images


class TestProbabilityMatrix:
    """测试条件概率矩阵。"""

    def test_probability_bounds(self, sample_df):
        """测试概率值在 [0, 1] 范围内。"""
        cooc = build_cooccurrence_matrix(sample_df)
        prob = build_cooccurrence_probability(cooc)
        assert prob.values.min() >= 0
        assert prob.values.max() <= 1.0

    def test_diagonal_is_one(self, sample_df):
        """测试对角线概率 = 1.0。"""
        cooc = build_cooccurrence_matrix(sample_df)
        prob = build_cooccurrence_probability(cooc)
        for cat in prob.index:
            assert prob.loc[cat, cat] == pytest.approx(1.0)


class TestLiftMatrix:
    """测试 Lift 值矩阵。"""

    def test_lift_diagonal_is_one(self, sample_df):
        """测试对角线 Lift = 1。"""
        cooc = build_cooccurrence_matrix(sample_df)
        lift = compute_lift_matrix(cooc)
        for cat in lift.index:
            assert lift.loc[cat, cat] == pytest.approx(1.0)


class TestTopPairs:
    """测试顶级共现对。"""

    def test_top_pairs_format(self, sample_df):
        """测试返回格式。"""
        cooc = build_cooccurrence_matrix(sample_df)
        pairs = find_top_cooccurring_pairs(cooc, top_k=10)

        assert isinstance(pairs, list)
        if pairs:  # 仅在存在共现对时检查
            pair = pairs[0]
            assert "cat1" in pair
            assert "cat2" in pair
            assert "cooccurrence_count" in pair
            assert "cat1_freq" in pair
            assert "cat2_freq" in pair

    def test_no_self_pairs(self, sample_df):
        """测试不包含自对 (cat1 == cat2)。"""
        cooc = build_cooccurrence_matrix(sample_df)
        pairs = find_top_cooccurring_pairs(cooc, top_k=50)
        for pair in pairs:
            assert pair["cat1"] != pair["cat2"]


class TestCooccurrenceSummary:
    """测试共现汇总。"""

    def test_summary(self, sample_df, analysis_config):
        """测试完整汇总。"""
        summary = generate_cooccurrence_summary(sample_df, analysis_config)
        assert "cooccurrence_matrix" in summary
        assert "probability_matrix" in summary
        assert "lift_matrix" in summary
        assert "top_pairs" in summary
