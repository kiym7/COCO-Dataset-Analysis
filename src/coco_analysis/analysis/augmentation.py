"""数据增强策略建议模块。

基于类别分布、边界框统计、共现关系和难度分析的结果，
生成针对性的数据增强策略建议，帮助提升目标检测模型的性能。
"""

from typing import Any, Dict, List

import pandas as pd


def recommend_mosaic_augmentation(
    difficulty_df: pd.DataFrame,
    small_ratio_threshold: float = 30.0,
) -> List[Dict[str, Any]]:
    """为小目标比例高的类别推荐 Mosaic 增强。

    Mosaic 增强将 4 张图像拼接为 1 张，可有效增加小目标的
    上下文多样性，对小目标检测特别有效。

    Args:
        difficulty_df: 检测难度 DataFrame（需包含 small_ratio 列）
        small_ratio_threshold: 小目标占比阈值 (%), 超过此值推荐 Mosaic

    Returns:
        推荐列表，每项包含 category_name, small_ratio, priority
    """
    if "small_ratio" not in difficulty_df.columns:
        return []

    mosaic_candidates = difficulty_df[
        difficulty_df["small_ratio"] > small_ratio_threshold
    ].copy()

    if mosaic_candidates.empty:
        return []

    mosaic_candidates["priority"] = pd.cut(
        mosaic_candidates["small_ratio"],
        bins=[small_ratio_threshold, 50, 70, 100],
        labels=["low", "medium", "high"],
    ).astype(str)

    return (
        mosaic_candidates[["category_name", "small_ratio", "priority"]]
        .sort_values("small_ratio", ascending=False)
        .to_dict("records")
    )


def recommend_copy_paste_augmentation(
    cooc_df: pd.DataFrame,
    difficulty_df: pd.DataFrame | None = None,
) -> Dict[str, Any]:
    """基于共现关系推荐 Copy-Paste 增强策略。

    Copy-Paste 增强将目标从一张图像复制粘贴到另一张图像，
    基于自然共现关系选择源和目标类别。

    Args:
        cooc_df: 共现矩阵
        difficulty_df: 检测难度分析结果（可选）

    Returns:
        字典，包含:
        - frequent_pairs: 基于频繁共现的推荐对
        - rare_pairs: 基于稀有共现的推荐对（增加多样性）
        - tail_categories_to_boost: 需要增加出现频率的 tail 类别
    """
    categories = cooc_df.index.tolist()
    n = len(categories)

    # 基于共现频率找增强候选
    pairs_above_diag = []
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue
            cooc_count = int(cooc_df.loc[cat1, cat2])
            cat1_total = int(cooc_df.loc[cat1, cat1])
            cat2_total = int(cooc_df.loc[cat2, cat2])
            pairs_above_diag.append({
                "cat1": cat1,
                "cat2": cat2,
                "cooccurrence": cooc_count,
                "cat1_total": cat1_total,
                "cat2_total": cat2_total,
                "cooc_ratio": (
                    cooc_count / min(cat1_total, cat2_total)
                    if min(cat1_total, cat2_total) > 0
                    else 0
                ),
            })

    # 高频共现对（这些是自然的共现场景）
    frequent = sorted(pairs_above_diag, key=lambda x: x["cooccurrence"], reverse=True)[:20]

    # 低共现但对模型泛化有意义的对
    rare = sorted(
        [p for p in pairs_above_diag if p["cooccurrence"] > 0],
        key=lambda x: x["cooccurrence"],
    )[:20]

    # Tail 类别推荐
    cat_totals = {cat: int(cooc_df.loc[cat, cat]) for cat in categories}
    median_count = pd.Series(cat_totals).median()
    tail_cats = [
        {"category": cat, "total": total}
        for cat, total in cat_totals.items()
        if total < median_count * 0.5
    ]
    tail_cats.sort(key=lambda x: x["total"])

    return {
        "frequent_pairs": frequent,
        "rare_pairs": rare,
        "tail_categories_to_boost": tail_cats,
    }


def recommend_class_balanced_sampling(
    freq_df: pd.DataFrame,
    head_tail: Dict[str, Any],
) -> Dict[str, Any]:
    """为类别不平衡问题推荐采样策略。

    Args:
        freq_df: category_frequency() 的输出
        head_tail: compute_head_tail_split() 的输出

    Returns:
        字典，包含:
        - oversample_categories: 需要过采样的 tail 类别
        - undersample_categories: 可能从欠采样中受益的 head 类别
        - sampling_ratio: 建议的过采样倍率
    """
    tail_cats = head_tail.get("tail_categories", [])
    head_cats = head_tail.get("head_categories", [])

    # 计算 tail 类别需要多少过采样才能达到中位数
    median_count = freq_df["count"].median()
    oversample_recs = []
    for _, row in freq_df[freq_df["category_name"].isin(tail_cats)].iterrows():
        ratio = max(1.0, median_count / row["count"])
        oversample_recs.append({
            "category": row["category_name"],
            "current_count": int(row["count"]),
            "target_count": int(median_count),
            "suggested_oversample_ratio": round(ratio, 1),
        })

    return {
        "oversample_categories": sorted(
            oversample_recs, key=lambda x: x["current_count"]
        ),
        "undersample_categories": head_cats,
        "median_instance_count": int(median_count),
    }


def generate_albumentations_config(
    difficulty_df: pd.DataFrame,
    category_df: pd.DataFrame,
) -> Dict[str, Any]:
    """生成可直接用于 Albumentations 库的增强配置建议。

    Args:
        difficulty_df: 检测难度分析结果
        category_df: 类别频率分析结果

    Returns:
        字典格式的增强配置建议
    """
    config = {
        "description": "基于 COCO 2017 分析结果的推荐增强配置",
        "transforms": [],
    }

    # 检查是否需要 Mosaic (小目标比例高)
    small_ratios = difficulty_df.get("small_ratio", pd.Series())
    if len(small_ratios) > 0 and small_ratios.mean() > 20:
        config["transforms"].append({
            "name": "Mosaic",
            "params": {"output_size": [640, 640], "probability": 0.5},
            "reason": f"小目标平均占比 {small_ratios.mean():.1f}%，Mosaic 可增加小目标上下文",
        })

    # 通用几何增强
    config["transforms"].extend([
        {
            "name": "HorizontalFlip",
            "params": {"p": 0.5},
            "reason": "大多数自然场景具有水平对称性",
        },
        {
            "name": "RandomScale",
            "params": {"scale_limit": 0.3, "p": 0.5},
            "reason": "COCO 数据集中目标尺度变化大，需要尺度增强",
        },
        {
            "name": "RandomBrightnessContrast",
            "params": {"brightness_limit": 0.2, "contrast_limit": 0.2, "p": 0.3},
            "reason": "模拟不同光照条件",
        },
    ])

    # 对极端宽高比的目标推荐 RandomRotate90
    if "ar_std" in difficulty_df.columns and difficulty_df["ar_std"].mean() > 0.5:
        config["transforms"].append({
            "name": "Rotate",
            "params": {"limit": 15, "p": 0.3},
            "reason": "宽高比方差较大，旋转增强可提高对非标准角度的鲁棒性",
        })

    return config


def summarize_recommendations(
    category_summary: Dict[str, Any],
    difficulty_summary: Dict[str, Any],
    cooccurrence_summary: Dict[str, Any],
) -> str:
    """生成人类可读的增强策略建议摘要。

    Args:
        category_summary: 类别分析结果
        difficulty_summary: 难度分析结果
        cooccurrence_summary: 共现分析结果

    Returns:
        Markdown 格式的增强策略建议文本
    """
    lines = [
        "# 数据增强策略建议",
        "",
        "基于对 COCO 2017 数据集的全面分析，以下为针对性的数据增强策略建议。",
        "",
    ]

    # 1. 类别不平衡
    gini = category_summary.get("gini_coefficient", 0)
    head_tail = category_summary.get("head_tail", {})
    tail_count = head_tail.get("tail_count", 0)

    lines.append("## 1. 类别不平衡处理")
    lines.append("")
    lines.append(f"- **Gini 系数**: {gini:.3f} (越高越不均匀)")
    lines.append(f"- **Tail 类别数量**: {tail_count} 个")
    lines.append(f"- **建议**: 对 Tail 类别使用 RepeatFactorDataset 或过采样策略")
    lines.append(f"- **建议**: 对 Head 类别使用类别平衡采样，减少对主导类别的偏向")
    lines.append("")

    # 2. 小目标增强
    hardest = difficulty_summary.get("hardest_categories", [])
    if hardest:
        lines.append("## 2. 小目标检测增强")
        lines.append("")
        lines.append("检测难度最高的类别 (前5):")
        for cat in hardest[:5]:
            lines.append(
                f"  - **{cat['category_name']}**: "
                f"难度得分 {cat['difficulty_score']:.1f}"
            )
        lines.append("")
        lines.append("- **建议**: 对以上类别启用 Mosaic 增强和 Multi-Scale Training")
        lines.append("- **建议**: 在高分辨率输入 (如 1024×1024) 上微调小目标检测器")
        lines.append("")

    # 3. 共现增强
    top_pairs = cooccurrence_summary.get("top_pairs", [])
    if top_pairs:
        lines.append("## 3. 共现关系利用")
        lines.append("")
        lines.append("最频繁共现的类别对 (前5):")
        for pair in top_pairs[:5]:
            lines.append(
                f"  - **{pair['cat1']}** ↔ **{pair['cat2']}**: "
                f"共现 {pair['cooccurrence_count']} 张图像"
            )
        lines.append("")
        lines.append("- **建议**: 使用 Copy-Paste 增强在训练图像中还原自然的共现模式")
        lines.append("- **建议**: 对稀有共现对进行人工增强，提高模型对罕见组合的泛化能力")
        lines.append("")

    # 4. 通用建议
    lines.append("## 4. 通用训练策略")
    lines.append("")
    lines.append("- 使用 **Multi-Scale Training** (输入尺寸 480-960)")
    lines.append("- 启用 **MixUp** 和 **CutMix** 增强")
    lines.append("- 对验证集保持原始分布，仅增强训练集")
    lines.append("- 使用 **EMA** (指数移动平均) 提升模型稳定性")
    lines.append("")

    return "\n".join(lines)
