#!/usr/bin/env python
"""COCO 2017 数据分析一键运行脚本。

串联数据加载 → 分析 → 可视化 → 报告生成的完整流水线。

用法:
    python scripts/run_analysis.py --annotations data/coco/annotations/instances_train2017.json
    python scripts/run_analysis.py --annotations <path> --output outputs/ --no-interactive
"""

import argparse
import sys
import time
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coco_analysis.config import get_config
from src.coco_analysis.data.loader import load_coco, coco_to_dataframe, get_dataset_summary
from src.coco_analysis.data.preprocessor import pipeline as preprocess_pipeline
from src.coco_analysis.analysis.category import generate_category_summary
from src.coco_analysis.analysis.bbox import generate_bbox_summary
from src.coco_analysis.analysis.cooccurrence import generate_cooccurrence_summary
from src.coco_analysis.analysis.difficulty import generate_difficulty_summary
from src.coco_analysis.analysis.augmentation import (
    recommend_mosaic_augmentation,
    recommend_copy_paste_augmentation,
    recommend_class_balanced_sampling,
    summarize_recommendations,
)
from src.coco_analysis.visualization.theme import initialize_visualization
from src.coco_analysis.visualization.category_plots import (
    plot_category_distribution,
    plot_long_tail_curve,
    plot_category_coverage,
)
from src.coco_analysis.visualization.bbox_plots import (
    plot_bbox_size_pie,
    plot_aspect_ratio_distribution,
    plot_position_heatmap,
    plot_bbox_area_violin,
)
from src.coco_analysis.visualization.difficulty_plots import (
    plot_small_object_scatter,
    plot_objects_per_image_histogram,
    plot_difficulty_ranking,
    plot_augmentation_recommendations,
)
from src.coco_analysis.report.generator import (
    generate_markdown_report,
    generate_html_report,
)


def main():
    parser = argparse.ArgumentParser(
        description="COCO 2017 数据集深度分析 — 一键运行",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_analysis.py --annotations data/coco/annotations/instances_train2017.json
  python run_analysis.py -a data/coco/annotations/instances_val2017.json -o my_outputs
  python run_analysis.py -a <path> --no-interactive --format md
        """,
    )
    parser.add_argument(
        "--annotations", "-a",
        type=str,
        required=True,
        help="COCO 标注 JSON 文件路径",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs",
        help="输出根目录 (默认: outputs)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="配置文件路径 (默认: config.yaml)",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="跳过交互式图表生成 (加速)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["md", "html", "both"],
        default="both",
        help="报告格式 (默认: both)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细日志输出",
    )

    args = parser.parse_args()

    # ========== 初始化 ==========
    print("=" * 60)
    print("  COCO 2017 数据集深度分析")
    print("=" * 60)

    if not Path(args.annotations).exists():
        print(f"\n❌ 错误: 标注文件不存在: {args.annotations}")
        print("  请先运行: python scripts/download_coco.py")
        sys.exit(1)

    config = get_config(args.config)
    figure_dir = str(Path(args.output) / "figures")
    report_dir = str(Path(args.output) / "reports")

    # 创建输出目录
    Path(figure_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)

    # 初始化可视化（中文字体等）
    font_name = initialize_visualization(config.visualization)
    if args.verbose:
        print(f"  使用字体: {font_name}")

    # ========== 阶段 1: 数据加载 ==========
    print("\n[1/6] 加载 COCO 标注数据...")
    t0 = time.time()

    coco = load_coco(args.annotations)
    df_raw = coco_to_dataframe(coco)
    ds_summary = get_dataset_summary(coco)

    if args.verbose:
        print(f"  图像数: {ds_summary['num_images']:,}")
        print(f"  标注数: {ds_summary['num_annotations']:,}")
        print(f"  类别数: {ds_summary['num_categories']}")

    # 预处理
    print("  预处理数据 (过滤 + 派生列计算)...")
    df = preprocess_pipeline(df_raw, config=config.analysis)
    t1 = time.time()
    print(f"  ✓ 完成 ({t1 - t0:.1f}s)")

    # ========== 阶段 2: 类别分析 ==========
    print("\n[2/6] 类别分布与长尾分析...")
    category_results = generate_category_summary(df, config=config.analysis)

    plot_category_distribution(
        category_results["category_frequency"],
        output_dir=figure_dir,
    )
    plot_long_tail_curve(
        category_results["category_frequency"],
        head_ratio=config.analysis.get("long_tail", {}).get("head_ratio", 0.20),
        tail_ratio=config.analysis.get("long_tail", {}).get("tail_ratio", 0.50),
        gini=category_results["gini_coefficient"],
        output_dir=figure_dir,
    )
    plot_category_coverage(df, output_dir=figure_dir)

    print(f"  Gini 系数: {category_results['gini_coefficient']:.3f}")
    head_tail = category_results["head_tail"]
    print(f"  Head/Body/Tail: {head_tail['head_count']}/{head_tail['body_count']}/{head_tail['tail_count']}")

    # ========== 阶段 3: 边界框分析 ==========
    print("\n[3/6] 边界框统计分析...")
    bbox_results = generate_bbox_summary(df, config=config.analysis)

    plot_bbox_size_pie(bbox_results["size_classification"], output_dir=figure_dir)
    plot_aspect_ratio_distribution(df, output_dir=figure_dir)
    plot_position_heatmap(df, output_dir=figure_dir)
    plot_bbox_area_violin(df, output_dir=figure_dir)

    size_cls = bbox_results["size_classification"]
    if not size_cls.empty:
        for _, row in size_cls.iterrows():
            print(f"  {row['bbox_size_class']}: {int(row['count']):,} ({row['percentage']:.1f}%)")

    # ========== 阶段 4: 共现分析 ==========
    print("\n[4/6] 目标共现网络分析...")
    cooccurrence_results = generate_cooccurrence_summary(df, config=config.analysis)

    # 共现图表在 notebook 中展示 (networkx plotly)
    top_pairs = cooccurrence_results.get("top_pairs", [])
    if top_pairs:
        print(f"  Top 3 共现对:")
        for pair in top_pairs[:3]:
            print(f"    {pair['cat1']} ↔ {pair['cat2']}: {pair['cooccurrence_count']:,} 张图像")

    communities = cooccurrence_results.get("communities")
    if communities:
        print(f"  检测到 {communities['num_communities']} 个社区 (模块度: {communities['modularity']:.3f})")

    # ========== 阶段 5: 难度分析 ==========
    print("\n[5/6] 检测难度量化分析...")
    difficulty_results = generate_difficulty_summary(df, config=config.analysis)

    plot_small_object_scatter(df, output_dir=figure_dir)
    plot_objects_per_image_histogram(df, output_dir=figure_dir)
    plot_difficulty_ranking(difficulty_results["difficulty_by_category"], output_dir=figure_dir)

    hardest = difficulty_results.get("hardest_categories", [])
    if hardest:
        print(f"  最难检测的类别: {hardest[0]['category_name']} (得分: {hardest[0]['difficulty_score']:.1f})")

    # ========== 阶段 6: 增强建议 + 报告 ==========
    print("\n[6/6] 生成增强策略与报告...")

    # 增强策略
    mosaic_recs = recommend_mosaic_augmentation(difficulty_results["difficulty_by_category"])
    sampling_recs = recommend_class_balanced_sampling(
        category_results["category_frequency"],
        category_results["head_tail"],
    )
    aug_text = summarize_recommendations(
        category_results, difficulty_results, cooccurrence_results
    )

    # 增强策略图
    plot_augmentation_recommendations(
        mosaic_recs,
        sampling_recs.get("oversample_categories", []),
        output_dir=figure_dir,
    )

    # 整合所有结果
    all_results = {
        "dataset_summary": ds_summary,
        "category": category_results,
        "bbox": bbox_results,
        "cooccurrence": cooccurrence_results,
        "difficulty": difficulty_results,
        "augmentation_text": aug_text,
    }

    # 收集图表路径
    figure_paths = {}
    for f in Path(figure_dir).glob("*.png"):
        figure_paths[f.stem] = str(f)
    for f in Path(figure_dir).glob("*.html"):
        figure_paths[f.stem] = str(f)

    # 生成 Markdown 报告
    if args.format in ("md", "both"):
        md_report = generate_markdown_report(all_results, figure_paths, lang="zh")
        md_path = Path(report_dir) / "analysis_report.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_report)
        print(f"  ✓ Markdown 报告: {md_path}")

    # 生成 HTML 报告
    if args.format in ("html", "both"):
        html_path = generate_html_report(
            md_report if args.format == "both" else generate_markdown_report(all_results, figure_paths),
            figure_paths,
            str(Path(report_dir) / "analysis_report.html"),
        )
        print(f"  ✓ HTML 报告: {html_path}")

    t_total = time.time() - t0

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print(f"  ✓ 分析完成! (总耗时: {t_total:.1f}s)")
    print("=" * 60)
    print(f"\n输出文件:")
    print(f"  图表: {figure_dir}/")
    for f in sorted(Path(figure_dir).glob("*.png")):
        print(f"    - {f.name}")
    print(f"  报告: {report_dir}/")
    for f in sorted(Path(report_dir).glob("*.*")):
        print(f"    - {f.name}")
    print(f"\n💡 提示: 使用 Jupyter Notebook 获得更好的交互体验:")
    print(f"   jupyter notebook notebooks/coco_analysis.ipynb")


if __name__ == "__main__":
    main()
