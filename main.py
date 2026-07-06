# -*- coding: utf-8 -*-
"""
COCO 2017 数据集深度分析 - 一键运行入口

在 PyCharm 里直接右键这个文件 -> Run 'main'，或者在终端执行:
    python main.py

程序会自动:
  1. 查找 data/coco/annotations/ 下的标注文件
  2. 如果没找到，提示运行 download_coco.py
  3. 执行完整分析并生成报告
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.coco_analysis.config import get_config
from src.coco_analysis.data.loader import load_coco, coco_to_dataframe, get_dataset_summary
from src.coco_analysis.data.preprocessor import pipeline as preprocess_pipeline
from src.coco_analysis.analysis.category import generate_category_summary
from src.coco_analysis.analysis.bbox import generate_bbox_summary
from src.coco_analysis.analysis.cooccurrence import generate_cooccurrence_summary
from src.coco_analysis.analysis.difficulty import generate_difficulty_summary
from src.coco_analysis.analysis.augmentation import (
    recommend_mosaic_augmentation,
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


def find_annotation_file():
    """Auto-find COCO annotation file."""
    search_dirs = [
        Path("data/coco/annotations"),
        Path("data/coco"),
    ]
    search_names = [
        "instances_val2017.json",
        "instances_train2017.json",
    ]

    for search_dir in search_dirs:
        if search_dir.exists():
            for name in search_names:
                candidate = search_dir / name
                if candidate.exists():
                    return candidate

    return None


def main():
    print("=" * 60)
    print("  COCO 2017 Dataset Deep Analysis")
    print("=" * 60)
    print()

    # 1. Find annotation file
    annotation_path = find_annotation_file()
    if annotation_path is None:
        print("[ERROR] COCO annotation file not found!")
        print()
        print("Please run the download script first:")
        print("  In PyCharm: right-click scripts/download_coco.py -> Run")
        print("  Or: python scripts/download_coco.py")
        print()
        print("After download, run this file again.")
        input("Press Enter to exit...")
        return

    print(f"[INFO] Found annotation file: {annotation_path.name}")
    print(f"       Path: {annotation_path}")

    # 2. Initialize
    config = get_config()
    figure_dir = Path("outputs/figures")
    report_dir = Path("outputs/reports")
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    initialize_visualization(config.visualization)

    # 3. Load data
    print()
    print("[1/6] Loading data...")
    t0 = time.time()
    coco = load_coco(str(annotation_path))
    df_raw = coco_to_dataframe(coco)
    ds_summary = get_dataset_summary(coco)
    df = preprocess_pipeline(df_raw, config=config.analysis)
    print(f"      Images: {ds_summary['num_images']:,}")
    print(f"      Annotations: {ds_summary['num_annotations']:,}")
    print(f"      Categories: {ds_summary['num_categories']}")

    # 4. Category analysis
    print()
    print("[2/6] Category distribution & long-tail analysis...")
    cat_results = generate_category_summary(df, config=config.analysis)
    plot_category_distribution(cat_results["category_frequency"], output_dir=str(figure_dir))
    plot_long_tail_curve(
        cat_results["category_frequency"], output_dir=str(figure_dir),
        head_ratio=0.20, tail_ratio=0.50, gini=cat_results["gini_coefficient"],
    )
    plot_category_coverage(df, output_dir=str(figure_dir))
    print(f"      Gini coefficient: {cat_results['gini_coefficient']:.3f}")
    ht = cat_results["head_tail"]
    print(f"      Head/Body/Tail: {ht['head_count']}/{ht['body_count']}/{ht['tail_count']} categories")

    # 5. Bbox analysis
    print()
    print("[3/6] Bounding box statistics...")
    bbox_results = generate_bbox_summary(df, config=config.analysis)
    plot_bbox_size_pie(bbox_results["size_classification"], output_dir=str(figure_dir))
    plot_aspect_ratio_distribution(df, output_dir=str(figure_dir))
    plot_position_heatmap(df, output_dir=str(figure_dir))
    plot_bbox_area_violin(df, output_dir=str(figure_dir))
    for _, row in bbox_results["size_classification"].iterrows():
        print(f"      {row['bbox_size_class']:>8s}: {int(row['count']):>8,} ({row['percentage']:.1f}%)")

    # 6. Co-occurrence analysis
    print()
    print("[4/6] Object co-occurrence network...")
    cooc_results = generate_cooccurrence_summary(df, config=config.analysis)
    top_pairs = cooc_results.get("top_pairs", [])
    for pair in top_pairs[:5]:
        print(f"      {pair['cat1']:20s} <-> {pair['cat2']:20s}: {pair['cooccurrence_count']:>6,} images")

    communities = cooc_results.get("communities")
    if communities:
        print(f"      Communities detected: {communities['num_communities']} (modularity: {communities['modularity']:.3f})")

    # 7. Difficulty analysis
    print()
    print("[5/6] Detection difficulty quantification...")
    diff_results = generate_difficulty_summary(df, config=config.analysis)
    plot_small_object_scatter(df, output_dir=str(figure_dir))
    plot_objects_per_image_histogram(df, output_dir=str(figure_dir))
    plot_difficulty_ranking(diff_results["difficulty_by_category"], output_dir=str(figure_dir))
    hardest = diff_results.get("hardest_categories", [])
    if hardest:
        print(f"      Hardest to detect: {hardest[0]['category_name']} (score: {hardest[0]['difficulty_score']:.1f})")

    # 8. Augmentation & Report
    print()
    print("[6/6] Generating augmentation strategies & report...")
    mosaic_recs = recommend_mosaic_augmentation(diff_results["difficulty_by_category"])
    sampling_recs = recommend_class_balanced_sampling(
        cat_results["category_frequency"], cat_results["head_tail"]
    )
    aug_text = summarize_recommendations(cat_results, diff_results, cooc_results)
    plot_augmentation_recommendations(
        mosaic_recs, sampling_recs.get("oversample_categories", []),
        output_dir=str(figure_dir),
    )

    # Collect all results
    all_results = {
        "dataset_summary": ds_summary,
        "category": cat_results,
        "bbox": bbox_results,
        "cooccurrence": cooc_results,
        "difficulty": diff_results,
        "augmentation_text": aug_text,
    }

    figure_paths = {}
    for f in figure_dir.glob("*.png"):
        figure_paths[f.stem] = str(f)

    md_report = generate_markdown_report(all_results, figure_paths, lang="zh")
    md_path = report_dir / "analysis_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)

    html_path = generate_html_report(md_report, figure_paths, str(report_dir / "analysis_report.html"))
    print(f"      Markdown report: {md_path}")
    print(f"      HTML report: {html_path}")

    t_total = time.time() - t0
    print()
    print("=" * 60)
    print(f"  Analysis complete! (total time: {t_total:.1f}s)")
    print("=" * 60)
    print()
    print(f"Charts saved to: {figure_dir}/")
    for f in sorted(figure_dir.glob("*.png")):
        print(f"  - {f.name}")
    print()
    print(f"Reports saved to: {report_dir}/")
    print(f"  Open outputs/reports/analysis_report.html in browser to view.")


if __name__ == "__main__":
    main()
