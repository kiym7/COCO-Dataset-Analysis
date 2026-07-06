"""分析报告生成器。

整合所有分析结果和可视化图表，生成 Markdown 和 HTML 格式的完整分析报告。
"""

import base64
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_markdown_report(
    results: Dict[str, Any],
    figure_paths: Dict[str, str],
    lang: str = "zh",
) -> str:
    """生成完整的 Markdown 分析报告。

    Args:
        results: 包含所有分析结果的字典，包含键:
            - category: 类别分析结果
            - bbox: 边界框分析结果
            - cooccurrence: 共现分析结果
            - difficulty: 难度分析结果
            - augmentation: 增强策略结果
            - dataset_summary: 数据集概览
        figure_paths: {figure_key: file_path} 映射
        lang: 语言 ('zh' 或 'en')

    Returns:
        Markdown 格式的报告内容
    """
    zh = lang == "zh"

    lines = []
    if zh:
        lines.append("# COCO 2017 数据集深度分析报告")
        lines.append("")
        lines.append(f"**生成日期**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")
    else:
        lines.append("# COCO 2017 Dataset Deep Analysis Report")
        lines.append("")
        lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("")

    # 1. 执行摘要
    lines += _section_executive_summary(results, zh)
    lines += _section_dataset_overview(results, zh)
    lines += _section_category_analysis(results, figure_paths, zh)
    lines += _section_bbox_analysis(results, figure_paths, zh)
    lines += _section_cooccurrence_analysis(results, figure_paths, zh)
    lines += _section_difficulty_analysis(results, figure_paths, zh)
    lines += _section_augmentation(results, zh)
    lines += _section_appendix(results, zh)

    return "\n".join(lines)


def _section_executive_summary(results: Dict[str, Any], zh: bool) -> List[str]:
    """生成执行摘要章节。"""
    lines = []
    if zh:
        lines.append("## 📊 执行摘要")
        lines.append("")
    else:
        lines.append("## 📊 Executive Summary")
        lines.append("")

    ds = results.get("dataset_summary", {})
    cat = results.get("category", {})
    diff = results.get("difficulty", {})

    if zh:
        lines.append(f"- **数据集**: COCO 2017, 共 {ds.get('num_images', 'N/A'):,} 张训练图像")
        lines.append(f"- **标注实例**: {ds.get('num_annotations', 'N/A'):,} 个")
        lines.append(f"- **目标类别**: {ds.get('num_categories', 'N/A')} 个")
        lines.append("")

        gini = cat.get("gini_coefficient", 0)
        lines.append(f"- **类别分布 Gini 系数**: {gini:.3f}")
        lines.append(f"  - Head 类别 ({cat.get('head_tail', {}).get('head_count', 'N/A')} 个) 贡献了 {cat.get('head_tail', {}).get('head_total_pct', 0):.1f}% 的实例")
        lines.append(f"  - Tail 类别 ({cat.get('head_tail', {}).get('tail_count', 'N/A')} 个) 仅贡献了 {cat.get('head_tail', {}).get('tail_total_pct', 0):.1f}% 的实例")

        bbox = results.get("bbox", {})
        size_cls = bbox.get("size_classification", None)
        if size_cls is not None and not size_cls.empty:
            small_row = size_cls[size_cls["bbox_size_class"] == "small"]
            large_row = size_cls[size_cls["bbox_size_class"] == "large"]
            if not small_row.empty:
                lines.append(f"- **小目标占比** (≤32² px): {small_row.iloc[0]['percentage']:.1f}%")
            if not large_row.empty:
                lines.append(f"- **大目标占比** (>96² px): {large_row.iloc[0]['percentage']:.1f}%")

        hardest = diff.get("hardest_categories", [])
        if hardest:
            lines.append(f"- **检测最难的类别**: {hardest[0]['category_name']} (难度得分: {hardest[0]['difficulty_score']:.1f})")
        lines.append("")
    else:
        lines.append(f"- **Dataset**: COCO 2017, {ds.get('num_images', 'N/A'):,} training images")
        lines.append(f"- **Annotations**: {ds.get('num_annotations', 'N/A'):,}")
        lines.append(f"- **Categories**: {ds.get('num_categories', 'N/A')}")
        lines.append("")

    lines.append("---")
    lines.append("")
    return lines


def _section_dataset_overview(results: Dict[str, Any], zh: bool) -> List[str]:
    """生成数据集概览章节。"""
    lines = []
    if zh:
        lines.append("## 1. 数据集概览")
    else:
        lines.append("## 1. Dataset Overview")
    lines.append("")

    ds = results.get("dataset_summary", {})
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")
    if zh:
        lines.append(f"| 图像总数 | {ds.get('num_images', 'N/A'):,} |")
        lines.append(f"| 标注总数 | {ds.get('num_annotations', 'N/A'):,} |")
        lines.append(f"| 类别总数 | {ds.get('num_categories', 'N/A')} |")
        lines.append(f"| 超类别数 | {len(ds.get('supercategories', []))} |")
        lines.append(f"| 每图平均目标数 | {ds.get('num_annotations', 0) / max(ds.get('num_images', 1), 1):.1f} |")
    else:
        lines.append(f"| Total Images | {ds.get('num_images', 'N/A'):,} |")
        lines.append(f"| Total Annotations | {ds.get('num_annotations', 'N/A'):,} |")
        lines.append(f"| Total Categories | {ds.get('num_categories', 'N/A')} |")
        lines.append(f"| Supercategories | {len(ds.get('supercategories', []))} |")
        lines.append(f"| Avg Objects/Image | {ds.get('num_annotations', 0) / max(ds.get('num_images', 1), 1):.1f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _section_category_analysis(
    results: Dict[str, Any], figure_paths: Dict[str, str], zh: bool
) -> List[str]:
    """生成类别分析章节。"""
    cat = results.get("category", {})

    lines = []
    if zh:
        lines.append("## 2. 类别分布与长尾分析")
    else:
        lines.append("## 2. Category Distribution & Long-tail Analysis")
    lines.append("")

    gini = cat.get("gini_coefficient", 0)
    head_tail = cat.get("head_tail", {})

    if zh:
        lines.append(f"**Gini 系数**: {gini:.3f} ({'高度' if gini > 0.5 else '中度' if gini > 0.3 else '低度'}不平衡)")
        lines.append("")
        lines.append(f"### Head-Body-Tail 划分")
        lines.append(f"- **Head** ({head_tail.get('head_count', 0)} 个类别): 贡献 {head_tail.get('head_total_pct', 0):.1f}% 实例")
        lines.append(f"- **Body** ({head_tail.get('body_count', 0)} 个类别): 中等频率")
        lines.append(f"- **Tail** ({head_tail.get('tail_count', 0)} 个类别): 仅贡献 {head_tail.get('tail_total_pct', 0):.1f}% 实例")
    else:
        lines.append(f"**Gini Coefficient**: {gini:.3f}")
        lines.append("")

    # Top-5 & Bottom-5
    top5 = cat.get("top5_categories", [])
    if top5:
        if zh:
            lines.append("### Top-5 高频类别")
        else:
            lines.append("### Top-5 Most Frequent Categories")
        lines.append("")
        lines.append("| 类别 | 实例数 | 占比 |")
        lines.append("|------|--------|------|")
        for item in top5:
            lines.append(f"| {item['category_name']} | {item['count']:,} | {item['percentage']:.1f}% |")
        lines.append("")

    bottom5 = cat.get("bottom5_categories", [])
    if bottom5:
        if zh:
            lines.append("### Bottom-5 低频类别")
        else:
            lines.append("### Bottom-5 Least Frequent Categories")
        lines.append("")
        lines.append("| 类别 | 实例数 | 占比 |")
        lines.append("|------|--------|------|")
        for item in bottom5:
            lines.append(f"| {item['category_name']} | {item['count']:,} | {item['percentage']:.2f}% |")
        lines.append("")

    # 图表引用
    _insert_figure(lines, figure_paths, "category_dist", "01_category_distribution.png", "类别实例分布", "Category Instance Distribution", zh)
    _insert_figure(lines, figure_paths, "long_tail", "02_long_tail_curve.png", "长尾分布曲线", "Long-tail Distribution Curve", zh)

    lines.append("---")
    lines.append("")
    return lines


def _section_bbox_analysis(
    results: Dict[str, Any], figure_paths: Dict[str, str], zh: bool
) -> List[str]:
    """生成边界框分析章节。"""
    lines = []
    if zh:
        lines.append("## 3. 边界框统计分析")
    else:
        lines.append("## 3. Bounding Box Statistics")
    lines.append("")

    bbox = results.get("bbox", {})
    size_cls = bbox.get("size_classification", None)
    if size_cls is not None and not size_cls.empty:
        lines.append("| 尺寸类别 | 实例数 | 占比 |")
        lines.append("|----------|--------|------|")
        for _, row in size_cls.iterrows():
            lines.append(f"| {row['bbox_size_class']} | {int(row['count']):,} | {row['percentage']:.1f}% |")
        lines.append("")

    pos_bias = bbox.get("position_bias", {})
    if pos_bias:
        if zh:
            lines.append(f"**空间偏置**: {pos_bias.get('center_bias_description', 'N/A')}")
            lines.append(f"- 中心 X 均值: {pos_bias.get('center_x_mean', 0):.3f}")
            lines.append(f"- 中心 Y 均值: {pos_bias.get('center_y_mean', 0):.3f}")
        else:
            lines.append(f"**Spatial Bias**: {pos_bias.get('center_bias_description', 'N/A')}")
        lines.append("")

    _insert_figure(lines, figure_paths, "bbox_size", "04_bbox_size_pie.png", "目标尺寸分布", "Object Size Distribution", zh)
    _insert_figure(lines, figure_paths, "aspect_ratio", "05_aspect_ratio_distribution.png", "宽高比分布", "Aspect Ratio Distribution", zh)
    _insert_figure(lines, figure_paths, "position", "06_position_heatmap.png", "空间位置热力图", "Spatial Position Heatmap", zh)

    lines.append("---")
    lines.append("")
    return lines


def _section_cooccurrence_analysis(
    results: Dict[str, Any], figure_paths: Dict[str, str], zh: bool
) -> List[str]:
    """生成共现分析章节。"""
    lines = []
    if zh:
        lines.append("## 4. 目标共现网络分析")
    else:
        lines.append("## 4. Object Co-occurrence Network Analysis")
    lines.append("")

    coc = results.get("cooccurrence", {})
    top_pairs = coc.get("top_pairs", [])
    if top_pairs:
        if zh:
            lines.append("### 最频繁共现的类别对 (Top 10)")
        else:
            lines.append("### Top Co-occurring Category Pairs")
        lines.append("")
        lines.append("| 类别 A | 类别 B | 共现图像数 |")
        lines.append("|--------|--------|------------|")
        for pair in top_pairs[:10]:
            lines.append(f"| {pair['cat1']} | {pair['cat2']} | {pair['cooccurrence_count']:,} |")
        lines.append("")

    communities = coc.get("communities")
    if communities:
        if zh:
            lines.append(f"**社区检测**: 发现 {communities.get('num_communities', 0)} 个社区，模块度 = {communities.get('modularity', 0):.3f}")
            lines.append("")
            for comm in communities.get("communities", []):
                lines.append(f"- 社区 {comm['community_id'] + 1}: {', '.join(comm['members'][:5])}{'...' if len(comm['members']) > 5 else ''} ({comm['size']} 个类别)")
        else:
            lines.append(f"**Community Detection**: {communities.get('num_communities', 0)} communities found")
        lines.append("")

    _insert_figure(lines, figure_paths, "cooc_heatmap", "09_cooccurrence_heatmap.png", "共现聚类热力图", "Co-occurrence Clustered Heatmap", zh)

    lines.append("---")
    lines.append("")
    return lines


def _section_difficulty_analysis(
    results: Dict[str, Any], figure_paths: Dict[str, str], zh: bool
) -> List[str]:
    """生成难度分析章节。"""
    lines = []
    if zh:
        lines.append("## 5. 检测难度量化分析")
    else:
        lines.append("## 5. Detection Difficulty Quantification")
    lines.append("")

    diff = results.get("difficulty", {})

    hardest = diff.get("hardest_categories", [])
    if hardest:
        if zh:
            lines.append("### 检测难度最高的类别")
        else:
            lines.append("### Hardest Categories to Detect")
        lines.append("")
        lines.append("| 排名 | 类别 | 难度得分 | 总实例数 |")
        lines.append("|------|------|----------|----------|")
        for i, cat in enumerate(hardest[:10], 1):
            lines.append(f"| {i} | {cat['category_name']} | {cat['difficulty_score']:.1f} | {cat['total_count']:,} |")
        lines.append("")

    _insert_figure(lines, figure_paths, "small_obj", "13_small_object_scatter.png", "小目标分布分析", "Small Object Distribution", zh)
    _insert_figure(lines, figure_paths, "obj_per_img", "14_objects_per_image.png", "每图目标数分布", "Objects per Image Distribution", zh)
    _insert_figure(lines, figure_paths, "diff_rank", "15_difficulty_ranking.png", "检测难度排名", "Detection Difficulty Ranking", zh)

    lines.append("---")
    lines.append("")
    return lines


def _section_augmentation(results: Dict[str, Any], zh: bool) -> List[str]:
    """生成增强策略章节。"""
    lines = []

    aug_text = results.get("augmentation_text", "")
    if aug_text:
        lines.append(aug_text)
    else:
        if zh:
            lines.append("## 6. 数据增强策略建议")
        else:
            lines.append("## 6. Data Augmentation Strategy Recommendations")
        lines.append("")

        cat = results.get("category", {})
        diff = results.get("difficulty", {})

        if zh:
            lines.append("### 关键发现与建议")
            gini = cat.get("gini_coefficient", 0)
            if gini > 0.3:
                lines.append(f"- ⚠️ 类别分布 Gini 系数为 {gini:.3f}, 存在显著的长尾分布")
                lines.append(f"  → 建议: 对低频类别使用过采样或 Focal Loss")
            lines.append("")
            lines.append("- 🔍 **小目标检测**: 对高难度类别启用 Mosaic 增强 + 多尺度训练")
            lines.append("- 🔗 **共现关系**: 基于自然共现模式使用 Copy-Paste 增强")
            lines.append("- ⚖️ **类别平衡**: 对 Tail 类别使用 RepeatFactorDataset 策略")

    lines.append("")
    lines.append("---")
    lines.append("")
    return lines


def _section_appendix(results: Dict[str, Any], zh: bool) -> List[str]:
    """生成附录。"""
    lines = []
    if zh:
        lines.append("## 附录")
        lines.append("")
        lines.append("### 分析方法说明")
        lines.append("")
        lines.append("- **Gini 系数**: 衡量类别分布的均匀程度，0=完全均匀，1=完全不均")
        lines.append("- **COCO 尺寸分类**: Small (≤32² px), Medium (32²-96² px), Large (>96² px)")
        lines.append("- **Lift 值**: P(i,j) / (P(i)×P(j))，>1 表示正关联，<1 表示负关联")
        lines.append("- **难度评分**: 加权综合，权重见 config.yaml")
        lines.append("- **Louvain 社区检测**: 基于模块度优化的无监督图聚类算法")
    else:
        lines.append("## Appendix")
        lines.append("")
        lines.append("### Methodology")
        lines.append("")
        lines.append("- **Gini Coefficient**: Measures category distribution uniformity")
        lines.append("- **COCO Size Classes**: Small (≤32² px), Medium (32²-96² px), Large (>96² px)")
        lines.append("- **Lift Value**: P(i,j) / (P(i)×P(j)), >1 positive association, <1 negative")
        lines.append("- **Difficulty Score**: Weighted composite, weights in config.yaml")
        lines.append("- **Louvain Community Detection**: Unsupervised graph clustering based on modularity")
    lines.append("")

    if zh:
        lines.append("### 技术栈")
    else:
        lines.append("### Tech Stack")
    lines.append("")
    lines.append("- Python 3.10+ | pandas | numpy | matplotlib | seaborn | plotly")
    lines.append("- pycocotools | networkx | python-louvain")
    lines.append("- Jupyter Notebook | nbconvert")
    lines.append("")

    return lines


def _insert_figure(
    lines: List[str],
    figure_paths: Dict[str, str],
    key: str,
    fallback_filename: str,
    caption_zh: str,
    caption_en: str,
    zh: bool,
) -> None:
    """在报告中插入图片引用。"""
    path = figure_paths.get(key, "")
    if not path:
        # 尝试从 outputs/figures/ 拼接
        path = f"outputs/figures/{fallback_filename}"

    caption = caption_zh if zh else caption_en
    lines.append(f"![{caption}]({path})")
    lines.append(f"*{caption}*")
    lines.append("")


def generate_html_report(
    md_content: str,
    figure_paths: Dict[str, str],
    output_path: str = "outputs/reports/analysis_report.html",
    lang: str = "zh",
) -> str:
    """将 Markdown 报告转为独立的 HTML 文件（含内嵌图片）。

    Args:
        md_content: Markdown 报告内容
        figure_paths: 图表路径映射
        output_path: 输出文件路径
        lang: 语言

    Returns:
        输出文件路径

    Note:
        需要安装 markdown 库: pip install markdown
    """
    try:
        import markdown
    except ImportError:
        # Fallback: 生成简单的 HTML
        return _generate_simple_html(md_content, figure_paths, output_path, lang)

    # 将图片引用中的相对路径转为 base64 内嵌
    html_content = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
    )

    # 嵌入图片
    for key, path in figure_paths.items():
        if os.path.exists(path) and path.endswith(".png"):
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode()
            html_content = html_content.replace(
                f'src="{path}"',
                f'src="data:image/png;base64,{img_data}"',
            )
            # 也尝试 outputs/figures/ 前缀
            fallback = f"outputs/figures/{os.path.basename(path)}"
            html_content = html_content.replace(
                f'src="{fallback}"',
                f'src="data:image/png;base64,{img_data}"',
            )

    title = "COCO 2017 数据集深度分析报告" if lang == "zh" else "COCO 2017 Dataset Analysis Report"

    full_html = f"""<!DOCTYPE html>
<html lang="{'zh-CN' if lang == 'zh' else 'en'}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 2em;
            line-height: 1.6;
            color: #333;
            background: #fafafa;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 0.3em; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #bdc3c7; padding-bottom: 0.2em; margin-top: 2em; }}
        h3 {{ color: #7f8c8d; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #34495e; color: white; }}
        tr:nth-child(even) {{ background: #f2f2f2; }}
        img {{ max-width: 100%; height: auto; border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); margin: 1em 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }}
        pre {{ background: #2c3e50; color: #ecf0f1; padding: 1em; border-radius: 6px; overflow-x: auto; }}
        blockquote {{ border-left: 4px solid #3498db; padding-left: 1em; color: #7f8c8d; margin: 1em 0; }}
    </style>
</head>
<body>
{html_content}
</body>
</html>"""

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        f.write(full_html)

    return str(output_path_obj)


def _generate_simple_html(
    md_content: str,
    figure_paths: Dict[str, str],
    output_path: str,
    lang: str,
) -> str:
    """生成简单的 HTML 报告（不依赖 markdown 库的降级方案）。

    Args:
        md_content: Markdown 内容
        figure_paths: 图表路径
        output_path: 输出路径
        lang: 语言

    Returns:
        输出文件路径
    """
    # 简单地将 Markdown 转换为 HTML (基本处理)
    html = md_content

    # 标题
    html = html.replace("### ", "<h3>").replace("## ", "<h2>").replace("# ", "<h1>")
    for tag in ["h1", "h2", "h3"]:
        html = html.replace(f"<{tag}>", f"<{tag}>")
    # (简化处理，主要依赖 markdown 库)

    title = "COCO 2017 数据集深度分析报告" if lang == "zh" else "COCO 2017 Dataset Analysis Report"

    full_html = f"""<!DOCTYPE html>
<html lang="{'zh-CN' if lang == 'zh' else 'en'}">
<head><meta charset="UTF-8"><title>{title}</title></head>
<body style="font-family: sans-serif; max-width: 1000px; margin: 0 auto; padding: 2em;">
<pre>{md_content}</pre>
</body></html>"""

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "w", encoding="utf-8") as f:
        f.write(full_html)

    print("⚠ markdown 库未安装，生成了简化版 HTML。")
    print("  安装: pip install markdown")
    return str(output_path_obj)
