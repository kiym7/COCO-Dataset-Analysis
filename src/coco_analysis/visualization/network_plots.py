"""共现网络相关的可视化函数。"""

from pathlib import Path
from typing import Dict, Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .theme import (
    get_supercategory_color,
    save_figure,
)


def plot_cooccurrence_heatmap(
    cooc_df: pd.DataFrame,
    title: str = "COCO 2017 目标共现矩阵",
    output_dir: str = "outputs/figures",
    log_scale: bool = True,
) -> str:
    """绘制共现矩阵的聚类热力图。

    Args:
        cooc_df: build_cooccurrence_matrix() 的返回结果
        title: 图表标题
        output_dir: 输出目录
        log_scale: 是否对颜色使用对数变换

    Returns:
        保存的文件路径
    """
    # 用层次聚类排序行列
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    # 构建距离矩阵 (1 - 归一化共现)
    diag = np.diag(cooc_df.values)
    norm_cooc = cooc_df.values / np.sqrt(np.outer(diag, diag))

    # 使用 linkage 做层次聚类
    dist = 1 - norm_cooc
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    order = leaves_list(Z)

    # 重排
    categories = cooc_df.index.tolist()
    ordered_cats = [categories[i] for i in order]
    ordered_matrix = cooc_df.loc[ordered_cats, ordered_cats].copy()

    # 对数变换
    if log_scale:
        plot_data = np.log1p(ordered_matrix.values)
    else:
        plot_data = ordered_matrix.values

    fig, ax = plt.subplots(figsize=(20, 18))

    sns.heatmap(
        plot_data,
        xticklabels=ordered_cats,
        yticklabels=ordered_cats,
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "log(共现次数 + 1)" if log_scale else "共现次数"},
        linewidths=0.1,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("类别", fontsize=12)
    ax.set_ylabel("类别", fontsize=12)
    ax.tick_params(axis="both", labelsize=7, rotation=90)

    plt.tight_layout()
    return save_figure(fig, "09_cooccurrence_heatmap", output_dir)[0]


def plot_cooccurrence_network_interactive(
    G: "nx.Graph",
    communities: Optional[Dict[str, Any]] = None,
    title: str = "COCO 2017 目标共现网络",
    output_dir: str = "outputs/figures",
) -> str:
    """使用 plotly 绘制交互式共现网络图。

    Args:
        G: networkx Graph 对象
        communities: detect_communities() 的输出 (可选)
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    import networkx as nx
    import plotly.graph_objects as go

    # 计算布局
    pos = nx.spring_layout(G, k=1.5, iterations=100, seed=42, weight="weight")

    # 节点颜色（按社区或按度）
    if communities and "partition" in communities:
        partition = communities["partition"]
        community_ids = list(set(partition.values()))
        color_map = {cid: f"hsl({i * 360 / len(community_ids)}, 60%, 50%)" for i, cid in enumerate(community_ids)}
        node_colors = [color_map.get(partition.get(n, -1), "#999") for n in G.nodes()]
        legend_text = f"社区数: {len(community_ids)}"
    else:
        degrees = dict(G.degree(weight="weight"))
        max_deg = max(degrees.values()) if degrees else 1
        node_colors = [degrees.get(n, 0) / max_deg for n in G.nodes()]
        legend_text = "按加权度着色"

    # 节点
    node_x, node_y = [], []
    node_text = []
    node_sizes = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_sizes.append(np.log1p(G.nodes[node].get("weight", 1)) * 10)
        node_text.append(
            f"<b>{node}</b><br>"
            f"出现图像数: {G.nodes[node].get('weight', 0):,}<br>"
            f"度: {G.degree(node)}<br>"
            f"加权度: {G.degree(node, weight='weight'):.1f}"
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            size=node_sizes,
            color=node_colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=legend_text),
            line=dict(width=1, color="#333"),
        ),
        hovertext=node_text,
        hoverinfo="text",
    )

    # 边
    edge_x, edge_y = [], []
    edge_weights = []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(data.get("weight", 1))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="#888", shape="linear"),
        hoverinfo="none",
        opacity=0.4,
    )

    fig = go.Figure(data=[edge_trace, node_trace])

    fig.update_layout(
        title=dict(text=title, font=dict(size=18)),
        showlegend=False,
        hovermode="closest",
        width=1200,
        height=900,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(t=50, l=25, r=25, b=25),
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filepath = output_path / "10_cooccurrence_network.html"
    fig.write_html(str(filepath))
    return str(filepath)


def plot_top_cooccurring_chord(
    cooc_df: pd.DataFrame,
    top_k: int = 20,
    title: str = "COCO 2017 高频共现类别对",
    output_dir: str = "outputs/figures",
) -> str:
    """绘制 top-k 类别之间的共现关系条形图（Chord 图的简化替代）。

    Args:
        cooc_df: 共现矩阵
        top_k: 显示前 k 对
        title: 图表标题
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    # 提取边
    categories = cooc_df.index.tolist()
    pairs = []
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue
            pairs.append({
                "pair": f"{cat1} — {cat2}",
                "count": int(cooc_df.loc[cat1, cat2]),
            })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    top_pairs = pairs[:top_k]

    fig, ax = plt.subplots(figsize=(12, 10))

    names = [p["pair"] for p in top_pairs]
    values = [p["count"] for p in top_pairs]

    # 渐变颜色
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(values)))

    bars = ax.barh(range(len(names)), values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    ax.set_xlabel("共现次数 (图像数)", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    # 标注数值
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    return save_figure(fig, "11_top_cooccurring_pairs", output_dir)[0]
