"""目标共现网络分析模块。

分析 COCO 数据集中不同类别目标在同一图像中的共现模式，
构建共现矩阵和网络图，检测社区结构。
"""

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


def build_cooccurrence_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """构建 N×N 共现矩阵。

    matrix[i, j] = 类别 i 和 j 在同一张图像中共同出现的图像数。

    注意: 对角线 matrix[i, i] = 类别 i 出现的图像数。

    Args:
        df: 标注 DataFrame（需包含 image_id 和 category_name 列）

    Returns:
        以 category_name 为行/列标签的对称共现矩阵
    """
    # 获取每张图像中包含的类别集合
    img_cats = df.groupby("image_id")["category_name"].apply(set).reset_index()
    img_cats.columns = ["image_id", "cat_set"]

    categories = sorted(df["category_name"].unique())
    n = len(categories)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}

    cooc = np.zeros((n, n), dtype=int)

    for _, row in img_cats.iterrows():
        cat_list = list(row["cat_set"])
        indices = [cat_to_idx[c] for c in cat_list]
        for i in indices:
            for j in indices:
                cooc[i, j] += 1

    return pd.DataFrame(cooc, index=categories, columns=categories)


def build_cooccurrence_probability(cooc_df: pd.DataFrame) -> pd.DataFrame:
    """从共现矩阵计算条件概率 P(j | i)。

    P(j | i) = 类别 i 和 j 共现的图像数 / 类别 i 出现的图像数。

    Args:
        cooc_df: build_cooccurrence_matrix() 的返回结果

    Returns:
        条件概率矩阵 (非对称)
    """
    prob = cooc_df.copy().astype(float)
    for i in prob.index:
        total_i = prob.loc[i, i]  # 对角线 = 类别 i 出现的总图像数
        if total_i > 0:
            prob.loc[i] = prob.loc[i] / total_i
    return prob


def compute_lift_matrix(cooc_df: pd.DataFrame) -> pd.DataFrame:
    """计算 Lift 值矩阵。

    Lift(i, j) = P(j | i) / P(j) = P(i and j) / (P(i) * P(j))

    Lift > 1 表示两个类别在一起出现的概率高于随机期望
    (有正向关联)，Lift < 1 表示负向关联。

    Args:
        cooc_df: 共现矩阵

    Returns:
        Lift 值矩阵
    """
    categories = cooc_df.index.tolist()
    n = len(categories)
    total_images = cooc_df.max().max()  # 对角线最大值

    lift = pd.DataFrame(np.eye(n), index=categories, columns=categories)

    for i in categories:
        p_i = cooc_df.loc[i, i] / total_images
        for j in categories:
            if i == j:
                continue
            p_j = cooc_df.loc[j, j] / total_images
            p_ij = cooc_df.loc[i, j] / total_images
            if p_i > 0 and p_j > 0:
                lift.loc[i, j] = p_ij / (p_i * p_j)

    return lift


def find_top_cooccurring_pairs(
    cooc_df: pd.DataFrame, top_k: int = 30
) -> List[Dict[str, Any]]:
    """寻找最频繁共现的类别对（排除自对）。

    Args:
        cooc_df: 共现矩阵
        top_k: 返回前 k 对

    Returns:
        列表，每项包含:
        - cat1, cat2: 类别名称
        - cooccurrence_count: 共现图像数
        - cat1_freq: 类别 1 单独出现的图像数
        - cat2_freq: 类别 2 单独出现的图像数
    """
    pairs = []
    categories = cooc_df.index.tolist()

    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue
            count = int(cooc_df.loc[cat1, cat2])
            cat1_freq = int(cooc_df.loc[cat1, cat1])
            cat2_freq = int(cooc_df.loc[cat2, cat2])
            pairs.append({
                "cat1": cat1,
                "cat2": cat2,
                "cooccurrence_count": count,
                "cat1_freq": cat1_freq,
                "cat2_freq": cat2_freq,
            })

    pairs.sort(key=lambda x: x["cooccurrence_count"], reverse=True)
    return pairs[:top_k]


def build_cooccurrence_graph(
    cooc_df: pd.DataFrame,
    lift_df: pd.DataFrame | None = None,
    min_edges: int = 50,
    use_lift_weight: bool = True,
) -> "nx.Graph":
    """构建共现网络图。

    节点 = 类别，边权重 = 共现次数或 Lift 值。

    Args:
        cooc_df: 共现矩阵
        lift_df: Lift 值矩阵（可选，用于边权重）
        min_edges: 最小共现次数阈值，低于此值的边不加入
        use_lift_weight: True 时使用 Lift 值作为边权重，否则用共现次数

    Returns:
        networkx.Graph 对象
    """
    if not HAS_NETWORKX:
        raise ImportError("需要安装 networkx: pip install networkx")

    G = nx.Graph()
    categories = cooc_df.index.tolist()

    # 添加节点
    for cat in categories:
        G.add_node(cat, weight=int(cooc_df.loc[cat, cat]))

    # 添加边
    for i, cat1 in enumerate(categories):
        for j, cat2 in enumerate(categories):
            if i >= j:
                continue
            weight = int(cooc_df.loc[cat1, cat2])
            if weight < min_edges:
                continue

            if use_lift_weight and lift_df is not None:
                edge_weight = float(lift_df.loc[cat1, cat2])
            else:
                edge_weight = float(weight)

            G.add_edge(cat1, cat2, weight=edge_weight, cooccurrence=weight)

    return G


def compute_graph_metrics(G: "nx.Graph") -> pd.DataFrame:
    """计算网络图的关键拓扑指标。

    Args:
        G: networkx 图对象

    Returns:
        DataFrame，每个节点一行，包含:
        - category: 类别名称
        - degree: 度（连接的边数）
        - weighted_degree: 加权度
        - betweenness_centrality: 介数中心性
        - eigenvector_centrality: 特征向量中心性
        - clustering_coefficient: 聚类系数
    """
    if not HAS_NETWORKX:
        raise ImportError("需要安装 networkx: pip install networkx")

    degree_dict = dict(G.degree())
    weighted_degree = dict(G.degree(weight="weight"))

    # 介数中心性（对小图使用精确计算）
    betweenness = nx.betweenness_centrality(G, weight="weight")

    # 特征向量中心性
    try:
        eigenvector = nx.eigenvector_centrality_numpy(G, weight="weight")
    except Exception:
        eigenvector = {n: 0.0 for n in G.nodes()}

    # 聚类系数
    clustering = nx.clustering(G, weight="weight")

    nodes = list(G.nodes())
    result = pd.DataFrame({
        "category": nodes,
        "degree": [degree_dict.get(n, 0) for n in nodes],
        "weighted_degree": [weighted_degree.get(n, 0) for n in nodes],
        "betweenness_centrality": [betweenness.get(n, 0) for n in nodes],
        "eigenvector_centrality": [eigenvector.get(n, 0) for n in nodes],
        "clustering_coefficient": [clustering.get(n, 0) for n in nodes],
    })

    return result.sort_values("betweenness_centrality", ascending=False).reset_index(drop=True)


def detect_communities(G: "nx.Graph") -> Dict[str, Any]:
    """使用 Louvain 算法检测社区结构。

    Args:
        G: networkx 图对象

    Returns:
        字典，包含:
        - partition: {category: community_id} 映射
        - num_communities: 社区数量
        - communities: 每个社区包含的类别列表
        - modularity: 模块度得分
    """
    if not HAS_LOUVAIN:
        raise ImportError("需要安装 python-louvain: pip install python-louvain")

    partition = community_louvain.best_partition(G, weight="weight")
    modularity = community_louvain.modularity(partition, G, weight="weight")

    # 按社区分组
    communities_dict: Dict[int, List[str]] = {}
    for node, comm_id in partition.items():
        communities_dict.setdefault(comm_id, []).append(node)

    communities_list = [
        {"community_id": cid, "members": sorted(members), "size": len(members)}
        for cid, members in sorted(communities_dict.items())
    ]

    return {
        "partition": partition,
        "num_communities": len(communities_dict),
        "communities": communities_list,
        "modularity": modularity,
    }


def generate_cooccurrence_summary(
    df: pd.DataFrame, config: dict | None = None
) -> Dict[str, Any]:
    """生成共现网络分析的完整摘要。

    Args:
        df: 标注 DataFrame
        config: 分析配置

    Returns:
        包含所有共现分析结果的字典
    """
    cfg = config or {}
    coc_cfg = cfg.get("cooccurrence", {})
    min_cooc = coc_cfg.get("min_cooccurrence", 10)
    top_k = coc_cfg.get("top_k_pairs", 30)
    edge_threshold = coc_cfg.get("network_edge_threshold", 50)

    cooc_matrix = build_cooccurrence_matrix(df)
    prob_matrix = build_cooccurrence_probability(cooc_matrix)
    lift_matrix = compute_lift_matrix(cooc_matrix)
    top_pairs = find_top_cooccurring_pairs(cooc_matrix, top_k=top_k)

    # 图形分析（如果可用）
    graph_metrics = None
    communities = None
    try:
        G = build_cooccurrence_graph(
            cooc_matrix, lift_matrix, min_edges=edge_threshold
        )
        graph_metrics = compute_graph_metrics(G)
    except (ImportError, Exception):
        G = None

    if G is not None:
        try:
            communities = detect_communities(G)
        except (ImportError, Exception):
            pass

    return {
        "cooccurrence_matrix": cooc_matrix,
        "probability_matrix": prob_matrix,
        "lift_matrix": lift_matrix,
        "top_pairs": top_pairs,
        "graph_metrics": graph_metrics,
        "communities": communities,
    }
