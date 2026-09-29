"""拓扑数据分析 (TDA): 持久同调 (Persistent Homology)

将状态空间看作度量空间(距离=最短路径长度), 构建 Vietoris-Rips 复形,
计算持久同调, 输出条形码(Barcode)与持久图(Persistence Diagram)。

性能优化:
- 距离矩阵构建: 字典索引代替 O(N^2) 双重遍历
- H1 计算: 倍增 LCA 预处理 O(N log N) + 单次查询 O(log N), 替代原 O(N) BFS
"""
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
from itertools import combinations
import numpy as np
import networkx as nx

from 字体工具 import apply_cn_font, CN_FONT


class TDAnalyzer:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.distance_matrix = None
        self.persistence = None

    # ==================== 距离矩阵 ====================

    def _build_distance_matrix(self, max_samples: int = 200,
                               sampled_nodes: Optional[List] = None) -> Tuple[np.ndarray, List]:
        """构建采样节点间的最短路径距离矩阵

        优化: 用 scipy C-level BFS 替代 NetworkX 纯 Python BFS, 大图提速 10-50x
        AI 加速: sampled_nodes 可由 GNN 嵌入选多样性节点, 提升覆盖质量
        """
        G = self.graph
        n = G.number_of_nodes()
        if n == 0:
            return np.array([]), []

        nodes = list(G.nodes())
        if sampled_nodes is not None:
            # AI 引导采样: 直接使用外部传入的多样性节点
            sampled = sampled_nodes
        elif n > max_samples:
            rng = np.random.default_rng(42)
            idx = sorted(rng.choice(n, size=max_samples, replace=False))
            sampled = [nodes[i] for i in idx]
        else:
            sampled = nodes

        m = len(sampled)
        D = np.full((m, m), np.inf, dtype=np.float64)
        np.fill_diagonal(D, 0.0)

        # ---- 大图: 用 scipy BFS (C 实现, 快 10-50x) ----
        if n > 2000:
            from scipy.sparse.csgraph import breadth_first_order
            node_to_global_idx = {node: i for i, node in enumerate(nodes)}
            adj_csr = nx.adjacency_matrix(G, nodelist=nodes).astype(np.int8)
            sampled_global_idx = np.array([node_to_global_idx[node] for node in sampled], dtype=np.int32)

            for i in range(m):
                src_idx = int(sampled_global_idx[i])
                try:
                    node_order, predecessors = breadth_first_order(
                        adj_csr, src_idx, directed=False, return_predecessors=True
                    )
                except Exception:
                    continue
                # 从 predecessors 重建距离
                dist = np.full(n, -1, dtype=np.int32)
                dist[src_idx] = 0
                for node in node_order:
                    p = predecessors[node]
                    if p >= 0:
                        dist[node] = dist[p] + 1
                # 提取采样节点间的距离
                for j in range(i + 1, m):
                    d = dist[sampled_global_idx[j]]
                    if d >= 0:
                        D[i, j] = float(d)
                        D[j, i] = float(d)
        else:
            # ---- 小图: NetworkX BFS (够快) ----
            node_to_idx = {node: i for i, node in enumerate(sampled)}
            for i, src in enumerate(sampled):
                try:
                    dist_dict = nx.single_source_shortest_path_length(G, src)
                    for tgt, dist in dist_dict.items():
                        j = node_to_idx.get(tgt)
                        if j is not None:
                            D[i, j] = float(dist)
                except Exception:
                    continue

        finite = D[np.isfinite(D)]
        fill_val = float(2.0 * np.max(finite)) if len(finite) > 0 else 100.0
        D[~np.isfinite(D)] = fill_val

        self.distance_matrix = D
        return D, sampled

    # ==================== Vietoris-Rips 持久同调 ====================

    @staticmethod
    def _vietoris_rips_pairs(D: np.ndarray, max_dim: int = 1,
                             max_epsilon: Optional[float] = None) -> Dict[int, List[Tuple[float, float]]]:
        """标准 Vietoris-Rips (等价于 flag 复形) 持久同调, Z2 系数。

        算法: 把顶点/边/三角形全部按「直径」入过滤, 同值时先低维 (faces before
        cofaces), 然后对边界矩阵做标准列约简; 每个 pivot 对 (i, j) 给出一条
        H_dim 的条 (birth=filt(i), death=filt(j)), 维数 dim = dim(i),
        因为 H_k 由 k-单形出生、被 (k+1)-单形杀死。

        注意 (2026-09 修正): 旧实现用「Kruskal 最小生成树 + 每条非树边查 MST
        路径最大边」来产生 H1 条, 那只用到 MST, 没有考虑三角形把环填掉,
        因此对标准 Rips 会算错。实测对照 (标准 Rips H1 / 旧实现 H1):
            C4 四环   3 / 2      C6 六环  10 / 9      C8 八环  21 / 20
            K6 完全图 10 / 0     Q3 立方体 21 / 16
        H0 两种算法都给对 (H0 只依赖边的顺序 + 并查集)。

        复杂度: 三角形数量 O(n^3), 故采样点数 max_samples 直接决定本函数开销。
        """
        n = D.shape[0]
        if n == 0:
            return {0: [], 1: []}
        if max_epsilon is None:
            max_epsilon = float(np.max(D)) * 1.05

        result = {0: [], 1: []}

        # ---------- 1. 枚举单纯形并按过滤值排序 ----------
        # 单纯形用 (维数, 顶点元组) 表示; 过滤值单独放 filt 列表, 不参与查表键
        # (否则同过滤值的多个单形会互相覆盖)。
        simplices = [(0, (i,)) for i in range(n)]

        iu, ju = np.triu_indices(n, k=1)
        weights = D[iu, ju]
        keep = weights <= max_epsilon
        simplices.extend((1, (int(a), int(b)))
                         for a, b in zip(iu[keep], ju[keep]))

        if max_dim >= 1:
            for t in combinations(range(n), 3):
                if max(D[t[0], t[1]], D[t[0], t[2]], D[t[1], t[2]]) <= max_epsilon:
                    simplices.append((2, t))

        def filt_of(s):
            dim, verts = s
            if dim == 0:
                return 0.0
            return float(max(D[a, b] for a, b in combinations(verts, 2)))

        filt = [filt_of(s) for s in simplices]

        # 同过滤值先低维 (faces before cofaces) —— 这是持久同调正确性的前提
        order = sorted(range(len(simplices)), key=lambda k: (filt[k], simplices[k][0]))
        simplices = [simplices[k] for k in order]
        filt = [filt[k] for k in order]
        index = {s: k for k, s in enumerate(simplices)}

        # ---------- 2. 边界矩阵约简 (Z2, 位集表示列) ----------
        # 约简后每个非零列的 low = 最大行号; low_owner[low] = 拥有该 low 的列号
        reduced = []
        low_owner: Dict[int, int] = {}
        pairs: List[Tuple[int, int]] = []
        for j, (dim, verts) in enumerate(simplices):
            col = set()
            if dim > 0:
                # 面 = 去掉一个顶点, 即所有 dim 元子集:
                # 1-单形(边) -> 2 个顶点; 2-单形(三角形) -> 3 条边
                for face in combinations(verts, dim):
                    col.add(index[(dim - 1, face)])
            # 消元: 只要当前 low 已被别的列占用, 就异或掉那一列
            while col:
                low = max(col)
                owner = low_owner.get(low)
                if owner is None:
                    break
                col ^= reduced[owner]
            reduced.append(col)
            if col:
                low = max(col)
                low_owner[low] = j
                pairs.append((low, j))

        # ---------- 3. 配对 -> 条码 ----------
        born = set()
        died = set()
        for birth_i, death_j in pairs:
            birth_dim = simplices[birth_i][0]
            if birth_dim in result:
                result[birth_dim].append((filt[birth_i], filt[death_j]))
            born.add(birth_i)
            died.add(death_j)

        # 本质类: 该维数的单纯形既不是任何配对的出生侧也不是死亡侧
        for k, (dim, _) in enumerate(simplices):
            if dim in result and k not in born and k not in died:
                result[dim].append((filt[k], float('inf')))

        for dim in result:
            result[dim].sort(key=lambda x: (x[0], x[1]))
        return result

    # ==================== 旧的 MST 近似 (保留备用) ====================
    # 说明: 这套「Kruskal + LCA」只近似了 MST 的下星过滤, 不是 Rips 持久同调,
    # 结果与标准实现不一致 (见 _vietoris_rips_pairs 的对照数据), 故不再被调用。
    # 保留在此仅供对照与回归测试使用。

    @staticmethod
    def _vietoris_rips_pairs_mst_approx(D: np.ndarray, max_dim: int = 1,
                                        max_epsilon: Optional[float] = None
                                        ) -> Dict[int, List[Tuple[float, float]]]:
        n = D.shape[0]
        if max_epsilon is None:
            max_epsilon = float(np.max(D)) * 1.05

        result = {0: [], 1: []}

        # ---- 构建 + 排序所有边 ----
        # 向量化: 上三角索引一次性提取
        iu, ju = np.triu_indices(n, k=1)
        weights = D[iu, ju]
        order = np.argsort(weights, kind='mergesort')
        edges = list(zip(weights[order], iu[order], ju[order]))

        # ---- H0: Kruskal + Union-Find ----
        parent = list(range(n))
        rank_ = [0] * n

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b) -> bool:
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank_[ra] < rank_[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            rank_[ra] += rank_[ra] == rank_[rb]
            return True

        components_remaining = n
        non_tree_edges = []
        for w, u, v in edges:
            if w > max_epsilon:
                break
            if union(u, v):
                components_remaining -= 1
                result[0].append((0.0, float(w)))
            else:
                non_tree_edges.append((w, u, v))

        for _ in range(components_remaining):
            result[0].append((0.0, np.inf))

        # ---- H1: 非树边 -> 独立环 ----
        if max_dim >= 1 and non_tree_edges:
            # 重建 MST 邻接表
            mst_adj = [[] for _ in range(n)]
            uf_parent = list(range(n))
            uf_rank = [0] * n

            def find2(x):
                while uf_parent[x] != x:
                    uf_parent[x] = uf_parent[uf_parent[x]]
                    x = uf_parent[x]
                return x

            def union2(a, b):
                ra, rb = find2(a), find2(b)
                if ra == rb:
                    return False
                if uf_rank[ra] < uf_rank[rb]:
                    ra, rb = rb, ra
                uf_parent[rb] = ra
                uf_rank[ra] += uf_rank[ra] == uf_rank[rb]
                return True

            for w, u, v in edges:
                if w > max_epsilon:
                    break
                if union2(u, v):
                    mst_adj[u].append((v, w))
                    mst_adj[v].append((u, w))

            # ---- 倍增 LCA 预处理: O(n log n) ----
            up, max_up, depth, LOG = TDAnalyzer._build_lca(mst_adj, n)

            # ---- 每条非树边查询: O(log n) ----
            for w, u, v in non_tree_edges:
                birth = TDAnalyzer._query_max_edge(u, v, up, max_up, depth, LOG)
                death = float(w)
                if death > birth and death <= max_epsilon:
                    result[1].append((float(birth), death))

        for dim in result:
            result[dim].sort(key=lambda x: (x[0], x[1]))

        return result

    # ==================== 倍增 LCA ====================

    @staticmethod
    def _build_lca(mst_adj, n, root=0):
        """构建倍增表: up[k][v] = v 的 2^k 祖先, max_up[k][v] = 路径最大边权

        预处理 O(n log n), 查询 O(log n)
        """
        LOG = max(1, n.bit_length())

        # 用 list of array 提速
        up = [[-1] * n for _ in range(LOG)]
        max_up = [[0.0] * n for _ in range(LOG)]
        depth = [0] * n

        # BFS 建树
        visited = [False] * n
        visited[root] = True
        q = deque([root])
        while q:
            u = q.popleft()
            for v, w in mst_adj[u]:
                if not visited[v]:
                    visited[v] = True
                    depth[v] = depth[u] + 1
                    up[0][v] = u
                    max_up[0][v] = w
                    q.append(v)

        # 倍增预处理
        for k in range(1, LOG):
            up_k = up[k]
            up_km1 = up[k - 1]
            max_k = max_up[k]
            max_km1 = max_up[k - 1]
            for v in range(n):
                p = up_km1[v]
                if p != -1:
                    up_k[v] = up_km1[p]
                    max_k[v] = max(max_km1[v], max_km1[p])
                else:
                    up_k[v] = -1
                    max_k[v] = 0.0

        return up, max_up, depth, LOG

    @staticmethod
    def _query_max_edge(u, v, up, max_up, depth, LOG):
        """查询 MST 上 u→v 路径的最大边权 (LCA)"""
        if u == v:
            return 0.0
        result = 0.0
        # 确保 u 更深
        if depth[u] < depth[v]:
            u, v = v, u
        # 提升 u 到 v 的深度
        diff = depth[u] - depth[v]
        k = 0
        while diff:
            if diff & 1:
                result = max(result, max_up[k][u])
                u = up[k][u]
            diff >>= 1
            k += 1
        if u == v:
            return result
        # 同时提升两者直到 LCA 子层
        for k in range(LOG - 1, -1, -1):
            if up[k][u] != up[k][v]:
                result = max(result, max_up[k][u], max_up[k][v])
                u = up[k][u]
                v = up[k][v]
        # u, v 现在是 LCA 的直接子节点
        result = max(result, max_up[0][u], max_up[0][v])
        return result

    # ==================== 持久同调主入口 ====================

    def compute_persistence(self, max_samples: int = 200, max_dim: int = 1,
                            progress_callback: Optional[Callable[[str], None]] = None,
                            embeddings: Optional[np.ndarray] = None,
                            nodes_list: Optional[List] = None) -> Dict:
        """计算持久同调

        AI 加速: 若提供 GNN embeddings, 用 K-means 选多样性采样节点,
        减少所需的 max_samples (100 即可达到随机 200 的覆盖质量)。
        """
        if progress_callback:
            progress_callback("distance_matrix")

        # AI 引导采样: 用 GNN 嵌入做 K-means, 取簇心附近节点
        sampled_nodes = None
        ai_sampling = False
        if embeddings is not None and nodes_list is not None:
            from sklearn.cluster import KMeans as _KMeans
            n_total = len(nodes_list)
            n_samp = min(max_samples, n_total)
            k = min(n_samp, n_total)
            if k < n_total:
                km = _KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(embeddings)
                # 每簇选离簇心最近的节点
                centers = km.cluster_centers_
                sampled_nodes = []
                for ci in range(k):
                    mask = labels == ci
                    if not np.any(mask):
                        continue
                    dists = np.linalg.norm(embeddings[mask] - centers[ci], axis=1)
                    local_idx = np.where(mask)[0]
                    best = local_idx[np.argmin(dists)]
                    sampled_nodes.append(nodes_list[best])
                ai_sampling = True
                if progress_callback:
                    progress_callback(f"AI采样({len(sampled_nodes)}节点)")

        D, nodes = self._build_distance_matrix(max_samples=max_samples,
                                                sampled_nodes=sampled_nodes)
        if len(D) == 0:
            return {'n_samples': 0, 'bars': {0: [], 1: []}, 'nodes': []}

        if progress_callback:
            progress_callback("vietoris_rips")
        bars = self._vietoris_rips_pairs(D, max_dim=max_dim)
        self.persistence = bars

        if progress_callback:
            progress_callback("stats")
        stats = {}
        for dim in sorted(bars.keys()):
            pairs = bars[dim]
            finite_pairs = [(b, d) for b, d in pairs if np.isfinite(d)]
            infinite = len(pairs) - len(finite_pairs)
            if finite_pairs:
                lifetimes = [d - b for b, d in finite_pairs]
                stats[dim] = {
                    'total_bars': len(pairs),
                    'finite_bars': len(finite_pairs),
                    'infinite_bars': infinite,
                    'mean_lifetime': float(np.mean(lifetimes)),
                    'max_lifetime': float(np.max(lifetimes)),
                    'median_lifetime': float(np.median(lifetimes)),
                    'persistence_entropy': self._entropy(lifetimes),
                }
            else:
                stats[dim] = {
                    'total_bars': len(pairs),
                    'finite_bars': 0,
                    'infinite_bars': infinite,
                }

        return {
            'n_samples': len(nodes),
            'bars': bars,
            'distance_matrix_shape': D.shape,
            'nodes': nodes,
            'stats': stats,
            'ai_sampling': ai_sampling,
        }

    @staticmethod
    def _entropy(lifetimes: List[float]) -> float:
        arr = np.asarray(lifetimes, dtype=np.float64)
        if arr.sum() <= 0:
            return 0.0
        p = arr / arr.sum()
        p = p[p > 0]
        return float(-np.sum(p * np.log(p)))

    # ==================== 条形码绘制 ====================

    def plot_barcode(self, ax, result: Dict, max_bars_per_dim: int = 80):
        bars = result.get('bars', {})
        # JSON 往返会把维数键 0/1 变成字符串 '0'/'1' (布局级结果缓存与会话存档都会发生),
        # 这里统一归一回 int, 否则 dim_colors[dim % len] 会对字符串取模抛 TypeError。
        bars = {int(k): v for k, v in bars.items()}
        dims = sorted(bars.keys())
        if not dims:
            ax.text(0.5, 0.5, "无持久同调数据", ha='center', va='center')
            ax.axis('off')
            return

        dim_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']

        y_pos = 0
        yticks = []
        yticklabels_raw = []
        max_x = 0.0

        for dim in dims:
            pairs = bars[dim]
            if not pairs:
                continue
            scored = sorted(pairs, key=lambda x: (x[1] - x[0]) if np.isfinite(x[1]) else 1e18, reverse=True)
            selected = scored[:max_bars_per_dim]
            selected.sort(key=lambda x: x[0])

            color = dim_colors[dim % len(dim_colors)]
            for b, d in selected:
                if np.isfinite(d):
                    end_x = d
                else:
                    end_x = max([pp[0] for pp in pairs if np.isfinite(pp[1])] + [b + 1.0]) * 1.1
                max_x = max(max_x, end_x)
                ax.plot([b, end_x], [y_pos, y_pos], color=color, linewidth=1.4, alpha=0.85)
                if not np.isfinite(d):
                    ax.plot(end_x, y_pos, marker='>', color=color, markersize=4)
                y_pos += 1
            yticks.append(y_pos - 0.5 - len(selected) / 2)
            yticklabels_raw.append(f"H{dim} ({len(selected)}/{len(pairs)})")
            if dim != dims[-1]:
                ax.axhline(y_pos - 0.3, color='gray', linewidth=0.5, linestyle=':')

        ax.set_xlim(-0.02 * max(max_x, 1), max_x * 1.05)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels_raw, fontsize=9)
        ax.grid(axis='x', alpha=0.3)

        apply_cn_font(ax,
                      title="持久同调条形码\n(维数-过滤时间 拓扑特征生存区间)",
                      xlabel="过滤参数 e (图距离 = 最短路径步数)",
                      ylabel=None,
                      title_fontsize=10, label_fontsize=9, tick_fontsize=9)
        for lbl in ax.get_yticklabels():
            lbl.set_fontproperties(CN_FONT)
            lbl.set_fontsize(9)
