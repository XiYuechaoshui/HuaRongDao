"""代数拓扑 + 代数图论 分析器

研究方向:
1. 谱图理论: 拉普拉斯矩阵特征值谱
2. 代数连通度 (Fiedler值)
3. 同调论: Betti数 (beta0连通分支, beta1独立环数)、Euler示性数
4. 围长 (Girth): 最短环长度
5. 环基 (Cycle Basis): 最小独立环集合
6. 拓扑中心性: Betweenness / Closeness / Eigenvector / PageRank

性能优化:
- 大图 (N>800) 谱计算用 scipy.sparse.linalg.eigsh 只算前 k 个特征值
- 大图中心性用采样近似
- 大图跳过昂贵的 minimum_cycle_basis
- full_analysis 各部分并行执行
"""
from typing import Dict, List, Tuple, Callable, Optional
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order

# ---- 性能阈值 (AI提速强化: 阈值降低, 更早启用AI近似) ----
_SPECTRUM_SPARSE_THRESHOLD = 500   # AI提速: 从800降低到500, 更早用稀疏特征值
_SPECTRUM_K = 200                    # 稀疏特征值数
_SPECTRUM_K_LARGE = 50               # 大图(>2000)减少到50
# 状态图规模(数万节点)下, 拉普拉斯小特征值用 ARPACK 只能求有限个;
# 这里给出"最多求几个"的硬上限族: n 越大, ARPACK 的每步迭代越贵, k 必须跟着降。
_SPECTRUM_K_SMALL_N = 20   # n <= 20000
_SPECTRUM_K_MID_N = 10     # n <= 60000
_SPECTRUM_K_BIG_N = 4      # n > 60000  (只为拿到 λ2, 也就是代数连通度)
_SPECTRUM_SKIP_NORM_LAP = 2000      # AI提速: 从5000降低到2000, 更早跳过归一化拉普拉斯
_CENTRALITY_SAMPLE_THRESHOLD = 500  # AI提速: 从1000降低到500, 更早采样
_CENTRALITY_SAMPLE_SIZE = 200       # 采样节点数
_CENTRALITY_SAMPLE_LARGE = 50       # 大图(>2000)采样数
_CENTRALITY_SKIP_BETWEENNESS = 500  # AI提速: 从2000降低到500, 更早跳过介数 (GNN嵌入近似)
_CYCLE_BASIS_NODE_LIMIT = 300       # AI提速: 从500降低到300, 更早跳过完整环基
_CYCLE_BASIS_BETA1_LIMIT = 30       # AI提速: 从50降低到30, 更早跳过完整环基


def _scipy_bfs_distances(adj_csr: csr_matrix, source_idx: int) -> np.ndarray:
    """用 scipy 的 C 实现 BFS 计算单源最短路径长度 (比 NetworkX 快 10-50x)

    返回: dist 数组, -1 表示不可达
    """
    n = adj_csr.shape[0]
    node_order, predecessors = breadth_first_order(
        adj_csr, source_idx, directed=False, return_predecessors=True
    )
    dist = np.full(n, -1, dtype=np.int32)
    dist[source_idx] = 0
    for node in node_order:
        p = predecessors[node]
        if p >= 0:
            dist[node] = dist[p] + 1
    return dist


class TopologyAnalyzer:

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self._spectrum_cache = None

    # ==================== 谱图理论 ====================

    def compute_spectrum(self, progress_callback: Optional[Callable[[str], None]] = None) -> Dict:
        """谱图特征

        小图 (n <= _SPECTRUM_SPARSE_THRESHOLD): 稠密精确分解, 拿到完整谱并据此算有效电阻。
        大图: 稀疏近似, 只求谱半径 / λ2 / 尽量多的小特征值。

        性能注记 (2026-09 实测): 旧实现对拉普拉斯用
        ``eigsh(L, k=k, sigma=1e-6, which='LM')``。带 sigma 会走 shift-invert,
        ARPACK 需要先做一次 (L - σI) 的稀疏 LU 分解, 复杂度接近 O(n^3), 实测:
            n=5,000   ->   2.9s
            n=20,000  -> 185.3s
            n=46,000  ->  >600s 仍未完成
        而状态图的 n 是 6.5 万~7.8 万, 这就是原来 topo_time 高达 18~265 秒的根因
        (占 full_analysis 总耗时的大部分)。改用 which='SA' (不用 sigma) 后,
        拉普拉斯只做矩阵-向量乘, 精度与 shift-invert 一致 (实测差 2e-14)。
        """
        if self._spectrum_cache is not None:
            return self._spectrum_cache

        G = self.graph
        n = G.number_of_nodes()

        if n <= 2:
            result = {
                'adjacency_eigenvalues': np.array([]),
                'laplacian_eigenvalues': np.array([0.0] * n),
                'normalized_laplacian_eigenvalues': np.array([0.0] * n),
                'spectral_radius': 0.0,
                'algebraic_connectivity': 0.0,
                'effective_resistance': None,
                'approximate': False,
            }
            self._spectrum_cache = result
            return result

        approximate = False
        note = None

        if n <= _SPECTRUM_SPARSE_THRESHOLD:
            # ---- 小图: 稠密精确分解 (完整谱) ----
            adj_spectrum = np.sort(nx.adjacency_spectrum(G).real)[::-1]
            lap_spectrum = np.sort(nx.laplacian_spectrum(G).real)
            norm_lap_spectrum = np.sort(nx.normalized_laplacian_spectrum(G).real)
        else:
            # ---- 大图: 稀疏近似 (无 sigma, 避免 shift-invert 的 LU 分解) ----
            approximate = True
            adj_spectrum, lap_spectrum, norm_lap_spectrum, note = self._sparse_spectrum(
                G, n, progress_callback)

        result = {
            'adjacency_eigenvalues': adj_spectrum,
            'laplacian_eigenvalues': lap_spectrum,
            'normalized_laplacian_eigenvalues': norm_lap_spectrum,
            'spectral_radius': float(np.max(np.abs(adj_spectrum))) if len(adj_spectrum) > 0 else 0.0,
            'algebraic_connectivity': float(lap_spectrum[1]) if len(lap_spectrum) > 1 else 0.0,
            'effective_resistance': None,
            'approximate': approximate,
            'n_nodes': n,
        }
        if note:
            result['spectrum_note'] = note

        # 有效电阻 (Kirchhoff 指数) 需要全部非零特征值; 谱不全时按已有部分估计并标注。
        if n > 1 and len(lap_spectrum) > 1:
            non_zero = lap_spectrum[lap_spectrum > 1e-10]
            if len(non_zero) > 0:
                result['effective_resistance'] = float(n * np.sum(1.0 / non_zero))
                result['effective_resistance_is_estimate'] = bool(approximate)

        self._spectrum_cache = result
        return result

    def _smallest_laplacian_eigs(self, L, n, k_max, progress_callback=None):
        """求拉普拉斯矩阵最小的若干特征值, 返回升序 ndarray (失败则空数组)。

        三条路线, 按"快且稳"排序, 并且**每条路线都做有效性校验** —— 宁可返回空数组
        让上层显示"未求出", 也不能返回一个错的特征值 (这比慢更糟)。

        背景 (2026-09 实测, 均为 153648 节点的图):
          1. shift-invert ``eigsh(..., sigma=...)``: 需要 (L-σI) 的稀疏 LU 分解,
             复杂度近 O(n^3)。n=20,000 就 185s, n=46,000 超过 600s 未完成。
             —— 这是旧版 topo_time 高达 18~265 秒的根因, 已弃用。
          2. ``eigsh(which='SA')``: 只看矩阵-向量乘, 但 λ2 与 0 很接近时收敛极慢
             (实测某个 15 万节点图上 k=4 跑 220s 仍未完成)。
          3. LOBPCG + Jacobi 预条件 (M = diag(1/d)): 实测比 'SA' 快 5~6 倍,
             但会在"λ2 紧贴 0"的图上收敛到明显偏小的错值
             (格点图案例: 给出 2e-05, 真值 5.06e-04)。
        所以: LOBPCG 打头, 校验不过就退 ARPACK, 再不行就放弃。
        """
        from scipy.sparse import diags
        from scipy.sparse.linalg import eigsh, lobpcg
        import warnings

        k = max(2, min(k_max, n - 2))
        diag = np.asarray(L.diagonal()).ravel()
        deg_min = float(diag.min()) if diag.size else 0.0

        def report(msg):
            if progress_callback:
                progress_callback(msg)

        def usable(vals):
            """校验: 最小特征值应≈0; λ2 必须为正且与 0 可分辨。"""
            if vals is None or len(vals) < 2:
                return False
            lam_min, lam2 = float(vals[0]), float(vals[1])
            if not np.isfinite(lam_min) or not np.isfinite(lam2):
                return False
            if abs(lam_min) > 1e-6:
                return False
            if lam2 <= 0:
                return False
            # λ2 相对 0 的可分辨度: 太接近就认为 LOBPCG 没算准
            if lam2 < max(1e-9, 1e-3 * deg_min):
                return False
            return True

        # ---- 路线 1: LOBPCG + Jacobi 预条件 ----
        try:
            report('laplacian(lobpcg)')
            rng = np.random.default_rng(0)
            X = rng.standard_normal((n, k))
            M = diags(1.0 / np.maximum(diag, 1e-9))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                vals, _ = lobpcg(L, X, M=M, largest=False, tol=1e-7,
                                 maxiter=max(200, 8 * k))
            vals = np.sort(np.asarray(vals, dtype=float).ravel())
            if usable(vals):
                return vals[:k]
        except Exception:
            pass

        # ---- 路线 2: ARPACK which='SA' (慢但稳), 必要时减半 k ----
        report('laplacian(arpack)')
        k_try = k
        while k_try >= 2:
            try:
                vals = eigsh(L, k=k_try, which='SA', return_eigenvectors=False,
                             tol=1e-4, maxiter=max(2000, 10 * n))
                vals = np.sort(vals.real)
                if usable(vals):
                    return vals[:k_try]
            except Exception:
                pass
            k_try //= 2

        return np.array([])

    def _sparse_spectrum(self, G, n, progress_callback=None):
        """大图稀疏特征值: 用 ARPACK 只算前 k 个, 全程不使用 sigma。

        返回 (adj_spectrum, lap_spectrum, norm_lap_spectrum, note)。
        求不到的谱一律返回空数组 (由调用方按空谱渲染), 不再用 [0.0, 0.0] 冒充。
        """
        from scipy.sparse.linalg import eigsh

        def report(msg):
            if progress_callback:
                progress_callback(msg)

        # k 随规模下降: ARPACK 每步迭代成本正比于 n, k 越大步数越多。
        if n <= 20000:
            k_cap = _SPECTRUM_K_SMALL_N
        elif n <= 60000:
            k_cap = _SPECTRUM_K_MID_N
        else:
            k_cap = _SPECTRUM_K_BIG_N

        notes = []

        # ---- 邻接矩阵: 最大模特征值 (谱半径), which='LM' 很快 ----
        A = nx.adjacency_matrix(G).astype(np.float64)
        adj_spectrum = np.array([])
        try:
            k_adj = min(n - 2, _SPECTRUM_K_LARGE if n > _SPECTRUM_SKIP_NORM_LAP else _SPECTRUM_K)
            adj_eigs = eigsh(A, k=k_adj, which='LM', return_eigenvectors=False, tol=1e-4)
            adj_spectrum = np.sort(adj_eigs.real)[::-1]
        except Exception:
            try:
                k2 = max(2, min(n - 2, k_cap))
                adj_eigs = eigsh(A, k=k2, which='LM', return_eigenvectors=False, tol=1e-3)
                adj_spectrum = np.sort(adj_eigs.real)[::-1]
            except Exception:
                notes.append('邻接谱未求出')
        del A

        # ---- 拉普拉斯矩阵: 最小特征值 (含 0 与 Fiedler 值 λ2) ----
        # 走 _smallest_laplacian_eigs: LOBPCG 打头 + 校验 + ARPACK 回退。
        # 绝不使用 sigma (shift-invert), 那会退化成 O(n^3) 的稀疏 LU 分解。
        report('laplacian')
        L = nx.laplacian_matrix(G).astype(np.float64)
        lap_spectrum = self._smallest_laplacian_eigs(L, n, k_cap, progress_callback)
        if len(lap_spectrum) == 0:
            notes.append('拉普拉斯谱未求出 (LOBPCG 校验未通过且 ARPACK 未收敛)')
        del L

        # ---- 归一化拉普拉斯 (大图跳过: 对研究价值有限且更慢) ----
        if n > _SPECTRUM_SKIP_NORM_LAP:
            norm_lap_spectrum = np.array([])
        else:
            NL = nx.normalized_laplacian_matrix(G).astype(np.float64)
            norm_lap_spectrum = self._smallest_laplacian_eigs(NL, n, k_cap, progress_callback)
            if len(norm_lap_spectrum) == 0:
                notes.append('归一化拉普拉斯谱未求出')
            del NL

        note = None
        if notes:
            note = '; '.join(notes)
        note = (note + '; ' if note else '') + (
            'n=%d 超过稠密阈值 %d, 谱为稀疏近似 (邻接取最大 %d 个 / 拉普拉斯取最小 %d 个)'
            % (n, _SPECTRUM_SPARSE_THRESHOLD, len(adj_spectrum), len(lap_spectrum)))
        return adj_spectrum, lap_spectrum, norm_lap_spectrum, note


    # ==================== 同调 ====================

    def compute_homology(self) -> Dict:
        G = self.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        beta0 = nx.number_connected_components(G)
        beta1 = m - n + beta0
        euler_characteristic = n - m

        return {
            'n_nodes': n,
            'n_edges': m,
            'beta_0': beta0,
            'beta_1': beta1,
            'euler_characteristic': euler_characteristic,
            'cyclomatic_number': beta1,
            'forest': beta1 == 0,
            'tree': beta1 == 0 and beta0 == 1 and n > 0,
            'unicyclic': beta1 == 1 and beta0 == 1,
        }

    # ==================== 环结构 ====================

    def compute_cycles(self) -> Dict:
        G = self.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()
        beta0 = nx.number_connected_components(G)
        beta1 = m - n + beta0

        result = {
            'girth': None,
            'longest_shortest_path': None,
            'cycle_basis_size': beta1,
            'cycle_lengths': [],
            'mean_cycle_length': None,
            'skipped': False,
        }

        # 大图或环数过多时跳过完整环基 (minimum_cycle_basis 极慢)
        if n > _CYCLE_BASIS_NODE_LIMIT or beta1 > _CYCLE_BASIS_BETA1_LIMIT:
            result['skipped'] = True
            # 仍然计算 girth (快得多, O(N*M))
            try:
                if n > 0 and nx.is_connected(G):
                    if nx.is_tree(G):
                        result['girth'] = 'inf'
                    else:
                        result['girth'] = nx.girth(G)
            except Exception:
                pass
            return result

        try:
            cycle_basis = nx.minimum_cycle_basis(G)
            result['cycle_basis_size'] = len(cycle_basis)
            lengths = [len(c) for c in cycle_basis]
            result['cycle_lengths'] = sorted(lengths)
            if lengths:
                result['girth'] = int(min(lengths))
                result['mean_cycle_length'] = float(np.mean(lengths))
        except Exception:
            pass

        if result['girth'] is None:
            try:
                if n > 0 and nx.is_connected(G):
                    if nx.is_tree(G):
                        result['girth'] = 'inf'
                    else:
                        result['girth'] = nx.girth(G)
            except Exception:
                pass

        return result

    # ==================== 中心性 ====================

    def compute_centrality(self) -> Dict:
        G = self.graph
        n = G.number_of_nodes()
        if n == 0:
            return {}

        result = {}

        # 度中心性: O(N+M), 很快
        try:
            dc = nx.degree_centrality(G)
            dc_vals = list(dc.values())
            result['degree_centrality'] = {
                'mean': float(np.mean(dc_vals)),
                'max': float(np.max(dc_vals)),
                'top_nodes': self._top_k(dc, 5),
            }
        except Exception:
            pass

        # 介数中心性: O(k*(N+M)), k=采样数; 大图跳过 (太慢)
        if n < _CENTRALITY_SKIP_BETWEENNESS:
            try:
                bc_k = min(50, n) if n <= _CENTRALITY_SAMPLE_THRESHOLD else min(15, n)
                bc = nx.betweenness_centrality(G, k=bc_k)
                bc_vals = list(bc.values())
                result['betweenness_centrality'] = {
                    'mean': float(np.mean(bc_vals)),
                    'max': float(np.max(bc_vals)),
                    'top_nodes': self._top_k(bc, 5),
                }
            except Exception:
                pass
        else:
            result['betweenness_centrality'] = {
                'mean': 0.0, 'max': 0.0, 'top_nodes': [],
                'skipped': True,
            }

        # 接近中心性: O(N*(N+M)), 大图采样
        try:
            cc = self._compute_closeness(G, n)
            cc_vals = list(cc.values())
            result['closeness_centrality'] = {
                'mean': float(np.mean(cc_vals)),
                'max': float(np.max(cc_vals)),
                'top_nodes': self._top_k(cc, 5),
            }
        except Exception:
            pass

        # 特征向量中心性 & PageRank: 迭代法, 通常 OK
        try:
            ec = nx.eigenvector_centrality(G, max_iter=500, tol=1e-3)
            ec_vals = list(ec.values())
            result['eigenvector_centrality'] = {
                'mean': float(np.mean(ec_vals)),
                'max': float(np.max(ec_vals)),
                'top_nodes': self._top_k(ec, 5),
            }
        except Exception:
            pass

        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1e-4)
            pr_vals = list(pr.values())
            result['pagerank'] = {
                'mean': float(np.mean(pr_vals)),
                'max': float(np.max(pr_vals)),
                'top_nodes': self._top_k(pr, 5),
            }
        except Exception:
            pass

        return result

    def _compute_closeness(self, G, n):
        """大图采样计算接近中心性, 用 scipy C-level BFS 替代 NetworkX"""
        if n <= _CENTRALITY_SAMPLE_THRESHOLD:
            return nx.closeness_centrality(G)

        # 大图: 用 scipy BFS + 减采样
        nodes = list(G.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        adj_csr = nx.adjacency_matrix(G, nodelist=nodes).astype(np.int8)

        sample_size = min(_CENTRALITY_SAMPLE_LARGE, n)
        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=sample_size, replace=False)
        result = {}
        for idx in indices:
            dist = _scipy_bfs_distances(adj_csr, int(idx))
            reachable = dist[dist >= 0]
            tot = int(reachable.sum())
            n_reach = len(reachable)
            if tot > 0 and n_reach > 1:
                result[nodes[idx]] = (n_reach - 1) / tot
            else:
                result[nodes[idx]] = 0.0
        return result

    # ==================== 完整分析 (并行) ====================

    def full_analysis(self, progress_callback: Optional[Callable[[str], None]] = None) -> Dict:
        analysis = {}

        # 快速部分先做
        if progress_callback:
            progress_callback("homology")
        analysis['homology'] = self.compute_homology()

        if progress_callback:
            progress_callback("cycles")
        analysis['cycles'] = self.compute_cycles()

        # 慢部分并行 (numpy/scipy 底层释放 GIL, 线程池有效)
        def _do_spectrum():
            if progress_callback:
                progress_callback("spectrum")
            return self.compute_spectrum(progress_callback=progress_callback)

        def _do_centrality():
            if progress_callback:
                progress_callback("centrality")
            return self.compute_centrality()

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_spec = pool.submit(_do_spectrum)
            fut_cent = pool.submit(_do_centrality)

            try:
                analysis['spectrum'] = fut_spec.result()
            except Exception as e:
                analysis['spectrum_error'] = str(e)

            try:
                analysis['centrality'] = fut_cent.result()
            except Exception as e:
                analysis['centrality_error'] = str(e)

        return analysis

    @staticmethod
    def _top_k(score_dict: Dict, k: int = 5) -> List[Tuple[int, float]]:
        items = list(score_dict.items())
        items.sort(key=lambda x: x[1], reverse=True)
        top = items[:k]
        return [(f"<state_{i}>", float(s)) for i, (_, s) in enumerate(top)]
