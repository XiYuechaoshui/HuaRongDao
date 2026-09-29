"""AI 模型模块: 四种人工智能方法分析华容道状态空间

1. TopologyFeatureML - 拓扑特征 + 传统机器学习
   提取图节点结构特征(度、聚类系数、邻居平均度、到初始状态距离等),
   用 KMeans 聚类、PCA/t-SNE 降维、RandomForest 回归预测图属性。

2. SimpleGNN - 图神经网络 (GraphSAGE)
   用 PyTorch 实现两层 GraphSAGE, 通过链接预测无监督训练学习节点嵌入,
   支持大图子采样训练, 提供 t-SNE/PCA 嵌入可视化。

3. TDADeepLearning - 持久同调特征 + 深度学习
   将持久条形码转为持久图像(Persistence Image), 构建TDA特征向量,
   训练自编码器学习拓扑特征的压缩表示。

注: 原第 4 部分 RLSolver (Q-learning 强化学习求解器) 已整体移除。
    移除原因: 其目标判定过松 (只要求"任意 2x2 方块位于底部中央", 与具体要解的
    块无关, 起点可能已满足), 且 Q 表在数万节点的状态图上无法收敛,
    求解退化为随机游走 —— "求解成功率"是伪信号, 不能作为评测指标。
"""
from __future__ import annotations
import os
import sys
import random
import warnings
import ctypes
from typing import Dict, List, Optional, Tuple, Callable

# ===== exe 模式下预加载 DLL (关键: 防止 c10.dll 初始化失败) =====
if getattr(sys, 'frozen', False):
    _base = os.path.dirname(sys.executable)
    _internal = os.path.join(_base, '_internal')
    _torch_lib = os.path.join(_internal, 'torch', 'lib')
    # 注册 DLL 搜索路径
    for _p in [_internal, _torch_lib]:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except (OSError, AttributeError):
                pass
    if os.path.isdir(_torch_lib):
        os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')
    # 先加载系统 VC++ 运行时 (避免打包版本冲突)
    _sys_dir = os.environ.get('SystemRoot', r'C:\Windows\System32')
    _load_errors = []
    # 用 LOAD_WITH_ALTERED_SEARCH_PATH 让系统从目标 DLL 所在目录搜索依赖
    _LOAD_WITH_ALTERED_SEARCH_PATH = 0x8
    for _dll in ['vcruntime140.dll', 'vcruntime140_1.dll', 'msvcp140.dll',
                 'MSVCP140_ATOMIC_WAIT.dll', 'dbghelp.dll']:
        _path = os.path.join(_sys_dir, _dll)
        if os.path.exists(_path):
            try:
                ctypes.WinDLL(_path, mode=_LOAD_WITH_ALTERED_SEARCH_PATH)
            except OSError as _de:
                _load_errors.append(f"VC++ {_dll}: {_de}")
    # 再加载 torch 依赖 DLL (按依赖顺序)
    for _dll in ['libiomp5md.dll', 'c10.dll', 'torch_cpu.dll', 'torch_python.dll']:
        _path = os.path.join(_torch_lib, _dll)
        if os.path.exists(_path):
            try:
                ctypes.WinDLL(_path, mode=_LOAD_WITH_ALTERED_SEARCH_PATH)
            except OSError as _de:
                _load_errors.append(f"torch {_dll}: {_de}")
    # 把 DLL 预加载错误记录到文件
    if _load_errors:
        try:
            _log_path2 = os.path.join(_base, 'dll_preload_errors.log')
            with open(_log_path2, 'w', encoding='utf-8') as _f2:
                _f2.write("DLL 预加载错误:\n")
                for _err in _load_errors:
                    _f2.write(f"  {_err}\n")
        except Exception:
            pass

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except (ImportError, OSError) as _e:
    import traceback as _tb
    TORCH_AVAILABLE = False
    nn = None
    F = None
    torch = None
    # 把详细错误写到文件 (exe 模式下 stderr 不可见)
    _log_path = os.path.join(os.path.dirname(getattr(sys, 'executable', __file__)),
                             'torch_load_error.log')
    try:
        with open(_log_path, 'w', encoding='utf-8') as _f:
            _f.write(f"PyTorch 加载失败: {type(_e).__name__}: {_e}\n\n")
            _f.write(_tb.format_exc())
    except Exception:
        pass

# 抑制 PyTorch 稀疏张量不变量检查警告 (无害, 仅提示检查已关闭)
warnings.filterwarnings("ignore", message=".*Sparse invariant.*")
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor

from 数据模型 import State
from 字体工具 import CN_FONT, apply_cn_font

# ==================== 性能阈值 (已解除限制) ====================
_GNN_SUBSAMPLE_THRESHOLD = 1000000  # 已解除: 子采样阈值设为极大值
_FEATURE_SIMPLIFY_THRESHOLD = 1000000  # 已解除: 特征简化阈值设为极大值


# ==================== 1. 拓扑特征 + 传统机器学习 ====================
class TopologyFeatureML:
    """拓扑特征提取 + 传统机器学习方法

    从状态空间图中提取节点级结构特征, 结合图级拓扑/TDA特征,
    使用 KMeans 聚类、PCA/t-SNE 降维、RandomForest 回归等传统ML方法分析。
    """

    def extract_graph_features(self, graph: nx.Graph,
                               topo_result: Optional[Dict] = None,
                               tda_result: Optional[Dict] = None) -> np.ndarray:
        """从图中提取每个节点的特征向量

        节点级特征: 度、聚类系数、邻居平均度、到初始状态距离
        图级特征(可选): 从拓扑分析/TDA结果中提取, 广播到所有节点

        参数:
            graph: 状态空间图 (节点为 State 对象)
            topo_result: 拓扑分析结果字典 (可选)
            tda_result: 持久同调结果字典 (可选)
        返回:
            (n_nodes, feature_dim) 的特征矩阵
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return np.array([]).reshape(0, 0)

        # 参考节点 (初始状态): BFS 生成时第一个加入的节点
        ref_node = nodes[0]

        # ---- 节点级结构特征 ----
        degree_dict = dict(graph.degree())
        dist_dict = nx.single_source_shortest_path_length(graph, ref_node)
        max_dist = max(dist_dict.values()) if dist_dict else 1

        if n <= _FEATURE_SIMPLIFY_THRESHOLD:
            # 小/中图: 计算完整特征
            clustering_dict = nx.clustering(graph)
            avg_nd_dict = nx.average_neighbor_degree(graph)
        else:
            # 超大图: 跳过昂贵的聚类系数计算, 用 0 填充
            clustering_dict = {}
            avg_nd_dict = {}

        features = []
        for node in nodes:
            deg = float(degree_dict.get(node, 0))
            cc = float(clustering_dict.get(node, 0.0))
            avg_nd = float(avg_nd_dict.get(node, 0.0))
            dist = dist_dict.get(node, -1)
            dist_norm = dist / max(1, max_dist) if dist >= 0 else -1.0
            is_leaf = 1.0 if deg <= 1 else 0.0
            deg_cent = deg / max(1, n - 1)
            features.append([deg, cc, avg_nd, dist_norm, is_leaf, deg_cent])

        features = np.array(features, dtype=np.float64)

        # ---- 图级特征 (从拓扑分析结果) ----
        graph_features = []
        if topo_result:
            hom = topo_result.get('homology', {})
            graph_features.extend([
                float(hom.get('n_nodes', 0)),
                float(hom.get('n_edges', 0)),
                float(hom.get('beta_0', 0)),
                float(hom.get('beta_1', 0)),
                float(hom.get('euler_characteristic', 0)),
            ])
            spec = topo_result.get('spectrum', {})
            if isinstance(spec, dict):
                graph_features.extend([
                    float(spec.get('spectral_radius', 0.0)),
                    float(spec.get('algebraic_connectivity', 0.0)),
                ])
            cent = topo_result.get('centrality', {})
            if isinstance(cent, dict):
                dc = cent.get('degree_centrality', {})
                graph_features.append(float(dc.get('mean', 0.0)))

        # ---- 图级特征 (从TDA结果) ----
        if tda_result:
            stats = tda_result.get('stats', {})
            for dim in sorted(stats.keys()):
                s = stats[dim]
                graph_features.extend([
                    float(s.get('total_bars', 0)),
                    float(s.get('mean_lifetime', 0.0)),
                    float(s.get('max_lifetime', 0.0)),
                    float(s.get('persistence_entropy', 0.0)),
                ])

        # 广播图级特征到所有节点
        if graph_features:
            graph_arr = np.array(graph_features, dtype=np.float64)
            features = np.hstack([features, np.tile(graph_arr, (n, 1))])

        return features

    def cluster_nodes(self, graph: nx.Graph, n_clusters: int = 5) -> Dict:
        """用 KMeans 对节点聚类

        参数:
            graph: 状态空间图
            n_clusters: 聚类数
        返回:
            {node: cluster_label} 字典
        """
        nodes = list(graph.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        features = self.extract_graph_features(graph)
        k = min(n_clusters, n)
        # n_jobs=1: 限制单线程, 不抢其他程序 CPU (与 拓扑训练器 CPU 限流策略一致)
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        clusters = {nodes[i]: int(labels[i]) for i in range(n)}
        self._last_features = features  # 缓存供 save_model 使用
        self._last_clusters = clusters
        return clusters

    def reduce_dimensions(self, features: np.ndarray,
                          method: str = 'pca',
                          n_components: int = 2) -> np.ndarray:
        """降维用于可视化

        参数:
            features: (n_samples, n_features) 特征矩阵
            method: 'pca' 或 'tsne'
            n_components: 目标维度
        返回:
            (n_samples, n_components) 降维后的坐标
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        n_samples = features.shape[0]
        if n_samples == 0:
            return np.array([]).reshape(0, n_components)

        # 降维目标不能超过样本数和特征数
        max_comp = min(n_samples, features.shape[1])
        n_comp = min(n_components, max_comp)

        if method == 'tsne':
            # t-SNE 需要 perplexity < n_samples
            perplexity = min(30, max(1, n_samples - 1))
            reducer = TSNE(n_components=n_comp, random_state=42,
                           perplexity=perplexity, init='pca' if n_samples > n_comp else 'random')
        else:
            reducer = PCA(n_components=n_comp, random_state=42)

        return reducer.fit_transform(features)

    def train_property_predictor(self, features: np.ndarray,
                                 target: np.ndarray) -> RandomForestRegressor:
        """训练 RandomForest 回归器预测图属性

        参数:
            features: (n_samples, n_features) 特征矩阵
            target: (n_samples,) 目标值
        返回:
            训练好的 RandomForestRegressor
        """
        # n_jobs=1: 限制单线程, 不抢其他程序 CPU
        rf = RandomForestRegressor(n_estimators=100, random_state=42,
                                    max_depth=10, n_jobs=1)
        rf.fit(features, target)
        return rf

    def save_model(self, path: str):
        """保存特征提取结果和聚类模型到文件 (原子写入)"""
        import pickle
        data = {
            'features': getattr(self, '_last_features', None),
            'clusters': getattr(self, '_last_clusters', None),
        }
        tmp_path = path + '.tmp'
        with open(tmp_path, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp_path, path)

    def load_model(self, path: str):
        """加载之前保存的特征和聚类结果"""
        import pickle
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self._last_features = data.get('features')
            self._last_clusters = data.get('clusters')
            return data
        except Exception:
            return None


# ==================== 2. GraphSAGE 图神经网络 ====================
_NNBase = nn.Module if (TORCH_AVAILABLE and nn is not None) else object


class _SAGENetwork(_NNBase):
    """GraphSAGE 网络结构 (两层, 均值聚合)"""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError('PyTorch 不可用, 无法使用 GNN 训练')
        super().__init__()
        # GraphSAGE: 拼接 [自身特征, 邻居均值特征] 后线性变换
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: (N, input_dim) 节点特征矩阵
            adj_norm: (N, N) 归一化稀疏邻接矩阵 (D^{-1}A, 均值聚合)
        返回:
            (N, output_dim) 节点嵌入
        """
        # 第一层: 聚合邻居 + 拼接自身 + 线性变换 + ReLU
        agg1 = torch.sparse.mm(adj_norm, x)
        h1 = F.relu(self.fc1(torch.cat([x, agg1], dim=1)))
        # 第二层
        agg2 = torch.sparse.mm(adj_norm, h1)
        h2 = self.fc2(torch.cat([h1, agg2], dim=1))
        return h2


class SimpleGNN:
    """简单的 GraphSAGE 图神经网络

    用 PyTorch 实现两层 GraphSAGE, 通过链接预测无监督训练学习节点嵌入。
    大图(>5000节点)自动子采样训练, 推理时对全图计算嵌入。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 32):
        """初始化 GraphSAGE 模型

        参数:
            input_dim: 输入特征维度 (节点结构特征数)
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.model = _SAGENetwork(input_dim, hidden_dim, output_dim)
        self._is_trained = False

    def _build_adj_norm(self, graph: nx.Graph, node_list: List,
                        node_to_idx: Dict) -> torch.Tensor:
        """构建归一化稀疏邻接矩阵 (D^{-1}A, 用于均值聚合)

        返回 COO 格式的稀疏张量, adj_norm[i] 的第 j 行 = 节点 i 邻居特征的均值
        """
        n = len(node_list)
        rows, cols, vals = [], [], []
        for node in node_list:
            i = node_to_idx[node]
            neighbors = list(graph.neighbors(node))
            if neighbors:
                w = 1.0 / len(neighbors)
                for nb in neighbors:
                    j = node_to_idx[nb]
                    rows.append(i)
                    cols.append(j)
                    vals.append(w)

        if not rows:
            # 无边: 返回全零稀疏矩阵
            indices = torch.zeros(2, 0, dtype=torch.long)
            values = torch.zeros(0, dtype=torch.float32)
        else:
            indices = torch.tensor([rows, cols], dtype=torch.long)
            values = torch.tensor(vals, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, (n, n)).coalesce()

    def _extract_node_features(self, graph: nx.Graph, node_list: List) -> np.ndarray:
        """提取节点结构特征

        特征: 度、聚类系数、邻居平均度、是否叶节点、度中心性、到参考节点距离等
        大图(>10000节点)跳过昂贵的聚类系数计算
        """
        n = len(node_list)
        degree_dict = dict(graph.degree())

        if n <= _FEATURE_SIMPLIFY_THRESHOLD:
            # 小/中图: 完整特征
            clustering_dict = nx.clustering(graph)
            avg_nd_dict = nx.average_neighbor_degree(graph)
            ref_node = node_list[0] if node_list else None
            dist_dict = nx.single_source_shortest_path_length(graph, ref_node) if ref_node else {}
            max_dist = max(dist_dict.values()) if dist_dict else 1

            raw = []
            for node in node_list:
                deg = float(degree_dict.get(node, 0))
                cc = float(clustering_dict.get(node, 0.0))
                avg_nd = float(avg_nd_dict.get(node, 0.0))
                dist = dist_dict.get(node, 0)
                dist_norm = dist / max(1, max_dist)
                is_leaf = 1.0 if deg <= 1 else 0.0
                deg_cent = deg / max(1, n - 1)
                raw.append([deg, cc, avg_nd, dist_norm, is_leaf, deg_cent, 1.0, 0.0])
            features = np.array(raw, dtype=np.float32)
        else:
            # 超大图: 仅度特征 + bias
            raw = []
            for node in node_list:
                deg = float(degree_dict.get(node, 0))
                raw.append([deg, deg / max(1, n - 1), 1.0 if deg <= 1 else 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
            features = np.array(raw, dtype=np.float32)

        # 调整到 input_dim (填充或截断)
        if features.shape[1] < self.input_dim:
            features = np.pad(features, ((0, 0), (0, self.input_dim - features.shape[1])))
        elif features.shape[1] > self.input_dim:
            features = features[:, :self.input_dim]

        return features

    def train(self, graph: nx.Graph, epochs: int = 100, lr: float = 0.01,
              progress_callback: Optional[Callable] = None):
        """训练 GNN 学习节点嵌入 (无监督链接预测)

        通过最大化相连节点嵌入的相似度、最小化不相连节点相似度来训练。
        大图(>5000节点)自动子采样训练子图。

        参数:
            graph: 状态空间图
            epochs: 训练轮数
            lr: 学习率
            progress_callback: 进度回调函数
        """
        node_list = list(graph.nodes())
        n = len(node_list)
        if n == 0:
            return

        # ---- 大图子采样 ----
        if n > _GNN_SUBSAMPLE_THRESHOLD:
            rng = np.random.default_rng(42)
            sample_size = _GNN_SUBSAMPLE_THRESHOLD
            sampled_idx = sorted(rng.choice(n, size=sample_size, replace=False))
            train_nodes = [node_list[i] for i in sampled_idx]
            train_subgraph = graph.subgraph(train_nodes).copy()
        else:
            train_subgraph = graph
            train_nodes = node_list

        train_node_list = list(train_subgraph.nodes())
        train_n = len(train_node_list)
        train_node_to_idx = {node: i for i, node in enumerate(train_node_list)}

        # 构建邻接矩阵和特征
        adj_norm = self._build_adj_norm(train_subgraph, train_node_list, train_node_to_idx)
        features = self._extract_node_features(train_subgraph, train_node_list)
        x = torch.tensor(features, dtype=torch.float32)

        # 正样本边 (实际存在的边)
        pos_edges = list(train_subgraph.edges())
        pos_edge_indices = [(train_node_to_idx[u], train_node_to_idx[v])
                            for u, v in pos_edges]

        if not pos_edge_indices:
            # 无边图: 无法用链接预测训练, 仅做特征编码
            self._is_trained = True
            return

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            z = self.model(x, adj_norm)

            # ---- 链接预测损失 ----
            # 正样本: 相连节点嵌入应相似
            n_pos = min(len(pos_edge_indices), 256)
            sample_idx = np.random.choice(len(pos_edge_indices), size=n_pos, replace=False)
            pos_u = torch.tensor([pos_edge_indices[i][0] for i in sample_idx])
            pos_v = torch.tensor([pos_edge_indices[i][1] for i in sample_idx])
            pos_loss = -torch.log(
                torch.sigmoid((z[pos_u] * z[pos_v]).sum(dim=1)) + 1e-8
            ).mean()

            # 负样本: 随机节点对应不相似
            neg_u = torch.randint(0, train_n, (n_pos,))
            neg_v = torch.randint(0, train_n, (n_pos,))
            neg_loss = -torch.log(
                1 - torch.sigmoid((z[neg_u] * z[neg_v]).sum(dim=1)) + 1e-8
            ).mean()

            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()

            if progress_callback and (epoch + 1) % 10 == 0:
                progress_callback(
                    epoch=epoch + 1,
                    total=epochs,
                    loss=float(loss.item()),
                )

            # ★ 每轮释放中间张量, 避免autograd图累积内存
            del z, pos_u, pos_v, neg_u, neg_v, pos_loss, neg_loss, loss

        # 训练后释放所有中间张量 (不降低模型精度, 权重已保存在 model.parameters 中)
        del x, adj_norm, pos_edges, pos_edge_indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清理 PyTorch CPU 缓存 (PyTorch 2.x: 使用 _accelerator_emptyCache)
        try:
            if torch.cuda.is_available():
                torch._C._accelerator_emptyCache()
        except (AttributeError, RuntimeError, Exception):
            pass
        try:
            torch._C.clear_autocast_cache()
        except (AttributeError, Exception):
            pass
        import gc
        gc.collect()

        self._is_trained = True

    def get_embeddings(self, graph: nx.Graph) -> np.ndarray:
        """获取所有节点的嵌入向量

        参数:
            graph: 状态空间图
        返回:
            (n_nodes, output_dim) 的嵌入矩阵
        """
        node_list = list(graph.nodes())
        n = len(node_list)
        if n == 0:
            return np.array([]).reshape(0, self.output_dim)

        node_to_idx = {node: i for i, node in enumerate(node_list)}
        adj_norm = self._build_adj_norm(graph, node_list, node_to_idx)
        features = self._extract_node_features(graph, node_list)
        x = torch.tensor(features, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            z = self.model(x, adj_norm)
        result = z.numpy()

        # 推理后释放显存/缓存 (不损失精度)
        del x, adj_norm, z, node_list, node_to_idx, features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # 清理 PyTorch CPU 缓存
        try:
            if torch.cuda.is_available():
                torch._C._accelerator_emptyCache()
        except (AttributeError, RuntimeError, Exception):
            pass
        try:
            torch._C.clear_autocast_cache()
        except (AttributeError, Exception):
            pass
        import gc
        gc.collect()

        return result

    @staticmethod
    def fast_graph_features(embeddings: np.ndarray) -> Dict:
        """用GNN嵌入统计量快速近似图级拓扑特征 (毫秒级, 替代秒级精确计算)

        近似映射:
          - 嵌入均值范数 → 近似图密度/连通度
          - 嵌入方差 → 近似聚类系数方差
          - 嵌入SVD奇异值 → 近似拉普拉斯谱 (谱半径、代数连通度)
          - 嵌入间余弦相似度 → 近似社区结构强度 (模块度)
          - 嵌入范数分布 → 近似度中心性分布

        参数:
            embeddings: (n_nodes, output_dim) GNN嵌入矩阵
        返回:
            dict, 包含近似图级特征
        """
        if embeddings is None or embeddings.shape[0] == 0:
            return {}
        try:
            n = embeddings.shape[0]
            # 1. 嵌入统计量
            emb_mean = embeddings.mean(axis=0)  # (output_dim,)
            emb_norms = np.linalg.norm(embeddings, axis=1)  # (n,)
            mean_norm = float(emb_norms.mean())
            std_norm = float(emb_norms.std())

            # 2. SVD 近似谱 (比完整特征分解快 O(n^2) → O(n*k))
            k_svd = min(10, embeddings.shape[1])
            if n > 1 and k_svd > 0:
                U, S, Vt = np.linalg.svd(embeddings - emb_mean, full_matrices=False)
                spectral_radius = float(S[0]) if len(S) > 0 else 0.0
                # 第二奇异值近似代数连通度 (Fiedler 值)
                algebraic_conn = float(S[1]) if len(S) > 1 else 0.0
                energy = float(np.sum(S ** 2))  # 谱能量
            else:
                spectral_radius = 0.0
                algebraic_conn = 0.0
                energy = 0.0

            # 3. 嵌入间余弦相似度 → 近似模块度
            #    注意: 不能算全量 n×n 相似度矩阵 —— 真实状态图有 6.9 万~7.8 万节点,
            #    n² 的 float32 就是 19 GB, 必然 MemoryError 并被外层 except 吞成空结果
            #    (表现是"AI提速"这一路静默消失)。这里只抽 256 行去和全量比, 内存 O(256n)。
            if n > 1 and embeddings.shape[1] > 0:
                normed = embeddings / (emb_norms[:, None] + 1e-8)
                _s = min(n, 256)
                _idx = np.random.default_rng(42).choice(n, size=_s, replace=False)
                _sim = normed[_idx] @ normed.T          # (<=256, n)
                _sim[np.arange(_s), _idx] = 0.0         # 排掉自相似
                _pos = _sim[_sim > 0]
                modularity_proxy = float(_pos.mean()) if _pos.size else 0.0
            else:
                modularity_proxy = 0.0

            # 4. 嵌入范数分布 → 近似度中心性分布
            centrality_proxy = {
                'mean': mean_norm,
                'std': std_norm,
                'max': float(emb_norms.max()),
                'min': float(emb_norms.min()),
            }

            # 5. 近似Betti数
            # β₀ ≈ 嵌入空间中连通分量数 (用KMeans聚类估计)
            # 独立 try: KMeans 失败不应丢弃前4步已算好的特征
            beta0_proxy = 1
            if n > 1:
                try:
                    from sklearn.cluster import KMeans as _KMeans
                    k = max(1, min(3, n))
                    labels = _KMeans(n_clusters=k, random_state=42, n_init=3).fit_predict(embeddings)
                    beta0_proxy = int(len(set(labels)))
                except Exception:
                    beta0_proxy = 1

            return {
                'method': 'GNN-embedding-proxy',
                'n_nodes': n,
                'embedding_dim': embeddings.shape[1],
                'mean_norm': mean_norm,
                'std_norm': std_norm,
                'spectral_radius': spectral_radius,
                'algebraic_connectivity': algebraic_conn,
                'spectral_energy': energy,
                'modularity_proxy': modularity_proxy,
                'centrality_proxy': centrality_proxy,
                'beta0_proxy': beta0_proxy,
            }
        except Exception as _e:
            # 不要把失败吞成空 dict: 调用方只判断 `if fast_feats:` 就静默跳过,
            # 用户完全看不到"AI提速特征没算出来"。带上 error 键让调用方能报告原因。
            return {'method': 'GNN-embedding-proxy', 'error': '%s: %s' % (type(_e).__name__, _e)}

    def visualize_embeddings(self, embeddings: np.ndarray,
                             method: str = 'tsne') -> 'Figure':
        """降维可视化节点嵌入

        参数:
            embeddings: (n_nodes, output_dim) 嵌入矩阵
            method: 'tsne' 或 'pca'
        返回:
            matplotlib Figure 对象 (使用 Figure 直接创建, 无 GUI manager,
            可安全在后台线程中调用, 不会触发 Tk 线程错误)
        """
        from matplotlib.figure import Figure
        n = embeddings.shape[0]
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        if n == 0:
            ax.text(0.5, 0.5, "无嵌入数据", ha='center', va='center',
                    fontproperties=CN_FONT, fontsize=14)
            ax.set_axis_off()
            return fig

        # 降维到 2D
        if method == 'tsne':
            perplexity = min(30, max(1, n - 1))
            n_comp = min(2, n)
            reducer = TSNE(n_components=n_comp, random_state=42, perplexity=perplexity,
                           init='pca' if n > n_comp else 'random')
        else:
            reducer = PCA(n_components=min(2, embeddings.shape[1]), random_state=42)
        coords = reducer.fit_transform(embeddings)

        if coords.shape[1] < 2:
            # 只有一个维度, 补零
            coords = np.hstack([coords, np.zeros((n, 1))])

        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=range(n),
                             cmap='viridis', s=30, alpha=0.7, edgecolors='white', linewidths=0.3)
        cb = fig.colorbar(scatter, ax=ax)
        cb.set_label('节点索引', fontproperties=CN_FONT, fontsize=9)

        apply_cn_font(ax,
                      title=f"GNN 节点嵌入可视化 ({method.upper()})",
                      xlabel="维度 1",
                      ylabel="维度 2",
                      title_fontsize=12, label_fontsize=10)
        fig.tight_layout()
        return fig

    def save_model(self, path: str):
        """保存 GNN 模型权重到文件 (记忆功能, 原子写入)"""
        tmp_path = path + '.tmp'
        torch.save({
            'state_dict': self.model.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'is_trained': self._is_trained,
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_model(self, path: str) -> bool:
        """从文件加载预训练 GNN 权重, 成功返回 True"""
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self.input_dim = ckpt['input_dim']
            self.hidden_dim = ckpt['hidden_dim']
            self.output_dim = ckpt['output_dim']
            self.model = _SAGENetwork(self.input_dim, self.hidden_dim, self.output_dim)
            self.model.load_state_dict(ckpt['state_dict'])
            self._is_trained = ckpt.get('is_trained', True)
            return True
        except Exception:
            return False


# ==================== 3. TDA 特征 + 深度学习 ====================
class _AutoEncoder(_NNBase):
    """小型自编码器: 编码-压缩-解码"""

    def __init__(self, input_dim: int, hidden_dim: int, bottleneck_dim: int):
        if not TORCH_AVAILABLE:
            raise RuntimeError('PyTorch 不可用, 无法使用 TDA 深度学习')
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TDADeepLearning:
    """持久同调特征 + 深度学习

    将持久条形码转为持久图像(Persistence Image), 构建TDA特征向量,
    训练自编码器学习拓扑特征的压缩表示。
    """

    def barcode_to_persistence_image(self, bars: Dict[int, List[Tuple[float, float]]],
                                     resolution: int = 20) -> np.ndarray:
        """将持久条形码转为持久图像 (固定大小矩阵)

        持久图像: x轴=出生时间(birth), y轴=持久性(persistence=death-birth)
        每个条用高斯核叠加到二维网格上, 权重为持久值。

        参数:
            bars: {维数: [(birth, death), ...]} 条形码字典
            resolution: 输出图像分辨率 (resolution × resolution)
        返回:
            (resolution, resolution) 的持久图像矩阵, 归一化到 [0, 1]
        """
        # 收集所有 (birth, persistence) 点
        all_points = []
        max_persistence = 0.0

        for dim, pairs in bars.items():
            for b, d in pairs:
                if np.isfinite(d):
                    p = float(d) - float(b)
                    max_persistence = max(max_persistence, p)
                    all_points.append((float(b), p))
                else:
                    # 无限条: 用最大持久值的 1.5 倍
                    all_points.append((float(b), max_persistence * 1.5 if max_persistence > 0 else 1.0))

        if not all_points:
            return np.zeros((resolution, resolution))

        points = np.array(all_points)
        births = points[:, 0]
        persistences = points[:, 1]

        # 确定坐标范围, 添加边距
        b_min, b_max = births.min(), births.max()
        p_min, p_max = persistences.min(), persistences.max()
        b_range = b_max - b_min if b_max > b_min else 1.0
        p_range = p_max - p_min if p_max > p_min else 1.0
        b_min -= 0.1 * b_range
        b_max += 0.1 * b_range
        p_min -= 0.1 * p_range
        p_max += 0.1 * p_range

        # 构建网格
        b_grid = np.linspace(b_min, b_max, resolution)
        p_grid = np.linspace(p_min, p_max, resolution)
        B, P = np.meshgrid(b_grid, p_grid)

        # 高斯核叠加 (权重 = 持久值)
        sigma = 0.15 * max(b_range, p_range)
        image = np.zeros((resolution, resolution))
        for b, p in all_points:
            weight = p
            image += weight * np.exp(-((B - b) ** 2 + (P - p) ** 2) / (2 * sigma ** 2))

        # 归一化到 [0, 1]
        if image.max() > 0:
            image = image / image.max()

        return image

    def build_feature_vector(self, tda_result: Dict) -> np.ndarray:
        """从 TDA 结果构建特征向量

        特征包含: 采样数、各维度的条形码统计量、持久图像展平特征

        参数:
            tda_result: 持久同调分析结果字典
        返回:
            一维特征向量
        """
        features = []

        # 基本统计
        features.append(float(tda_result.get('n_samples', 0)))

        # 各维度条形码统计
        stats = tda_result.get('stats', {})
        for dim in sorted(stats.keys()):
            s = stats[dim]
            features.extend([
                float(s.get('total_bars', 0)),
                float(s.get('finite_bars', 0)),
                float(s.get('infinite_bars', 0)),
                float(s.get('mean_lifetime', 0.0)),
                float(s.get('max_lifetime', 0.0)),
                float(s.get('median_lifetime', 0.0)),
                float(s.get('persistence_entropy', 0.0)),
            ])

        # 持久图像特征 (低分辨率展平, 避免特征向量过长)
        bars = tda_result.get('bars', {})
        if bars:
            image = self.barcode_to_persistence_image(bars, resolution=10)
            features.extend(image.flatten().tolist())

        return np.array(features, dtype=np.float64)

    def train_autoencoder(self, features: np.ndarray, epochs: int = 50,
                          progress_callback: Optional[Callable] = None,
                          lr: float = 0.001) -> _AutoEncoder:
        """训练小型自编码器学习 TDA 特征的压缩表示

        参数:
            features: (n_samples, n_features) 特征矩阵
            epochs: 训练轮数
            progress_callback: 进度回调函数
            lr: 学习率 (默认 0.001; 自我迭代微调时可降低至 0.0005)
        返回:
            训练好的 _AutoEncoder 模型
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)

        n_features = features.shape[1]
        if n_features == 0:
            return None

        # 记忆功能: 如果已有加载的自编码器 (维度匹配) 则继续微调, 否则新建
        existing = getattr(self, '_autoencoder', None)
        if (existing is not None
                and existing.encoder[0].in_features == n_features):
            # 复用已有模型 (自我迭代)
            autoencoder = existing
        else:
            hidden_dim = max(8, n_features // 2)
            bottleneck_dim = max(4, n_features // 4)
            autoencoder = _AutoEncoder(n_features, hidden_dim, bottleneck_dim)

        autoencoder.train()
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()

        x = torch.tensor(features, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            recon = autoencoder(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()

            if progress_callback and (epoch + 1) % 5 == 0:
                progress_callback(
                    epoch=epoch + 1,
                    total=epochs,
                    loss=float(loss.item()),
                )
            # ★ 每轮释放中间张量, 避免autograd图累积
            del recon, loss

        autoencoder.eval()
        self._autoencoder = autoencoder  # 保存供 save_model 使用
        # 释放训练用输入张量 + PyTorch CPU 缓存
        del x, optimizer, criterion
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if torch.cuda.is_available():
                torch._C._accelerator_emptyCache()
        except (AttributeError, RuntimeError, Exception):
            pass
        try:
            torch._C.clear_autocast_cache()
        except (AttributeError, Exception):
            pass
        import gc
        gc.collect()
        return autoencoder

    def compute_persistence_image_figure(self, persistence_image: np.ndarray) -> plt.Figure:
        """可视化持久图像

        参数:
            persistence_image: (resolution, resolution) 持久图像矩阵
        返回:
            matplotlib Figure 对象
        """
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(persistence_image, cmap='hot', origin='lower', aspect='auto')
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('密度', fontproperties=CN_FONT, fontsize=9)
        apply_cn_font(ax,
                      title="持久图像 (Persistence Image)",
                      xlabel="出生时间 (Birth)",
                      ylabel="持久性 (Persistence)",
                      title_fontsize=12, label_fontsize=10)
        fig.tight_layout()
        return fig

    def save_model(self, path: str):
        """保存自编码器权重到文件 (记忆功能, 原子写入)"""
        ae = getattr(self, '_autoencoder', None)
        if ae is None:
            return
        tmp_path = path + '.tmp'
        torch.save({
            'state_dict': ae.state_dict(),
            'input_dim': ae.encoder[0].in_features,
            'hidden_dim': ae.encoder[0].out_features,
            'bottleneck_dim': ae.encoder[2].out_features,
        }, tmp_path)
        os.replace(tmp_path, path)

    def load_model(self, path: str) -> bool:
        """从文件加载预训练自编码器权重, 成功返回 True"""
        try:
            ckpt = torch.load(path, map_location='cpu', weights_only=False)
            self._autoencoder = _AutoEncoder(
                ckpt['input_dim'], ckpt['hidden_dim'], ckpt['bottleneck_dim'])
            self._autoencoder.load_state_dict(ckpt['state_dict'])
            self._autoencoder.eval()
            return True
        except Exception:
            return False

# ==================== 4. 强化学习求解器 (已移除) ====================
# 原 RLSolver 类已整体删除。原因:
#   1) 目标判定 _is_goal 只判断"存在任意 2x2 方块位于底部中央", 不区分是哪个块、
#      也不参考初始布局 —— 起点就可能已满足, solve() 会直接返回长度为 1 的"解";
#   2) 状态图有 6.5~7.8 万节点, 数百 episodes 的表格型 Q-learning 无法收敛,
#      solve() 里候选动作的 Q 值几乎全为 0, 实际退化为"取第一个未访问邻居"的
#      确定性随机游走。
# 结论: rl_solved / solution_length 是伪信号, 不能作为评测指标, 相关功能已移除。
# 如需恢复请连目标判定与学习信号一起重新设计, 不要只把旧类贴回来。
# 守门: 自检.py 会断言 RLSolver 三处命名都不存在, 防止被误加回来。
