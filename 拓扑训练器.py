"""华容道 拓扑训练器 — 训练流水线 + 独立启动器

只训练原程序的【拓扑学分析】部分, 不含任何 RL 求解:
  1. 代数拓扑: Betti 数 (β₀ 连通分支、β₁ 独立环数)、Euler 示性数 χ、cyclomatic number
  2. 谱图理论: 拉普拉斯/邻接矩阵特征值谱、谱半径、代数连通度 (Fiedler 值 λ₂)、有效电阻
  3. 环结构: 围长 Girth、最小环基、平均环长度
  4. 中心性: 度/介数/接近/特征向量/PageRank (采样近似)
  5. TDA 持久同调: H₀/H₁ 条形码、持久图像 (Persistence Image)、自编码器压缩表示
  6. 社区检测: Louvain 贪心模块度 / GNN 嵌入 KMeans 替代 (n≥800)
  7. AI 模型: 拓扑特征 ML 聚类 (KMeans + PCA/t-SNE)、GNN GraphSAGE 嵌入、TDA 深度学习自编码器

本文件 = 训练流水线 (被 GUI 与主程序共用) + 独立窗口入口:
  • GUI 在 训练器面板.py 的 TrainerPanel (ttk.Frame);
  • 它同时被主界面左侧「训练」页 和 本文件 main() 使用 —— 只有一份 UI 代码, 不会漂移;
  • 模型库 (含 模型\\ 子目录的目录) 的新建/管理在 模型管理面板.py, 对应主界面「模型」页。

特性:
  • 棋盘 2x2 ~ 6x6 随机 (每个布局独立尺寸), 块配置智能自适应
  • 自我迭代: 加载当前模型库 模型\\ 下已有 GNN / TDA-DL / ML 模型继续微调
  • 状态空间过大 (超过 max_nodes) 时停止 BFS, 但所有分析/AI 在部分图上继续运行
  • CPU 占用率上限可调 (默认 50%, 不跑满)
"""
import os
import sys
import gc
import time
import json
import pickle
import random
import threading
import tempfile
from typing import Dict, List, Optional, Tuple
from collections import Counter

# ==================== CPU 占用限制 (必须在 numpy/torch/sklearn 导入前设置) ====================
# 默认用一半 CPU 核 (不跑满), 通过环境变量限制 BLAS/MKL/OpenMP 后端线程数
_CPU_LIMIT_RATIO = 0.5  # CPU 使用比例 (0.5=半数核, 0.25=1/4核)
_n_cpu = max(1, int(os.cpu_count() * _CPU_LIMIT_RATIO))
os.environ.setdefault('OMP_NUM_THREADS', str(_n_cpu))
os.environ.setdefault('MKL_NUM_THREADS', str(_n_cpu))
os.environ.setdefault('OPENBLAS_NUM_THREADS', str(_n_cpu))
os.environ.setdefault('NUMEXPR_NUM_THREADS', str(_n_cpu))
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', str(_n_cpu))

import numpy as np
import networkx as nx

# Matplotlib Agg 后端 (GUI 内嵌用 Figure)
import matplotlib
matplotlib.use('TkAgg')

# PyTorch 线程数限制 + 内存碎片化缓解
import torch
torch.set_num_threads(_n_cpu)  # ★ 关键: 限制 PyTorch CPU 推理/训练线程数
try:
    torch.backends.cudnn.benchmark = False  # 不优化卷积, 降低显存碎片
    torch.backends.cudnn.deterministic = True
except Exception:
    pass

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

from 数据模型 import State, Block, LayoutData
from 状态空间 import StateSpaceGenerator
from 拓扑分析 import TopologyAnalyzer
from 持久同调 import TDAnalyzer
from 社区检测 import CommunityAnalyzer
from AI模型 import TopologyFeatureML, SimpleGNN, TDADeepLearning


# ==================== 资源占用优化工具 ====================

def _tda_stat(tda_result, dim: int, key: str):
    """从 TDA 结果的按维数分组 stats 里安全取值。

    compute_persistence 返回的 stats 形如 {0: {...}, 1: {...}} (键是 int);
    经 JSON 往返后键会变成字符串 '0'/'1' (缓存结果缓存/会话存档都会发生),
    所以这里两种键都认。取不到返回 None。
    """
    if not isinstance(tda_result, dict):
        return None
    stats = tda_result.get('stats')
    if not isinstance(stats, dict):
        return None
    entry = stats.get(dim)
    if entry is None:
        entry = stats.get(str(dim))
    if not isinstance(entry, dict):
        return None
    return entry.get(key)


def _lower_process_priority():
    """将当前进程优先级降到 BELOW_NORMAL, 不抢其他程序 CPU

    Windows 优先级档位:
        IDLE=64  → 几乎不抢 CPU (后台服务用)
        BELOW_NORMAL=16384 → 比普通程序低一档 (推荐用于 AI 训练后台)
        NORMAL=32 (默认)
    本程序选 BELOW_NORMAL, 既不卡其他程序, 自己也能稳定跑满空闲CPU。
    """
    if not sys.platform.startswith('win'):
        return
    try:
        import ctypes
        # 0x00004000 = BELOW_NORMAL_PRIORITY_CLASS
        ctypes.windll.kernel32.SetPriorityClass(
            ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:
        pass


class CPULimiter:
    """主动 CPU 占用率限流器

    通过周期性采样本进程 CPU 占用率, 超过阈值就主动 sleep 让出 CPU 时间片。
    不依赖底层线程数限制 (那个只能限制并行度, 不能限制峰值占用率),
    本类实现"软性"占用率上限, 例如设 50% → 长期来看 CPU 占用率不超过 50%。
    """
    def __init__(self, target_ratio: float = 0.5, check_interval: float = 0.3):
        """
        参数:
            target_ratio: 目标 CPU 占用率 (0~1)
                         0.5 = 50% (默认, 推荐用于不抢其他程序)
                         0.3 = 30% (保守模式, 老电脑推荐)
                         0.7 = 70% (激进模式, 接近全速)
            check_interval: 检查间隔 (秒), 默认 0.3s
        """
        self.target_ratio = max(0.05, min(0.95, target_ratio))
        self.check_interval = check_interval
        self._last_check = time.time()
        self._cpu_busy = False  # 上次采样时 CPU 是否超阈值

    def throttle(self):
        """在工作循环中调用: 主动让出 CPU 时间片, 实现长期占用率上限"""
        now = time.time()
        elapsed = now - self._last_check
        if elapsed < self.check_interval:
            return  # 间隔不到, 不检查也不 sleep
        self._last_check = now

        # 用 Windows GetProcessTimes 读取本进程 CPU 时间 (用户态+内核态)
        cpu_percent = self._get_process_cpu_percent()
        ncores = os.cpu_count() or 1
        # CPU 占用率相对单核: 例如 4 核机器, 100% 利用率 = 400% (一个进程)
        # 我们要的 target_ratio 是"相对总 CPU"的占用, 所以归一化到 0~1
        norm_usage = cpu_percent / (100.0 * ncores)

        if norm_usage > self.target_ratio:
            # 超过目标, 主动 sleep 一段时间让出 CPU
            # sleep 时长 = 检查间隔 × (超标比例), 上限 0.3s
            over_ratio = norm_usage / max(0.01, self.target_ratio)
            sleep_t = min(0.3, self.check_interval * (over_ratio - 1.0))
            if sleep_t > 0.001:
                self._cpu_busy = True
                time.sleep(sleep_t)
        else:
            self._cpu_busy = False

    def _get_process_cpu_percent(self) -> float:
        """获取本进程 CPU 占用率 (Windows: 用 GetProcessTimes)"""
        if not sys.platform.startswith('win'):
            return 0.0
        try:
            import ctypes
            from ctypes import c_void_p, c_uint64, c_uint32
            kernel32 = ctypes.windll.kernel32
            hProc = kernel32.GetCurrentProcess()

            class FILETIME(ctypes.Structure):
                _fields_ = [('dwLowDateTime', c_uint32), ('dwHighDateTime', c_uint32)]

            ft_create, ft_exit, ft_kernel, ft_user = (FILETIME(), FILETIME(),
                                                      FILETIME(), FILETIME())
            # 转换为 100ns 单位的总 CPU 时间
            def to_int64(ft):
                return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

            # 采样两次 (间隔很短), 计算增量
            kernel32.GetProcessTimes(hProc, ctypes.byref(ft_create),
                                     ctypes.byref(ft_exit),
                                     ctypes.byref(ft_kernel),
                                     ctypes.byref(ft_user))
            cpu_t1 = to_int64(ft_kernel) + to_int64(ft_user)
            wall_t1 = time.time()
            time.sleep(0.05)  # 短暂等待测差值
            kernel32.GetProcessTimes(hProc, ctypes.byref(ft_create),
                                     ctypes.byref(ft_exit),
                                     ctypes.byref(ft_kernel),
                                     ctypes.byref(ft_user))
            cpu_t2 = to_int64(ft_kernel) + to_int64(ft_user)
            wall_t2 = time.time()
            # CPU 时间是 100ns 单位, 转秒
            cpu_dt = (cpu_t2 - cpu_t1) / 1e7
            wall_dt = wall_t2 - wall_t1
            if wall_dt <= 0:
                return 0.0
            # 占用率 (一个进程, 单核=100%)
            return 100.0 * cpu_dt / wall_dt
        except Exception:
            return 0.0


def _release_memory(graph=None):
    """显式释放一批大对象引用 + PyTorch 缓存

    关键: 不降低 AI 性能 (模型权重已保存到磁盘, 这里只释放运行时中间张量)。
    """
    # 1. 释放图对象 (这是最大的内存占用源, 几万节点+边可达数百MB)
    if graph is not None:
        try:
            graph.clear()
        except Exception:
            pass

    # 2. 清空 PyTorch CUDA 缓存 (CPU 模式下也有少量缓存可释放)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

    # 3. 触发 Python 垃圾回收 (回收 graph/张量等大对象)
    gc.collect()



# ==================== 布局生成器 (同自动训练.py, 无修改) ====================

_BLOCK_TEMPLATES = [
    {'size': (2, 2), 'weight': 1.2, 'min_board': (2, 2)},
    {'size': (2, 1), 'weight': 1.0, 'min_board': (1, 2)},
    {'size': (1, 2), 'weight': 1.0, 'min_board': (2, 1)},
    {'size': (3, 1), 'weight': 0.5, 'min_board': (1, 3)},
    {'size': (1, 3), 'weight': 0.5, 'min_board': (3, 1)},
    {'size': (1, 1), 'weight': 1.5, 'min_board': (1, 1)},
]


def _available_block_templates(rows, cols, rng):
    c = []
    for t in _BLOCK_TEMPLATES:
        w, h = t['size']
        mr, mc = t['min_board']
        if rows >= mr and cols >= mc:
            c.append((t['size'], t['weight']))
    rng.shuffle(c)
    c.sort(key=lambda x: -x[1])
    return [s for s, _ in c]


def random_board_size(rng, min_dim=2, max_dim=6):
    rows = rng.randint(min_dim, max_dim)
    cols = rng.randint(min_dim, max_dim)
    if rng.random() < 0.7 and cols > rows:
        rows, cols = cols, rows
    return rows, cols


class DynamicLayoutGenerator:
    def __init__(self, max_nodes=200000, min_dim=2, max_dim=6,
                 min_empty_cells=2, max_empty_cells=4):
        self.max_nodes = max_nodes
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.min_empty = min_empty_cells
        self.max_empty = max_empty_cells

    def generate(self, seed=None):
        rng = random.Random(seed)
        for _ in range(200):
            rows, cols = random_board_size(rng, self.min_dim, self.max_dim)
            r = self._try(rows, cols, rng)
            if r is not None:
                return r
        return None

    def _try(self, rows, cols, rng):
        total = rows * cols
        target_empty = min(self.max_empty, max(self.min_empty, total // 5))
        for _ in range(80):
            grid = [[None] * cols for _ in range(rows)]
            blocks = {}
            bid = 0
            filled = 0
            for (w, h) in _available_block_templates(rows, cols, rng):
                area = w * h
                m = max(1, (total // area) * 2)
                for _ in range(m):
                    if filled + area + target_empty > total:
                        break
                    if self._place(grid, blocks, bid, w, h, rng):
                        filled += area
                        bid += 1
            while True:
                rem = total - filled
                if rem <= target_empty:
                    break
                if self._place(grid, blocks, bid, 1, 1, rng):
                    filled += 1
                    bid += 1
                else:
                    break
            final_empty = total - filled
            if final_empty < self.min_empty or bid < 1:
                continue
            return LayoutData(rows, cols, blocks)
        return None

    def _place(self, grid, blocks, bid, w, h, rng):
        rows = len(grid); cols = len(grid[0])
        if rows < h or cols < w:
            return False
        for _ in range(500):
            r = rng.randint(0, rows - h)
            c = rng.randint(0, cols - w)
            if all(grid[r+dr][c+dc] is None for dr in range(h) for dc in range(w)):
                coords = [(r+dr, c+dc) for dr in range(h) for dc in range(w)]
                blocks[bid] = coords
                for dr in range(h):
                    for dc in range(w):
                        grid[r+dr][c+dc] = bid
                return True
        return False


# ==================== 单布局拓扑训练流水线 (纯拓扑, 无 RL) ====================

def topology_train_pipeline(layout: LayoutData, idx: int, output_dir: str,
                            gnn_epochs: int = 200, resume: bool = True,
                            max_nodes: int = 200000,
                            log_fn=None,
                            cpu_limiter: Optional['CPULimiter'] = None) -> Optional[dict]:
    """单布局完整拓扑学训练流水线

    返回: 结果字典 (所有拓扑指标 + AI 结果 + 耗时)
    """
    def _t():
        """每步之间调用: CPU 占用率限流 (不跑满)"""
        if cpu_limiter is not None:
            cpu_limiter.throttle()
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg, flush=True)

    model_dir = os.path.join(output_dir, '模型')
    os.makedirs(model_dir, exist_ok=True)

    # 1. 状态 + 校验
    blocks = [Block(bid, coords) for bid, coords in layout.blocks.items()]
    state = State(blocks, (layout.rows, layout.cols))
    if not state.is_valid():
        log("  ✗ 布局不合法")
        return None

    log(f"[{idx:04d}] {layout.rows}x{layout.cols} 棋盘, {len(layout.blocks)} 块")

    # 2. 生成状态空间 (超限截断但保留部分图, 尽力而为)
    _t()
    t0 = time.time()
    gen = StateSpaceGenerator()

    def _mon(**kw):
        if kw.get('states', 0) > max_nodes:
            gen.should_stop = True
    try:
        graph = gen.generate(state, progress_callback=_mon)
    except MemoryError:
        log("  ✗ 内存不足")
        return None
    gen_t = time.time() - t0
    partial = bool(gen.should_stop)
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    if n_nodes < 2:
        log("  ✗ 图太小 (<2节点), 跳过")
        return None
    tag = " [部分图, 超限截断]" if partial else ""
    log(f"  状态空间: {n_nodes}节点 / {n_edges}边 ({gen_t:.1f}s){tag}")

    # 3. 拓扑分析 (TopologyAnalyzer.full_analysis)
    _t()
    t0 = time.time()
    topo = TopologyAnalyzer(graph)
    try:
        topo_result = topo.full_analysis()
    except Exception as e:
        log(f"  ⚠ 拓扑分析异常: {e}")
        topo_result = {'error': str(e), 'homology': {}, 'spectrum': {}, 'cycles': {}, 'centrality': {}}
    topo_t = time.time() - t0
    hom = topo_result.get('homology', {})
    spc = topo_result.get('spectrum', {})
    cyc = topo_result.get('cycles', {})
    fiedler = spc.get('algebraic_connectivity')
    chi = hom.get('euler_characteristic')
    log(f"  拓扑: β₀={hom.get('beta_0')}, β₁={hom.get('beta_1')}, χ={chi}, "
        f"λ₂={fiedler}, girth={cyc.get('girth')} ({topo_t:.1f}s)")
    # ★ 分阶段释放: 拓扑分析器内部缓存了大图中间结果, 完成后立即释放
    del topo
    gc.collect()

    # 4. TDA 持久同调
    _t()
    t0 = time.time()
    tda = TDAnalyzer(graph)
    if n_nodes >= 50000:
        max_samples = 50
    elif n_nodes >= 10000:
        max_samples = 100
    else:
        max_samples = 200
    try:
        tda_result = tda.compute_persistence(max_samples=max_samples, max_dim=1)
    except Exception as e:
        log(f"  ⚠ TDA异常: {e}")
        tda_result = {'bars': {}, 'stats': {}, 'n_samples': 0, 'error': str(e)}
    tda_t = time.time() - t0
    bars = tda_result.get('bars', {})
    log(f"  TDA: H₀={len(bars.get(0, []))}, H₁={len(bars.get(1, []))}, "
        f"max_samples={max_samples} ({tda_t:.1f}s)")
    # ★ 分阶段释放: TDA分析器内部缓存了采样矩阵等大对象
    del tda
    gc.collect()

    # 5. 社区检测
    # 注: 不再用 {'n_communities': 1, 'modularity': 0.0} 兜底 —— 那个常量恰好等于
    # "图连通且无社区结构"的真实结果, 会把失败伪装成正常结果。失败会留下 degraded 标记。
    _t()
    t0 = time.time()
    comm = CommunityAnalyzer(graph)
    try:
        comm_result = comm.detect_communities()
    except Exception as e:
        log(f"  ⚠ 社区检测异常: {type(e).__name__}: {e}")
        comm_result = {'n_communities': 0, 'modularity': None,
                       'degraded': True,
                       'diagnostic': f'{type(e).__name__}: {e}'}
    if comm_result.get('degraded'):
        log(f"  ⚠ 社区检测降级: {comm_result.get('diagnostic')}")
    comm_t = time.time() - t0
    # ★ 分阶段释放: 社区检测器内部缓存了模块度计算结果
    del comm
    gc.collect()

    # 6. 拓扑特征 ML (提取特征 + 聚类)
    _t()
    t0 = time.time()
    ml = TopologyFeatureML()
    try:
        features = ml.extract_graph_features(graph, topo_result, tda_result)
        n_clusters = min(5, max(1, n_nodes))
        clusters = ml.cluster_nodes(graph, n_clusters=n_clusters)
    except Exception as e:
        log(f"  ⚠ ML聚类异常: {e}")
        features = np.zeros((n_nodes, 8), dtype=np.float64)
        clusters = {n: 0 for n in graph.nodes()}
    # 保存ML (全局累积)
    try:
        ml.save_model(os.path.join(model_dir, 'ml_global.pkl'))
    except Exception:
        pass
    ml_t = time.time() - t0
    log(f"  ML: 特征{features.shape}, {len(set(clusters.values()))}簇 "
        f"| 模块度Q={comm_result.get('modularity', 0):.3f}, "
        f"社区={comm_result.get('n_communities', 0)} ({ml_t + comm_t:.1f}s)")
    # 注意: ml 不在此处释放, 因为逐布局保存仍需要引用 ml.save_model()
    # 但可以清理 PyTorch/numpy 中间缓存, 降低峰值内存
    gc.collect()

    # 7. GNN GraphSAGE 嵌入 (自我迭代: 先加载再微调)
    _t()
    t0 = time.time()
    gnn = SimpleGNN(input_dim=8, hidden_dim=64, output_dim=32)
    gnn_path = os.path.join(model_dir, 'gnn_global.pt')
    loaded_gnn = False
    if resume and os.path.exists(gnn_path):
        loaded_gnn = gnn.load_model(gnn_path)
        if loaded_gnn:
            pass  # 日志稍后打印
    epochs_here = max(5, gnn_epochs if n_nodes < 10000 else gnn_epochs // 2)
    try:
        gnn.train(graph, epochs=epochs_here, lr=0.005 if loaded_gnn else 0.01,
                  progress_callback=lambda **kw: None)
    except Exception as e:
        log(f"  ⚠ GNN训练异常: {e}")
    try:
        gnn.save_model(gnn_path)
    except Exception:
        pass
    try:
        embeddings = gnn.get_embeddings(graph)
    except Exception:
        embeddings = np.zeros((n_nodes, 32), dtype=np.float64)
    gnn_t = time.time() - t0
    load_tag = " (自我迭代)" if loaded_gnn else ""
    log(f"  GNN: 嵌入{embeddings.shape}{load_tag} ({gnn_t:.1f}s, {epochs_here}epochs)")
    # 注意: gnn 不在此处释放, 因为逐布局保存仍需要引用 gnn.save_model()
    # 但可以清理 PyTorch 缓存, 降低峰值内存 (gnn.train 内部已释放中间张量)
    _release_memory()

    # 8. TDA 深度学习自编码器 (自我迭代)
    _t()
    t0 = time.time()
    tda_dl = TDADeepLearning()
    loaded_ae = False
    try:
        feat_vec = tda_dl.build_feature_vector(tda_result)
        tda_ae_path = os.path.join(model_dir, 'tda_ae_global.pt')
        if resume and os.path.exists(tda_ae_path) and feat_vec.shape[0] > 0:
            loaded_ae = tda_dl.load_model(tda_ae_path)
            ae = getattr(tda_dl, '_autoencoder', None)
            if loaded_ae and ae is not None:
                if ae.encoder[0].in_features != feat_vec.shape[0]:
                    loaded_ae = False
        ae_epochs = 50 if n_nodes < 10000 else 20
        lr_here = 0.0005 if loaded_ae else 0.001
        try:
            tda_dl.train_autoencoder(feat_vec.reshape(1, -1), epochs=ae_epochs,
                                     progress_callback=lambda **kw: None,
                                     lr=lr_here)
        except Exception as e:
            log(f"  ⚠ TDA-DL训练异常: {e}")
        try:
            tda_dl.save_model(tda_ae_path)
        except Exception:
            pass
    except Exception as e:
        log(f"  ⚠ TDA-DL异常: {e}")
        feat_vec = np.array([])
    dl_t = time.time() - t0
    load_tag2 = " (自我迭代)" if loaded_ae else ""
    log(f"  TDA-DL: 特征{feat_vec.shape}{load_tag2} ({dl_t:.1f}s)")

    # 9. 保存逐布局结果 (必须在分阶段释放之前, 因为保存模型需要引用 gnn/tda_dl/ml 对象)
    save_dir = os.path.join(output_dir, f'puzzle_{idx:04d}')
    os.makedirs(save_dir, exist_ok=True)

    layout_path = os.path.join(save_dir, 'layout.json')
    tmp_p = layout_path + '.tmp'
    with open(tmp_p, 'w', encoding='utf-8') as f:
        json.dump(layout.to_dict(), f, indent=2, ensure_ascii=False)
    os.replace(tmp_p, layout_path)
    try:
        gpath = os.path.join(save_dir, 'graph.pkl')
        tmp_g = gpath + '.tmp'
        with open(tmp_g, 'wb') as f:
            pickle.dump(graph, f)
        os.replace(tmp_g, gpath)
    except Exception:
        pass
    try:
        np.save(os.path.join(save_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(save_dir, 'features.npy'), features)
    except Exception:
        pass
    try:
        gnn.save_model(os.path.join(save_dir, 'gnn_model.pt'))
        tda_dl.save_model(os.path.join(save_dir, 'tda_ae_model.pt'))
        ml.save_model(os.path.join(save_dir, 'ml_model.pkl'))
    except Exception:
        pass

    # ★ 分阶段释放: 所有模型已保存到磁盘, 这里释放运行时对象
    del ml, gnn, tda_dl
    _release_memory()

    cent = topo_result.get('centrality', {})

    result = {
        'idx': idx,
        'rows': layout.rows, 'cols': layout.cols,
        'n_blocks': len(layout.blocks),
        'n_nodes': n_nodes, 'n_edges': n_edges,
        'was_partial': partial,
        'gen_time': round(gen_t, 2),
        'topo_time': round(topo_t, 2),
        'tda_time': round(tda_t, 2),
        'comm_time': round(comm_t, 2),
        'ml_time': round(ml_t, 2),
        'gnn_time': round(gnn_t, 2),
        'dl_time': round(dl_t, 2),
        'total_time': round(gen_t + topo_t + tda_t + comm_t + ml_t + gnn_t + dl_t, 2),
        # 拓扑指标
        'beta_0': hom.get('beta_0'),
        'beta_1': hom.get('beta_1'),
        'euler_char': chi,
        'algebraic_connectivity': fiedler,
        'spectral_radius': spc.get('spectral_radius'),
        'effective_resistance': spc.get('effective_resistance'),
        'girth': cyc.get('girth') if not isinstance(cyc.get('girth'), str) else None,
        'mean_cycle_length': cyc.get('mean_cycle_length'),
        'n_communities': comm_result.get('n_communities'),
        'modularity': round((comm_result.get('modularity') or 0), 4),
        'community_method': comm_result.get('method'),
        'community_degraded': bool(comm_result.get('degraded')),
        'community_note': comm_result.get('note'),        'size_entropy': comm_result.get('size_entropy'),
        # TDA 摘要 (stats 是按维数分组的 {0: {...}, 1: {...}}, 键名是 mean_lifetime 等;
        # 历史版本这里误读 stats['mean_lifespan_H0'/'H1'], 导致两项恒为 null)
        'tda_h0': len(bars.get(0, [])),
        'tda_h1': len(bars.get(1, [])),
        'tda_samples': tda_result.get('n_samples', 0),
        'tda_mean_h0_life': _tda_stat(tda_result, 0, 'mean_lifetime'),
        'tda_mean_h1_life': _tda_stat(tda_result, 1, 'mean_lifetime'),
        # AI 形状
        'features_shape': list(features.shape),
        'embeddings_shape': list(embeddings.shape),
        'tda_dl_shape': list(feat_vec.shape),
        # 自我迭代标记
        'resumed_gnn': loaded_gnn,
        'resumed_tda_dl': loaded_ae,
    }

    rp = os.path.join(save_dir, 'result.json')
    tmp_r = rp + '.tmp'
    with open(tmp_r, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp_r, rp)

    log(f"  ✓ 完成 | 总耗时 {result['total_time']:.1f}s")

    # ★ 释放本布局的运行时大对象 (模型已存盘, 这里只清运行时缓存)
    #   graph / features / embeddings 是本布局特有的, 函数返回后不再需要
    #   注意: topo/tda/comm/ml/gnn/tda_dl 已在分阶段释放中删除, 此处只需清理剩余对象
    del features, embeddings, feat_vec
    _release_memory(graph)

    return result


# ==================== 独立启动器 (GUI 在 训练器面板.py) ====================
# 原先这个文件里有一整套 GUI (class TopologyTrainerApp, 约 430 行), 且只能独立开窗。
# 现在 GUI 抽到 训练器面板.py 的 TrainerPanel (ttk.Frame):
#   • 主界面「训练」页 与 独立窗口 共用同一份面板代码, 不会出现两套 UI 各自漂移;
#   • 本文件只保留训练流水线 (DynamicLayoutGenerator / topology_train_pipeline / CPULimiter
#     / 资源工具) 与这个独立窗口入口。

def main():
    """独立窗口启动 (内容与主界面「训练」页完全一致)"""
    import 训练器面板
    try:
        from 布局哈希 import get_model_base
        初始输出 = get_model_base() or '训练数据'
    except Exception:
        初始输出 = '训练数据'
    训练器面板.run_standalone(初始输出目录=初始输出)


if __name__ == '__main__':
    main()
