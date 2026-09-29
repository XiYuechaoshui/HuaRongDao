"""华容道 AI 自动训练程序 (增强版)

特性:
  1. 棋盘大小自适应随机: 2x2 ~ 6x6, 每个布局独立随机
  2. 块配置智能随机: 根据棋盘尺寸动态决定块数量/形状/大小, 确保至少 2 个空格
  3. 状态空间过大时尽力运行: 达到节点上限后 BFS 停止, 但 AI 在部分图上继续训练 (不跳过)
  4. AI 自我迭代: 优先加载 训练数据/模型/ 下已有的 GNN / TDA-DL 模型, 在其基础上继续训练 (累积记忆, 不断优化)

用法:
    python 自动训练.py                      # 默认生成 50 个布局
    python 自动训练.py --n 200              # 生成 200 个 (积累记忆)
    python 自动训练.py --max-nodes 500000   # 进一步放宽节点上限
    python 自动训练.py --no-resume          # 禁用自我迭代 (从0开始重新训练)
"""
import argparse
import json
import os
import pickle
import random
import sys
import time
from typing import Optional, List, Dict, Tuple

import numpy as np
import networkx as nx

# Matplotlib 非交互后端, 避免线程问题
import matplotlib
matplotlib.use('Agg')

import torch

from 数据模型 import State, Block, LayoutData
from 状态空间 import StateSpaceGenerator
from 拓扑分析 import TopologyAnalyzer
from 持久同调 import TDAnalyzer
from 社区检测 import CommunityAnalyzer
from AI模型 import TopologyFeatureML, SimpleGNN, TDADeepLearning
from 布局哈希 import compute_layout_hash


# ==================== 棋盘尺寸与块配置策略 ====================

# 可接受的块形状: (宽, 高, 权重, 最小棋盘(rows,cols))
#   权重越大被优先放置的概率越高
_BLOCK_TEMPLATES = [
    # 曹操级 (2x2, 大块)
    {'size': (2, 2), 'weight': 1.2, 'min_board': (2, 2)},
    # 大将级 (横/竖矩形)
    {'size': (2, 1), 'weight': 1.0, 'min_board': (1, 2)},
    {'size': (1, 2), 'weight': 1.0, 'min_board': (2, 1)},
    {'size': (3, 1), 'weight': 0.5, 'min_board': (1, 3)},
    {'size': (1, 3), 'weight': 0.5, 'min_board': (3, 1)},
    # 卒级 (1x1)
    {'size': (1, 1), 'weight': 1.5, 'min_board': (1, 1)},
]


def _available_block_templates(rows: int, cols: int, rng: random.Random) -> List[Tuple[int, int]]:
    """根据棋盘尺寸筛选可放置的块形状, 按权重随机排序 (放前面的优先尝试放)"""
    candidates = []
    for t in _BLOCK_TEMPLATES:
        w, h = t['size']
        min_r, min_c = t['min_board']
        if rows >= min_r and cols >= min_c:
            candidates.append((t['size'], t['weight']))
    # 按权重带随机采样, 得到优先级顺序 (保证大块优先)
    # Fisher-Yates 洗牌后按权重排序
    rng.shuffle(candidates)
    candidates.sort(key=lambda x: -x[1])
    return [s for s, _ in candidates]


def random_board_size(rng: random.Random,
                      min_dim: int = 2, max_dim: int = 6) -> Tuple[int, int]:
    """随机选择棋盘尺寸 (非方形概率更高, 更贴近真实华容道比例)"""
    rows = rng.randint(min_dim, max_dim)
    cols = rng.randint(min_dim, max_dim)
    # 70% 概率保证 rows >= cols (华容道通常竖长)
    if rng.random() < 0.7 and cols > rows:
        rows, cols = cols, rows
    return rows, cols


class DynamicLayoutGenerator:
    """按棋盘尺寸智能生成随机合法布局 (块不固定, 自适应)"""

    def __init__(self, max_nodes: int = 200000,
                 min_dim: int = 2, max_dim: int = 6,
                 min_empty_cells: int = 2, max_empty_cells: int = 4):
        self.max_nodes = max_nodes
        self.min_dim = min_dim
        self.max_dim = max_dim
        self.min_empty = min_empty_cells
        self.max_empty = max_empty_cells

    def generate(self, seed: Optional[int] = None) -> Optional[LayoutData]:
        """生成一个布局: 随机尺寸 + 随机块配置"""
        rng = random.Random(seed)
        for _ in range(200):
            rows, cols = random_board_size(rng, self.min_dim, self.max_dim)
            layout = self._try_generate_for_size(rows, cols, rng)
            if layout is not None:
                return layout
        return None

    def _try_generate_for_size(self, rows: int, cols: int,
                                rng: random.Random) -> Optional[LayoutData]:
        """对给定棋盘尺寸尝试生成一个合法布局"""
        total_cells = rows * cols
        # 目标空格数: 2 ~ 4 (取决于棋盘总大小)
        target_empty = min(self.max_empty, max(self.min_empty, total_cells // 5))
        # 放置块直到剩余空格足够
        for _ in range(80):
            grid = [[None] * cols for _ in range(rows)]
            blocks: Dict[int, list] = {}
            bid = 0
            filled = 0
            # 每个模板尝试放若干次
            templates = _available_block_templates(rows, cols, rng)
            for (w, h) in templates:
                block_area = w * h
                # 尽可能多地放这个尺寸的块, 直到空间不够或填满
                max_tries_per_size = max(1, (total_cells // block_area) * 2)
                placed_this_size = 0
                for _ in range(max_tries_per_size):
                    remaining_empty = total_cells - filled - len(blocks) * 0 + bid * 0  # placeholder
                    remaining_empty = total_cells - filled
                    # 放置后还要保留 target_empty 个空格, 放不下就停止这个尺寸
                    if filled + block_area + target_empty > total_cells:
                        break
                    if self._place_block(grid, blocks, bid, w, h, rng):
                        filled += block_area
                        bid += 1
                        placed_this_size += 1
                    else:
                        # 放不下去就换个位置再试几次 (由 max_tries_per_size 控制)
                        continue
            # 填充 1x1 直到达到空格目标
            while True:
                remaining_empty = total_cells - filled
                if remaining_empty <= target_empty:
                    break
                if self._place_block(grid, blocks, bid, 1, 1, rng):
                    filled += 1
                    bid += 1
                else:
                    break
            # 最终校验: 必须有至少 min_empty 个空格, 且至少放了 1 个块
            final_empty = total_cells - filled
            if final_empty < self.min_empty or bid < 1:
                continue
            return LayoutData(rows, cols, blocks)
        return None

    def _place_block(self, grid, blocks, bid, w, h, rng) -> bool:
        """在 grid 上随机放置一个 w*h 的块, 返回是否成功"""
        rows = len(grid)
        cols = len(grid[0])
        if rows < h or cols < w:
            return False
        for _ in range(500):
            r = rng.randint(0, rows - h)
            c = rng.randint(0, cols - w)
            if all(grid[r + dr][c + dc] is None
                   for dr in range(h) for dc in range(w)):
                coords = [(r + dr, c + dc) for dr in range(h) for dc in range(w)]
                blocks[bid] = coords
                for dr in range(h):
                    for dc in range(w):
                        grid[r + dr][c + dc] = bid
                return True
        return False


# ==================== 单布局完整流水线 (带自我迭代) ====================

def run_single_puzzle(layout: LayoutData, idx: int, output_dir: str,
                      gnn_epochs: int = 200,
                      resume: bool = True) -> Optional[dict]:
    """对单个布局运行完整分析 + AI 训练 (支持自我迭代)

    resume=True 时, 会先加载 output_dir/模型/ 下已有的缓存模型,
    在已有权重的基础上继续训练, 实现 AI 自我迭代与累积记忆。

    如果状态空间超过 max_nodes, 会停止 BFS 生成部分图, 但 AI 仍继续在该部分图上运行 (尽力而为)。
    """
    print(f"\n{'=' * 60}")
    print(f"[布局 {idx:04d}] {layout.rows}x{layout.cols} 棋盘, "
          f"{len(layout.blocks)} 块")

    model_dir = os.path.join(output_dir, '模型')
    os.makedirs(model_dir, exist_ok=True)

    # ---- 1. 创建 State 并校验 ----
    blocks = [Block(bid, coords) for bid, coords in layout.blocks.items()]
    state = State(blocks, (layout.rows, layout.cols))
    if not state.is_valid():
        print("  ✗ 布局不合法 (重叠或越界)")
        return None

    # ---- 2. 生成状态空间 (超限停止但保留部分图, 不跳过后续 AI) ----
    t0 = time.time()
    gen = StateSpaceGenerator()
    max_nodes = getattr(run_single_puzzle, '_max_nodes', 200000)

    def _monitor(**kw):
        if kw.get('states', 0) > max_nodes:
            gen.should_stop = True

    try:
        graph = gen.generate(state, progress_callback=_monitor)
    except MemoryError:
        print("  ✗ 状态空间内存不足, 放弃此布局")
        return None
    gen_time = time.time() - t0
    was_partial = bool(gen.should_stop)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    partial_tag = " [部分图, 节点已达上限]" if was_partial else ""
    print(f"  状态空间: {n_nodes} 节点, {n_edges} 边 ({gen_time:.1f}s){partial_tag}")

    if n_nodes < 2:
        print("  ✗ 状态空间太小 (<2节点), 无法进行图分析, 跳过")
        return None

    # ---- 3. 拓扑分析 (小图全量, 大图尽力) ----
    # 大图 (节点过多) 时, spectrum/centrality 极慢且内存爆炸, 跳过避免 OOM。
    # 只保留 homology + cycles (秒算), spectrum/centrality 置空 dict,
    # 使 ML 特征维度保持不变 (AI模型.extract_graph_features 对空 dict 填充默认 0)。
    _TOPO_FULL_NODE_LIMIT = 10000
    t0 = time.time()
    topo = TopologyAnalyzer(graph)
    try:
        if n_nodes > _TOPO_FULL_NODE_LIMIT:
            topo_result = {
                'homology': topo.compute_homology(),
                'cycles': topo.compute_cycles(),
                'spectrum': {},
                'centrality': {},
            }
            print(f"  [大图] 跳过 spectrum/centrality (n={n_nodes} 超限 {_TOPO_FULL_NODE_LIMIT})")
        else:
            topo_result = topo.full_analysis()
    except Exception as e:
        print(f"  ⚠ 拓扑分析异常: {e}, 使用空结果继续")
        topo_result = {'error': str(e), 'homology': {}}
    topo_time = time.time() - t0

    hom = topo_result.get('homology', {})
    # 兼容旧写法: compute_homology() 返回的键是 euler_characteristic,
    # 历史版本这里误写成 euler_char, 导致 summary.json 里该字段恒为 null。
    euler_char = hom.get('euler_characteristic', hom.get('euler_char'))
    print(f"  拓扑: β0={hom.get('beta_0', '?')}, β1={hom.get('beta_1', '?')}, "
          f"χ={euler_char if euler_char is not None else '?'} ({topo_time:.1f}s)")

    # ---- 4. TDA 持久同调 (大图降采样) ----
    t0 = time.time()
    tda = TDAnalyzer(graph)
    # 节点数大时自动降采样, 避免耗时过久
    if n_nodes >= 50000:
        max_samples = 50
    elif n_nodes >= 10000:
        max_samples = 100
    else:
        max_samples = 200
    try:
        tda_result = tda.compute_persistence(max_samples=max_samples, max_dim=1)
    except Exception as e:
        print(f"  ⚠ TDA 异常: {e}, 使用空结果继续")
        tda_result = {'error': str(e), 'bars': {}, 'stats': {}, 'n_samples': 0}
    tda_time = time.time() - t0
    bars = tda_result.get('bars', {})
    print(f"  TDA: H0={len(bars.get(0, []))}, H1={len(bars.get(1, []))}, "
          f"max_samples={max_samples} ({tda_time:.1f}s)")

    # ---- 5. 社区检测 ----
    # 注: 这里不再用 {'n_communities': 1, 'modularity': 0.0} 兜底 —— 那个常量恰好
    # 等于"图连通且无社区结构"的真实结果, 会把失败伪装成正常结果 (summary.json 里
    # 的 1/0 就是这么来的)。现在失败会在结果里留下 degraded/diagnostic 标记。
    t0 = time.time()
    comm = CommunityAnalyzer(graph)
    try:
        comm_result = comm.detect_communities()
    except Exception as e:
        print(f"  ⚠ 社区检测异常: {type(e).__name__}: {e}")
        comm_result = {'n_communities': 0, 'modularity': None,
                       'degraded': True,
                       'diagnostic': f'{type(e).__name__}: {e}'}
    if comm_result.get('degraded'):
        print(f"  ⚠ 社区检测降级: {comm_result.get('diagnostic')}")
    comm_time = time.time() - t0

    # ---- 6. 特征提取 + ML 聚类 ----
    t0 = time.time()
    ml = TopologyFeatureML()
    try:
        features = ml.extract_graph_features(graph, topo_result, tda_result)
        n_clusters = min(5, max(1, n_nodes))
        clusters = ml.cluster_nodes(graph, n_clusters=n_clusters)
    except Exception as e:
        print(f"  ⚠ ML聚类异常: {e}")
        features = np.zeros((n_nodes, 8), dtype=np.float64)
        clusters = {n: 0 for n in graph.nodes()}
    ml_time = time.time() - t0
    print(f"  ML: 特征{features.shape}, {len(set(clusters.values()))}簇 ({ml_time:.1f}s)")

    # ---- 7. GNN 训练 (自我迭代: 先加载已有模型再继续训练) ----
    t0 = time.time()
    gnn = SimpleGNN(input_dim=8, hidden_dim=64, output_dim=32)
    gnn_path = os.path.join(model_dir, 'gnn_global.pt')
    loaded_gnn = False
    if resume and os.path.exists(gnn_path):
        loaded_gnn = gnn.load_model(gnn_path)
        if loaded_gnn:
            print(f"  GNN: 加载已有模型, 在其基础上继续训练 (自我迭代)")
    # 无论是否加载, 都在当前图上继续训练 (即使加载了也做 fine-tune)
    # 大图适当减少 epoch, 避免耗时过长
    epochs_here = max(5, gnn_epochs if n_nodes < 10000 else gnn_epochs // 2)
    try:
        gnn.train(graph, epochs=epochs_here, lr=0.005 if loaded_gnn else 0.01,
                  progress_callback=lambda **kw: None)
    except Exception as e:
        print(f"  ⚠ GNN训练异常: {e}")
    # 保存 (覆盖全局模型: 每次布局都更新, 实现持续迭代)
    try:
        gnn.save_model(gnn_path)
    except Exception as e:
        print(f"  ⚠ GNN保存失败: {e}")
    try:
        embeddings = gnn.get_embeddings(graph)
    except Exception:
        embeddings = np.zeros((n_nodes, 32), dtype=np.float64)
    gnn_time = time.time() - t0
    load_tag = " (迭代自旧模型)" if loaded_gnn else ""
    print(f"  GNN: 嵌入{embeddings.shape}{load_tag} ({gnn_time:.1f}s, {epochs_here}epochs)")

    # ---- 8. TDA 深度学习 (自我迭代) ----
    t0 = time.time()
    tda_dl = TDADeepLearning()
    try:
        feat_vec = tda_dl.build_feature_vector(tda_result)
        tda_ae_path = os.path.join(model_dir, 'tda_ae_global.pt')
        loaded_ae = False
        if resume and os.path.exists(tda_ae_path) and feat_vec.shape[0] > 0:
            loaded_ae = tda_dl.load_model(tda_ae_path)
            # 维度必须匹配
            ae = getattr(tda_dl, '_autoencoder', None)
            if loaded_ae and ae is not None:
                expected_dim = ae.encoder[0].in_features
                if expected_dim != feat_vec.shape[0]:
                    loaded_ae = False
        # 训练/微调 (加载过的话学习率降低, self._autoencoder 已存在会自动复用)
        ae_epochs = 50 if n_nodes < 10000 else 20
        lr_here = 0.0005 if loaded_ae else 0.001
        try:
            tda_dl.train_autoencoder(feat_vec.reshape(1, -1), epochs=ae_epochs,
                                     progress_callback=lambda **kw: None,
                                     lr=lr_here)
        except Exception as e:
            print(f"  ⚠ TDA-DL训练异常: {e}")
        try:
            tda_dl.save_model(tda_ae_path)
        except Exception:
            pass
    except Exception as e:
        print(f"  ⚠ TDA-DL异常: {e}")
        feat_vec = np.array([])
    dl_time = time.time() - t0
    print(f"  TDA-DL: 特征{feat_vec.shape} ({dl_time:.1f}s)")

    # ---- 9. 布局哈希 (供结果记录与文件命名使用) ----
    # 原第 9 步是 RL 求解 (Q-learning), 已整体移除:
    #   _is_goal 只判断"存在任意 2x2 方块位于底部中央", 起点就可能已满足;
    #   6.5~7.8 万节点上表格型 Q-learning 无法收敛, solve() 退化为随机游走,
    #   因此 rl_solved / solution_length 是伪信号, 不再产出这两个指标。
    layout_hash_str = compute_layout_hash(state)

    # ---- 10. 保存结果 (逐布局目录) ----
    save_dir = os.path.join(output_dir, f'puzzle_{idx:04d}')
    os.makedirs(save_dir, exist_ok=True)

    # 布局 JSON (原子写入)
    layout_path = os.path.join(save_dir, 'layout.json')
    tmp_p = layout_path + '.tmp'
    with open(tmp_p, 'w', encoding='utf-8') as f:
        json.dump(layout.to_dict(), f, indent=2, ensure_ascii=False)
    os.replace(tmp_p, layout_path)

    # 状态空间图 (pickle, 原子写入)
    try:
        gpath = os.path.join(save_dir, 'graph.pkl')
        tmp_g = gpath + '.tmp'
        with open(tmp_g, 'wb') as f:
            pickle.dump(graph, f)
        os.replace(tmp_g, gpath)
    except Exception as e:
        print(f"  ⚠ graph.pkl 保存失败: {e}")

    # 嵌入向量 + 拓扑特征
    try:
        np.save(os.path.join(save_dir, 'embeddings.npy'), embeddings)
        np.save(os.path.join(save_dir, 'features.npy'), features)
    except Exception:
        pass

    # 逐布局保存 (per-puzzle)
    try:
        gnn.save_model(os.path.join(save_dir, 'gnn_model.pt'))
        tda_dl.save_model(os.path.join(save_dir, 'tda_ae_model.pt'))
        ml.save_model(os.path.join(save_dir, 'ml_model.pkl'))
    except Exception:
        pass

    # 结果摘要
    result = {
        'idx': idx,
        'rows': layout.rows,
        'cols': layout.cols,
        'n_blocks': len(layout.blocks),
        'n_nodes': n_nodes,
        'n_edges': n_edges,
        'was_partial': was_partial,  # 是否部分图 (节点数超限截断)
        'gen_time': round(gen_time, 2),
        'topo_time': round(topo_time, 2),
        'tda_time': round(tda_time, 2),
        'ml_time': round(ml_time, 2),
        'gnn_time': round(gnn_time, 2),
        'dl_time': round(dl_time, 2),
        'total_time': round(gen_time + topo_time + tda_time + ml_time + gnn_time + dl_time, 2),
        'beta_0': hom.get('beta_0'),
        'beta_1': hom.get('beta_1'),
        'euler_char': euler_char,
        'n_communities': comm_result.get('n_communities', 0),
        'modularity': round((comm_result.get('modularity') or 0), 4),
        'community_method': comm_result.get('method'),
        'community_degraded': bool(comm_result.get('degraded')),
        'community_diagnostic': comm_result.get('diagnostic'),
        'community_note': comm_result.get('note'),
        'features_shape': list(features.shape),
        'embeddings_shape': list(embeddings.shape),
        'layout_hash': layout_hash_str,
        'resumed_gnn': loaded_gnn,
    }
    rp = os.path.join(save_dir, 'result.json')
    tmp_r = rp + '.tmp'
    with open(tmp_r, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    os.replace(tmp_r, rp)

    return result


# ==================== 批量训练主流程 ====================

def main():
    parser = argparse.ArgumentParser(description='华容道 AI 自动训练程序 (增强版)')
    parser.add_argument('--n', type=int, default=50, help='生成布局数量 (默认 50)')
    parser.add_argument('--min-dim', type=int, default=2, help='棋盘最小维度 (默认 2)')
    parser.add_argument('--max-dim', type=int, default=6, help='棋盘最大维度 (默认 6)')
    parser.add_argument('--gnn-epochs', type=int, default=200, help='GNN 每布局训练轮数 (默认 200)')
    parser.add_argument('--output', type=str, default='训练数据', help='输出目录 (默认 训练数据)')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认 42)')
    parser.add_argument('--max-nodes', type=int, default=200000,
                        help='BFS状态空间最大节点数, 超过则停止 (默认 200000; AI仍在部分图上继续)')
    parser.add_argument('--no-resume', action='store_true',
                        help='禁用自我迭代, 每次都从零开始训练 (默认False: 启用)')
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    model_dir = os.path.join(output_dir, '模型')
    os.makedirs(model_dir, exist_ok=True)

    # 设置单布局函数的 max_nodes 属性
    run_single_puzzle._max_nodes = args.max_nodes

    resume = not args.no_resume

    print("=" * 60)
    print("  华容道 AI 自动训练程序 (增强版)")
    print("=" * 60)
    print(f"  棋盘: {args.min_dim}x{args.min_dim} ~ {args.max_dim}x{args.max_dim} (每布局随机)")
    print(f"  块配置: 按棋盘尺寸智能随机 (不固定)")
    print(f"  数量: {args.n} 个布局")
    print(f"  GNN轮数: {args.gnn_epochs}")
    print(f"  BFS节点上限: {args.max_nodes} (超限AI仍在部分图上运行)")
    print(f"  自我迭代: {'启用 (加载已有模型继续训练 + 覆盖保存)' if resume else '禁用 (每次从零开始)'}")
    print(f"  输出: {output_dir}/")
    print(f"  随机种子: {args.seed}")
    print("=" * 60)

    # 启动前检查已有模型 (提示用户)
    if resume:
        existing = []
        for fn in ['gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl']:
            p = os.path.join(model_dir, fn)
            if os.path.exists(p):
                existing.append(fn)
        print(f"\n[自我迭代] 发现已有全局模型: {existing if existing else '无'}")
        print(f"[自我迭代] 每个布局训练后都会覆盖更新全局模型, 累积优化...\n")

    generator = DynamicLayoutGenerator(max_nodes=args.max_nodes,
                                       min_dim=args.min_dim,
                                       max_dim=args.max_dim)
    all_results: List[dict] = []
    failed = 0
    total_start = time.time()

    for i in range(args.n):
        seed = args.seed + i * 1000
        layout = generator.generate(seed=seed)
        if layout is None:
            print(f"\n[布局 {i:04d}] 生成失败, 跳过")
            failed += 1
            continue

        # 预检查
        try:
            blocks = [Block(bid, coords) for bid, coords in layout.blocks.items()]
            state = State(blocks, (layout.rows, layout.cols))
            if not state.is_valid():
                failed += 1
                continue
        except Exception:
            failed += 1
            continue

        try:
            result = run_single_puzzle(
                layout, i, output_dir,
                gnn_epochs=args.gnn_epochs,
                resume=resume,
            )
            if result:
                all_results.append(result)
            else:
                failed += 1
        except Exception as e:
            print(f"\n[布局 {i:04d}] 异常: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    total_time = time.time() - total_start

    # ==================== 汇总报告 ====================
    print(f"\n{'=' * 60}")
    print("  训练完成汇总")
    print(f"{'=' * 60}")
    print(f"  成功: {len(all_results)}/{args.n} (失败 {failed})")
    print(f"  总耗时: {total_time:.1f}s")
    if all_results:
        avg_time = total_time / len(all_results)
        print(f"  平均: {avg_time:.1f}s/布局")

        total_nodes = sum(r['n_nodes'] for r in all_results)
        total_edges = sum(r['n_edges'] for r in all_results)
        partials = sum(1 for r in all_results if r.get('was_partial'))
        resumed_gnn = sum(1 for r in all_results if r.get('resumed_gnn'))
        print(f"  总状态节点: {total_nodes:,}")
        print(f"  总边数: {total_edges:,}")
        print(f"  部分图(节点超限截断): {partials}/{len(all_results)} "
              f"({100*partials/len(all_results):.0f}%)")
        print(f"  GNN自我迭代次数: {resumed_gnn}/{len(all_results)}")

        # 棋盘尺寸分布
        from collections import Counter
        size_counter = Counter((r['rows'], r['cols']) for r in all_results)
        print(f"  棋盘尺寸分布: {dict(size_counter)}")

        # 拓扑统计
        beta0s = [r['beta_0'] for r in all_results if r['beta_0'] is not None]
        beta1s = [r['beta_1'] for r in all_results if r['beta_1'] is not None]
        if beta0s:
            print(f"  β0范围: {min(beta0s)}-{max(beta0s)} (均值 {np.mean(beta0s):.1f})")
        if beta1s:
            print(f"  β1范围: {min(beta1s)}-{max(beta1s)} (均值 {np.mean(beta1s):.1f})")

        mods = [r['modularity'] for r in all_results if r['modularity'] is not None]
        if mods:
            print(f"  模块度范围: {min(mods):.3f}-{max(mods):.3f}")

    # 保存汇总
    summary = {
        'n_puzzles': args.n,
        'n_succeeded': len(all_results),
        'n_failed': failed,
        'total_time': round(total_time, 2),
        'min_dim': args.min_dim,
        'max_dim': args.max_dim,
        'gnn_epochs': args.gnn_epochs,
        'resume_enabled': resume,
        'results': all_results,
    }
    summary_path = os.path.join(output_dir, 'summary.json')
    tmp_s = summary_path + '.tmp'
    with open(tmp_s, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    os.replace(tmp_s, summary_path)
    print(f"\n  汇总已保存: {summary_path}")
    print(f"  训练数据目录: {os.path.abspath(output_dir)}/")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
