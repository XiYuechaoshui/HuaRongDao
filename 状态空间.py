"""状态空间生成器: BFS 遍历全部可达状态, 构建 networkx.Graph

AI 记忆优化:
  - 支持布局级图缓存 (graph_{hash}.pkl), 跳过 BFS 生成
  - 支持 AI 引导式 BFS (优先扩展高价值节点, 大图可部分生成)
"""
import os
import time
import pickle
from collections import deque
from typing import Dict, Optional
import networkx as nx

from 数据模型 import State
from 布局哈希 import compute_layout_hash, layout_model_paths, ensure_model_dir

# 直径计算阈值: 超过该节点数时 diameter 默认跳过 (因为太慢),
# 仅在界面上用户点击"计算直径"时才执行
DIAMETER_THRESHOLD = 500


class StateSpaceGenerator:

    def __init__(self):
        self.graph = None
        self.initial_state = None
        self.is_running = False
        self.should_stop = False
        self.total_states = 0
        self.total_transitions = 0
        self.generation_time = 0
        self._analysis_cache = None
        self._layout_hash = None  # 当前布局哈希 (供缓存使用)

    # ============ 图缓存 (跳过 BFS) ============

    def try_load_cached_graph(self, initial_state: State) -> Optional[nx.Graph]:
        """尝试从布局级缓存加载图, 成功返回 Graph, 失败返回 None

        这是 AI 记忆系统的一部分: 同一布局第二次运行时跳过 BFS,
        直接加载已序列化的图, 可将状态空间生成时间从分钟级降到秒级。
        """
        layout_hash = compute_layout_hash(initial_state)
        self._layout_hash = layout_hash
        paths = layout_model_paths(layout_hash)
        graph_path = paths['graph']

        if not os.path.exists(graph_path):
            return None

        try:
            with open(graph_path, 'rb') as f:
                data = pickle.load(f)
            # 缓存是**某种对称性模式**下的状态空间: 对称模式决定 State 怎么算相等
            # (按 block ID 区分 vs 物理等价合并), 两种模式节点数完全不同。
            # 而布局哈希只含 (blocks, grid), 不含模式 —— 不做这道校验就会有:
            # 关掉对称性后"秒级命中"一份对称模式生成的图, 节点数、后续所有分析结果
            # 全是另一套, 界面还报"✓ AI记忆命中"。旧缓存没有该字段 → 按默认 True 处理。
            from 数据模型 import State as _State
            if bool(data.get('symmetric', True)) != bool(_State._symmetric):
                return None
            G = nx.Graph()
            G.add_nodes_from(data['nodes'])
            G.add_edges_from(data['edges'])
            self.graph = G
            self.total_states = G.number_of_nodes()
            self.total_transitions = G.number_of_edges()
            self.generation_time = data.get('generation_time', 0.0)
            self._analysis_cache = None
            return G
        except Exception:
            return None

    def save_cached_graph(self):
        """将当前图序列化保存为布局级缓存 (供下次加载)"""
        if self.graph is None or self._layout_hash is None:
            return
        try:
            ensure_model_dir()
            paths = layout_model_paths(self._layout_hash)
            # 原子写入: 先写 .tmp 再 os.replace
            tmp_path = paths['graph'] + '.tmp'
            with open(tmp_path, 'wb') as f:
                pickle.dump({
                    'nodes': list(self.graph.nodes()),
                    'edges': list(self.graph.edges()),
                    'generation_time': self.generation_time,
                    # 与 try_load_cached_graph 的校验配对: 模式不同的缓存不可复用
                    'symmetric': bool(State._symmetric),
                }, f)
            os.replace(tmp_path, paths['graph'])
        except Exception:
            pass  # 缓存保存失败不影响主流程

    def generate(self, initial_state: State,
                progress_callback: callable = None) -> nx.Graph:
        self.initial_state = initial_state
        self.is_running = True
        self.should_stop = False
        self.total_states = 0
        self.total_transitions = 0
        self._analysis_cache = None
        self._layout_hash = compute_layout_hash(initial_state)

        start_time = time.time()
        graph = nx.Graph()
        queue = deque([initial_state])
        visited = {initial_state}
        graph.add_node(initial_state)

        # 局部变量加速
        pop = queue.popleft
        add_visited = visited.add
        add_node = graph.add_node
        append_queue = queue.append
        add_edge = graph.add_edge
        get_moves = State.get_possible_moves

        try:
            while queue and not self.should_stop:
                current = pop()
                self.total_states += 1

                for next_state in get_moves(current):
                    if next_state not in visited:
                        add_visited(next_state)
                        add_node(next_state)
                        append_queue(next_state)
                    add_edge(current, next_state)
                    self.total_transitions += 1

                if progress_callback and self.total_states & 0x3F == 0:  # mod 64
                    progress_callback(
                        states=self.total_states,
                        transitions=self.total_transitions,
                        queue_size=len(queue)
                    )
                # 移除频繁 gc.collect(): Python 自带 GC 已经足够, 强制 GC 反而造成卡顿抖动

        except MemoryError:
            self.should_stop = True
            raise

        finally:
            self.is_running = False
            self.generation_time = time.time() - start_time
            self.graph = graph

            if progress_callback:
                progress_callback(
                    states=self.total_states,
                    transitions=self.total_transitions,
                    queue_size=0,
                    finished=True
                )

            # 生成完成后自动保存图缓存 (AI 记忆)
            if not self.should_stop:
                self.save_cached_graph()

        return graph

    def stop(self):
        self.should_stop = True

    def compute_diameter(self) -> int:
        """独立的直径计算 (用户手动触发, 不在 analyze() 中默认执行)"""
        if self.graph is None or not nx.is_connected(self.graph) or len(self.graph.nodes) <= 1:
            return None
        try:
            return nx.diameter(self.graph)
        except Exception:
            return None

    def analyze(self, include_diameter: bool = None) -> Dict:
        """快速分析: 默认跳过昂贵的 nx.diameter (超过阈值时)

        参数:
          include_diameter: True=强制计算, False=强制跳过, None=按节点数阈值自动决定
        """
        if self.graph is None or len(self.graph.nodes) == 0:
            return {}
        if self._analysis_cache is not None and include_diameter is None:
            return self._analysis_cache

        G = self.graph
        n = len(G.nodes)
        m = len(G.edges)

        analysis = {
            'num_states': n,
            'num_transitions': m,
            'generation_time': self.generation_time,
            'stopped_early': self.should_stop,
        }

        # BFS 生成的图一定是连通的 (除非被中途停止)
        analysis['connected'] = (not self.should_stop) or nx.is_connected(G)

        # 直径计算: 非常慢 (O(N*(N+M))), 默认阈值下跳过
        if include_diameter is None:
            include_diameter = (n <= DIAMETER_THRESHOLD)
        if include_diameter and analysis['connected'] and n > 1:
            try:
                analysis['diameter'] = nx.diameter(G)
            except Exception:
                analysis['diameter'] = 'N/A (计算超时或异常)'
        else:
            analysis['diameter'] = f'未计算 (节点数={n:,} > {DIAMETER_THRESHOLD}, 可点击界面单独计算)'

        analysis['density'] = nx.density(G)

        # 度统计: 一次性迭代, 避免多次构造 list
        if n > 0:
            deg_iter = (d for _, d in G.degree())
            s = 0
            mx = 0
            mn = float('inf')
            knt = 0
            for d in deg_iter:
                s += d
                knt += 1
                if d > mx:
                    mx = d
                if d < mn:
                    mn = d
            if knt > 0:
                analysis['avg_degree'] = s / knt
                analysis['max_degree'] = mx
                analysis['min_degree'] = mn

        # 连通分量: 不连通时才计算
        if not analysis['connected']:
            components = list(nx.connected_components(G))
            analysis['num_components'] = len(components)
            analysis['component_sizes'] = sorted([len(c) for c in components], reverse=True)

        if include_diameter is None:
            self._analysis_cache = analysis
        return analysis
