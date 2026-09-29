"""3D/2D 可视化器: 状态空间图 + 棋盘预览"""
from collections import deque
from typing import Optional
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from 数据模型 import State
from 字体工具 import CN_FONT
from 主题 import Theme


class HoverVisualizer:

    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        self.node_positions = {}
        self.node_to_state = {}
        self.graph = None

    def _auto_focus(self, ax, positions):
        if not positions:
            return
        all_pos = np.array(list(positions.values()))
        mins = all_pos.min(axis=0)
        maxs = all_pos.max(axis=0)
        padding = (maxs - mins) * 0.1 + 0.5
        ax.set_xlim(mins[0] - padding[0], maxs[0] + padding[0])
        ax.set_ylim(mins[1] - padding[1], maxs[1] + padding[1])
        ax.set_zlim(mins[2] - padding[2], maxs[2] + padding[2])

    def _decentralized_layout(self, graph, subgraph_nodes, depths, center_state, max_steps):
        """去中心化的均匀 3D 布局

        核心思想: 每个节点的子节点在该节点的"前方"等角度散开,
        不存在全局中心点。从中心节点开始 BFS, 每扩展一层,
        子节点在父节点位置的局部球面上均匀分布 (斐波那契球面),
        保证:
          · 同一节点的所有子节点到该节点等距
          · 子节点之间在球面上近似均匀分布
          · 父→子方向沿"背离根节点"方向 (使整体向外发散, 更立体)

        参数:
          graph: 完整图
          subgraph_nodes: 要布局的节点集合
          depths: 各节点到 center_state 的 BFS 深度
          center_state: 起点 (仅用于确定发散方向, 不作为视觉中心)
          max_steps: 最大深度
        返回:
          dict {node: (x, y, z)}
        """
        if not subgraph_nodes:
            return {}

        pos = {}
        # 每层间距 (z 方向不重要, 用半径控制)
        layer_radius = 1.5

        # 起点
        pos[center_state] = np.array([0.0, 0.0, 0.0])

        # BFS 扩展: 每个节点的子节点在该节点的局部球面上均匀分布
        # 球面方向: 以"从根指向当前节点的方向"为极轴, 子节点在该方向的半球面上散开
        visited = {center_state}
        # 记录每个节点指向"远离根"的方向 (单位向量)
        outward_dir = {center_state: np.array([0.0, 0.0, 1.0])}

        # 按深度处理, 保证父节点先于子节点被布局
        nodes_by_depth = {}
        for node in subgraph_nodes:
            d = depths.get(node, 0)
            nodes_by_depth.setdefault(d, []).append(node)

        for d in range(0, max(depths.values()) + 1 if depths else 0):
            for node in nodes_by_depth.get(d, []):
                if node not in pos:
                    continue
                parent_pos = pos[node]
                parent_dir = outward_dir.get(node, np.array([0.0, 0.0, 1.0]))

                # 找该节点的所有未访问邻居 (子节点)
                children = []
                for nb in graph.neighbors(node):
                    if nb in subgraph_nodes and nb not in visited:
                        children.append(nb)
                if not children:
                    continue

                # 在以 parent_dir 为极轴的半球面上用斐波那契分布均匀放置子节点
                n = len(children)
                child_positions = self._fibonacci_hemisphere(n, parent_dir, layer_radius)

                for child, offset in zip(children, child_positions):
                    if child in pos:
                        continue
                    child_pos = parent_pos + offset
                    pos[child] = child_pos
                    visited.add(child)
                    # 子节点的 outward 方向 = 偏移方向 (继续向外发散)
                    norm = np.linalg.norm(offset)
                    outward_dir[child] = offset / norm if norm > 1e-9 else parent_dir

        # 处理仍未布局的孤立节点 (理论上不应出现, 兜底)
        for node in subgraph_nodes:
            if node not in pos:
                pos[node] = np.array([0.0, 0.0, 0.0])

        return {node: tuple(pos[node]) for node in subgraph_nodes}

    def _fibonacci_hemisphere(self, n, axis, radius):
        """在以 axis 为极轴的半球面上用斐波那契螺旋均匀分布 n 个点

        返回 n 个 3D 偏移向量 (相对原点), 每个向量长度 ≈ radius。
        斐波那契螺旋保证球面上点近似等距。
        """
        axis = np.asarray(axis, dtype=float)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / norm

        # 构造正交基: axis 为 z', 找两个正交的 x', y'
        if abs(axis[2]) < 0.9:
            ref = np.array([0.0, 0.0, 1.0])
        else:
            ref = np.array([1.0, 0.0, 0.0])
        x_axis = np.cross(ref, axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(axis, x_axis)

        offsets = []
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        # 半球: 极角 theta 从 0 到 pi/2 (axis 方向的半球)
        for i in range(n):
            # 斐波那契分布
            t = (i + 0.5) / n
            theta = np.arccos(1.0 - t)  # 0 ~ pi/2, 沿 axis 方向的半球
            phi = i * golden_angle
            # 球面坐标转笛卡尔 (在局部基下)
            local = np.array([
                radius * np.sin(theta) * np.cos(phi),
                radius * np.sin(theta) * np.sin(phi),
                radius * np.cos(theta),
            ])
            # 转回世界坐标
            world = local[0] * x_axis + local[1] * y_axis + local[2] * axis
            offsets.append(world)
        return offsets


    def draw_state_grid(self, state: State, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        rows, cols = state.grid_shape

        # 逐格填充: 每个方块的每个格子单独绘制 1x1 矩形,
        # 避免非矩形方块 (L形/T形等) 的包围盒覆盖到其他方块的格子
        for block in state.blocks:
            for (r, c) in block.coords:
                rect = plt.Rectangle((c, r), 1, 1,
                                   facecolor=block.color, edgecolor='none',
                                   linewidth=0, alpha=0.8)
                ax.add_patch(rect)

        # 绘制方块外轮廓: 仅画出与异块/空格相邻的边, 同块内部不画线
        for block in state.blocks:
            coords_set = set(block.coords)
            for (r, c) in block.coords:
                if (r - 1, c) not in coords_set:
                    ax.plot([c, c + 1], [r, r], color='black', linewidth=1.5)
                if (r + 1, c) not in coords_set:
                    ax.plot([c, c + 1], [r + 1, r + 1], color='black', linewidth=1.5)
                if (r, c - 1) not in coords_set:
                    ax.plot([c, c], [r, r + 1], color='black', linewidth=1.5)
                if (r, c + 1) not in coords_set:
                    ax.plot([c + 1, c + 1], [r, r + 1], color='black', linewidth=1.5)

        # 方块编号: 标注在各格质心 (非矩形方块不会偏到包围盒外的格子上)
        for block in state.blocks:
            n = len(block.coords)
            avg_r = sum(r for r, c in block.coords) / n
            avg_c = sum(c for r, c in block.coords) / n
            ax.text(avg_c + 0.5, avg_r + 0.5, str(block.id),
                   ha='center', va='center', fontsize=10, fontweight='bold')

        for i in range(rows + 1):
            ax.axhline(y=i, color='black', linewidth=0.5)
        for j in range(cols + 1):
            ax.axvline(x=j, color='black', linewidth=0.5)

        ax.set_xlim(0, cols)
        ax.set_ylim(0, rows)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        ax.axis('off')
        return ax

    def draw_state_space(self, graph: nx.Graph, current_state: State, ax=None,
                        title: str = None, max_steps: int = 1, focus_state: Optional[State] = None):
        if ax is None:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')

        if len(graph.nodes) == 0:
            ax.text2D(0.5, 0.5, "空状态空间", transform=ax.transAxes, ha='center', va='center',
                     fontsize=14, fontproperties=CN_FONT)
            ax.set_axis_off()
            return ax

        center_state = focus_state or current_state
        if center_state is None:
            center_state = next(iter(graph.nodes()))

        visited = {center_state}
        queue = deque([(center_state, 0)])
        subgraph_nodes = {center_state}
        depths = {center_state: 0}

        while queue:
            state, steps = queue.popleft()
            if steps >= max_steps:
                continue
            for neighbor in graph.neighbors(state):
                if neighbor not in visited:
                    visited.add(neighbor)
                    subgraph_nodes.add(neighbor)
                    depths[neighbor] = steps + 1
                    queue.append((neighbor, steps + 1))

        subgraph = graph.subgraph(subgraph_nodes)

        # 去中心化布局: 每个节点的子节点在该节点的局部球面上均匀分布,
        # 保证同一父节点的所有子节点到该父节点等距, 整体向外发散, 更立体
        pos_3d = self._decentralized_layout(graph, subgraph_nodes, depths,
                                            center_state, max_steps)

        x = [pos_3d[n][0] for n in subgraph_nodes]
        y = [pos_3d[n][1] for n in subgraph_nodes]
        z = [pos_3d[n][2] for n in subgraph_nodes]

        # 节点按 BFS 深度着色。原来用 matplotlib 默认的 viridis (深紫→青→黄),
        # 在紫蓝主题里显得突兀; 改用 主题.Theme.深度色阶 (同色系单色阶, 明度仍严格单调,
        # 所以"靠亮度区分深度"的可判别性和色盲友好性没有丢)。
        # 注意: 这里是 3D 视图**实际渲染**走的路径; 主窗口._prepare_3d_draw_args 是后台
        # 预计算旁路, 两处必须用同一个色图, 否则按步骤数不同会看到两套配色。
        _深度图 = Theme.深度色图()
        depths_list = [depths[n] for n in subgraph_nodes]
        max_d = max(depths_list) if depths_list else 1
        colors_array = []
        for n in subgraph_nodes:
            d = depths[n]
            t = d / max_d if max_d > 0 else 0
            colors_array.append(_深度图(0.15 + t * 0.7))

        sizes = []
        for n in subgraph_nodes:
            if n == center_state:
                sizes.append(200)
            else:
                d = depths[n]
                sizes.append(max(30, 150 - d * 25))

        state_lookup = [n for n in subgraph_nodes]

        scatter = ax.scatter(x, y, z, c=colors_array, s=sizes, alpha=0.95, picker=12, linewidths=0.4, edgecolors='white')
        scatter._state_lookup = state_lookup
        scatter._base_sizes = np.array(sizes)
        scatter._base_colors = np.array(colors_array)
        scatter._node_index_map = {node: idx for idx, node in enumerate(subgraph_nodes)}
        self.node_positions = {node: pos_3d[node] for node in subgraph_nodes}
        self.node_to_state = {node: node for node in subgraph_nodes}
        self.graph = graph

        # 绘制边 (状态转移连线)
        for u, v in subgraph.edges():
            if u in pos_3d and v in pos_3d:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                # 深度越浅的边越突出
                d_avg = (depths.get(u, 0) + depths.get(v, 0)) / 2
                alpha = max(0.15, 0.6 - d_avg * 0.12)
                ax.plot([x0, x1], [y0, y1], [z0, z1],
                        color='#4a5568', alpha=alpha, linewidth=0.8, zorder=0)

        self._auto_focus(ax, pos_3d)
        ax.set_title(title or f"3D 状态空间视图（以 {len(subgraph_nodes)} 个状态为中心扩展）",
                    fontproperties=CN_FONT, fontsize=11, pad=8)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.8))
        ax.set_proj_type('persp')
        return ax
