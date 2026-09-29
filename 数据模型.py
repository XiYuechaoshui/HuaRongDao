"""核心数据结构: Block / State / LayoutData

Block 和 State 均使用元组 + frozenset 保证不可变和可哈希，
支持作为 networkx.Graph 的节点和 dict / set 的键。
"""
import json
from typing import List, Tuple, Dict

BLOCK_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
    '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#85929E'
]
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Block:
    """华容道单个方块 (不可变, 可哈希)"""

    __slots__ = ('id', 'coords', 'color', '_hash_cache')

    def __init__(self, id: int, coords: List[Tuple[int, int]], color: str = None):
        self.id = id
        self.coords = tuple(sorted((r, c) for r, c in coords))
        self.color = color or BLOCK_COLORS[id % len(BLOCK_COLORS)]
        self._hash_cache = None

    def move(self, dr: int, dc: int) -> 'Block':
        new_coords = tuple((r + dr, c + dc) for r, c in self.coords)
        return Block(self.id, new_coords, self.color)

    def __eq__(self, other):
        if not isinstance(other, Block):
            return False
        return self.id == other.id and self.coords == other.coords

    def __hash__(self):
        if self._hash_cache is None:
            self._hash_cache = hash((self.id, self.coords))
        return self._hash_cache

    def __repr__(self):
        return f"Block(id={self.id}, coords={list(self.coords)})"


class State:
    """华容道棋盘状态 (不可变, 可哈希)

    对称性模式 (_symmetric 类变量):
      True  (默认): 按 block ID 区分状态 (卒6在A + 卒7在B ≠ 卒6在B + 卒7在A)
      False:         物理等价合并 (同形状方块互换视为同一状态)
    """

    _symmetric = True  # 类变量: 控制全局对称性模式

    __slots__ = ('blocks', 'grid_shape', 'occupied', '_hash_cache', '_canon')

    def __init__(self, blocks: List[Block], grid_shape: Tuple[int, int],
                 _occupied: frozenset = None):
        self.blocks = tuple(blocks)
        self.grid_shape = grid_shape
        # 允许外部传入已构建的 occupied, 避免重复构建
        if _occupied is not None:
            self.occupied = _occupied
        else:
            # 一次性 union (比循环逐个 union 快得多)
            all_coords = []
            for b in self.blocks:
                all_coords.extend(b.coords)
            self.occupied = frozenset(all_coords)
        self._hash_cache = None
        self._canon = None

    def is_valid(self) -> bool:
        """完整合法性检查 (加载时用)"""
        total_cells = sum(len(b.coords) for b in self.blocks)
        if len(self.occupied) != total_cells:
            return False
        rows, cols = self.grid_shape
        for r, c in self.occupied:
            if r < 0 or r >= rows or c < 0 or c >= cols:
                return False
        return True

    def _can_move(self, block_idx: int, dr: int, dc: int) -> bool:
        """增量式 move 合法性检查 (不创建新 State 对象, 比先建再 is_valid 快 3~5x)"""
        rows, cols = self.grid_shape
        block = self.blocks[block_idx]
        # 其他 blocks 的占用集合 (去掉当前 block)
        # 这里直接遍历排除, 避免构建新 set
        orig_coords = block.coords
        for (r, c) in orig_coords:
            nr, nc = r + dr, c + dc
            # 边界检查
            if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                return False
            # 不与其他 blocks 重叠: 新位置必须要么在原 block 自身, 要么不在 occupied
            if (nr, nc) not in orig_coords and (nr, nc) in self.occupied:
                return False
        return True

    def get_possible_moves(self) -> List['State']:
        moves = []
        blocks = self.blocks
        rows, cols = self.grid_shape
        occupied = self.occupied
        for idx, block in enumerate(blocks):
            orig_coords = block.coords
            for dr, dc in DIRECTIONS:
                # ---- 增量合法性检查 (不创建对象) ----
                ok = True
                for (r, c) in orig_coords:
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
                        ok = False
                        break
                    if (nr, nc) not in orig_coords and (nr, nc) in occupied:
                        ok = False
                        break
                if not ok:
                    continue
                # ---- 通过检查, 构建新 State (传入 occupied 增量更新) ----
                new_block = block.move(dr, dc)
                # 构建新 blocks tuple
                new_blocks = blocks[:idx] + (new_block,) + blocks[idx + 1:]
                # 增量更新 occupied: 移除原 coords, 加入新 coords
                new_occ = occupied.difference(orig_coords).union(new_block.coords)
                new_state = State.__new__(State)
                new_state.blocks = new_blocks
                new_state.grid_shape = (rows, cols)
                new_state.occupied = new_occ
                new_state._hash_cache = None
                new_state._canon = None
                moves.append(new_state)
        return moves

    def _get_canon(self):
        """计算物理等价的规范化 key (忽略 block ID)

        将每个 block 的 (形状模式, 绝对坐标) 排序后组成 tuple,
        同形状方块互换会得到相同的 key。
        """
        if self._canon is not None:
            return self._canon
        keys = []
        for b in self.blocks:
            coords = b.coords
            min_r = min(r for r, c in coords)
            min_c = min(c for r, c in coords)
            # 形状模式: coords 相对于左上角的偏移 (区分 1x1 / 1x2横 / 2x1竖 / 2x2)
            shape = tuple(sorted((r - min_r, c - min_c) for r, c in coords))
            keys.append((shape, coords))
        keys.sort()
        self._canon = (tuple(keys), self.grid_shape)
        return self._canon

    def __eq__(self, other):
        if not isinstance(other, State):
            return False
        if State._symmetric:
            return self.blocks == other.blocks and self.grid_shape == other.grid_shape
        # 非对称模式: 比较规范化 key (忽略 block ID)
        return self._get_canon() == other._get_canon()

    def __hash__(self):
        sym = State._symmetric
        # 缓存格式: (模式, hash值), 模式不匹配时重算
        if self._hash_cache is not None:
            cached_sym, cached_h = self._hash_cache
            if cached_sym == sym:
                return cached_h
        if sym:
            h = hash((self.blocks, self.grid_shape))
        else:
            h = hash(self._get_canon())
        self._hash_cache = (sym, h)
        return h

    def __repr__(self):
        return f"State(blocks={len(self.blocks)}, grid={self.grid_shape})"


class LayoutData:
    """布局数据 (用于序列化/反序列化)"""

    def __init__(self, rows: int, cols: int, blocks: Dict[int, List[Tuple[int, int]]]):
        self.rows = rows
        self.cols = cols
        self.blocks = blocks

    def to_dict(self) -> dict:
        return {
            'rows': self.rows,
            'cols': self.cols,
            'blocks': {str(k): v for k, v in self.blocks.items()}
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> 'LayoutData':
        blocks = {}
        for k, v in data.get('blocks', {}).items():
            blocks[int(k)] = [(r, c) for r, c in v]
        return cls(data['rows'], data['cols'], blocks)
