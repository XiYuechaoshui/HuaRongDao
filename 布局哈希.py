"""布局哈希工具: 跨进程稳定的布局指纹 + 布局级模型记忆

统一的哈希方案:
  用 pickle.dumps 序列化 State.blocks + grid_shape → MD5 → 8位十六进制
  确保同一布局在任何进程中产生相同哈希值。

用途:
  1. 布局级模型记忆: gnn_{hash}.pt / tda_ae_{hash}.pt / ml_{hash}.pkl
  2. 布局级图缓存: graph_{hash}.pkl (跳过 BFS 生成)
  3. 布局级完整结果缓存: result_{hash}.json (跳过全部解析)
"""
import json as _json
import os
import sys
import hashlib
import pickle
from typing import Optional, List


def compute_layout_hash(state) -> str:
    """计算布局的跨进程稳定哈希 (8位十六进制)

    关键设计: 使用 pickle protocol=4 (Python 3.4+ 跨版本稳定)
    protocol=5 (Python 3.8+) 与 protocol=4 序列化结果不同,
    会导致同一布局在不同 Python 版本下产生不同哈希。

    参数:
        state: State 对象 (含 blocks + grid_shape) 或 (blocks, grid_shape) 元组
    返回:
        8位十六进制字符串, 如 '907a3aa9'
    """
    if hasattr(state, 'blocks') and hasattr(state, 'grid_shape'):
        blocks_data = tuple((b.id, tuple(tuple(c) for c in b.coords)) for b in state.blocks)
        grid = state.grid_shape
    else:
        blocks, grid = state
        blocks_data = tuple((b.id, tuple(tuple(c) for c in b.coords)) for b in blocks)

    payload = (blocks_data, grid)
    # protocol=4: Python 3.4+ 跨版本稳定, 不可用 HIGHEST_PROTOCOL (3.8+ 为 5)
    raw = pickle.dumps(payload, protocol=4)
    h = hashlib.md5(raw, usedforsecurity=False).hexdigest()
    return h[:8]


# ============ 布局级记忆文件路径工具 ============

# exe 模式下用 exe 所在目录, 否则用源码目录
if getattr(sys, 'frozen', False):
    _BASE_DIR = os.path.dirname(sys.executable)
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认模型来源目录 (含 '模型' 子目录的父目录)
_DEFAULT_MODEL_BASE = '训练数据'
_model_base = _DEFAULT_MODEL_BASE

# 供 layout_model_paths / global_model_paths / check_layout_memory 使用的内部变量
_MODEL_DIR = os.path.join(_BASE_DIR, _DEFAULT_MODEL_BASE, '模型')
_LAYOUT_CACHE_DIR = os.path.join(_BASE_DIR, _DEFAULT_MODEL_BASE, '布局缓存')


def set_model_base(base: str) -> None:
    """设置当前模型来源目录 (如 '训练数据' 或 '训练数据_实验A')

    Args:
        base: 目录名 (相对于 _BASE_DIR, 或绝对路径)
    """
    global _model_base, _MODEL_DIR, _LAYOUT_CACHE_DIR
    _model_base = base
    if os.path.isabs(base):
        _MODEL_DIR = os.path.join(base, '模型')
        _LAYOUT_CACHE_DIR = os.path.join(base, '布局缓存')
    else:
        _MODEL_DIR = os.path.join(_BASE_DIR, base, '模型')
        _LAYOUT_CACHE_DIR = os.path.join(_BASE_DIR, base, '布局缓存')


def get_model_base() -> str:
    """返回当前模型来源目录名"""
    return _model_base


def get_model_dir() -> str:
    """返回当前模型文件目录 (base/模型)"""
    return _MODEL_DIR


def scan_model_sources() -> List[str]:
    """扫描 _BASE_DIR 下所有含 '模型' 子目录的目录名。

    返回: 可用模型来源目录名列表 (默认包含 '训练数据')
    """
    sources = []
    try:
        for name in os.listdir(_BASE_DIR):
            full = os.path.join(_BASE_DIR, name)
            if os.path.isdir(full) and os.path.isdir(os.path.join(full, '模型')):
                sources.append(name)
    except OSError:
        pass
    # 确保默认目录在列表里
    if _DEFAULT_MODEL_BASE not in sources:
        sources.insert(0, _DEFAULT_MODEL_BASE)
    return sources


def ensure_model_dir() -> str:
    os.makedirs(_MODEL_DIR, exist_ok=True)
    return _MODEL_DIR


def ensure_layout_cache_dir() -> str:
    os.makedirs(_LAYOUT_CACHE_DIR, exist_ok=True)
    return _LAYOUT_CACHE_DIR


def layout_model_paths(layout_hash: str) -> dict:
    """返回布局级各模型文件的完整路径"""
    d = ensure_model_dir()
    return {
        'gnn': os.path.join(d, f'gnn_{layout_hash}.pt'),
        'tda_ae': os.path.join(d, f'tda_ae_{layout_hash}.pt'),
        'ml': os.path.join(d, f'ml_{layout_hash}.pkl'),
        'graph': os.path.join(d, f'graph_{layout_hash}.pkl'),
        'embeddings': os.path.join(d, f'embeddings_{layout_hash}.npy'),
        'result': os.path.join(d, f'result_{layout_hash}.json'),
    }


def global_model_paths() -> dict:
    """返回全局预训练模型路径 (用于跨布局知识迁移)"""
    d = ensure_model_dir()
    return {
        'gnn': os.path.join(d, 'gnn_global.pt'),
        'tda_ae': os.path.join(d, 'tda_ae_global.pt'),
        'ml': os.path.join(d, 'ml_global.pkl'),
    }


def check_layout_memory(layout_hash: str, state=None) -> dict:
    """检查某布局已有的记忆文件, 返回 {模型名: 存在路径}

    注: 原 RL Q 表的向后兼容迁移逻辑 (旧 hash() & 0xFFFFFFFF 命名) 已随 RL 功能
    一并移除 —— 那段代码依赖进程内不稳定的内置 hash(), 实际几乎不可能命中旧文件。
    `state` 参数保留仅为兼容既有调用点。
    """
    paths = layout_model_paths(layout_hash)
    existing = {}
    for name, path in paths.items():
        if os.path.exists(path):
            existing[name] = path
    return existing


# ============ 完整结果缓存 (跳过全部解析) ============



def save_full_result(layout_hash: str, result: dict):
    """保存完整解析结果 (拓扑/TDA/社区/AI增强) 到布局级缓存"""
    d = ensure_layout_cache_dir()
    path = os.path.join(d, f'result_{layout_hash}.json')

    # numpy 数组 + State/Block 需要特殊处理
    import numpy as np
    # 延迟导入避免循环依赖
    from 数据模型 import State, Block

    def _to_jsonable(obj):
        if isinstance(obj, dict):
            # 处理非字符串 key (如 State/Block 作为 key)
            items = []
            for k, v in obj.items():
                if isinstance(k, str):
                    items.append((k, _to_jsonable(v)))
                elif isinstance(k, State):
                    # State key: "__state__:rows,cols||id:r,c;r,c|id:r,c;r,c"
                    gs_str = f'{k.grid_shape[0]},{k.grid_shape[1]}'
                    blocks_str = '|'.join(
                        f'{b.id}:{";".join(f"{r},{c}" for r,c in b.coords)}'
                        for b in k.blocks
                    )
                    items.append((f'__state__:{gs_str}||{blocks_str}', _to_jsonable(v)))
                elif isinstance(k, Block):
                    # Block key: "__block__:id:r,c;r,c"
                    coords_str = ';'.join(f'{r},{c}' for r, c in k.coords)
                    items.append((f'__block__:{k.id}:{coords_str}', _to_jsonable(v)))
                else:
                    items.append((str(k), _to_jsonable(v)))
            return dict(items)
        if isinstance(obj, (list, tuple)):
            return [_to_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return {'__ndarray__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, frozenset):
            return {'__frozenset__': True, 'data': [_to_jsonable(v) for v in obj]}
        if isinstance(obj, set):
            return {'__set__': True, 'data': [_to_jsonable(v) for v in obj]}
        if isinstance(obj, State):
            return {'__state__': True,
                    'blocks': [[b.id, [list(c) for c in b.coords]] for b in obj.blocks],
                    'grid_shape': list(obj.grid_shape)}
        if isinstance(obj, Block):
            return {'__block__': True,
                    'id': obj.id,
                    'coords': [list(c) for c in obj.coords]}
        return obj

    # 原子写入: 先写 .tmp 再 os.replace, 防止进程被杀留下损坏文件
    # 附带对称性模式: 完整结果是在某种状态空间上算出来的, 而布局哈希不含该模式,
    # 不记录就会出现"关掉对称性后命中对称模式的结果"这种错误命中 (见 load_full_result)
    from 数据模型 import State as _State
    result = dict(result)
    result['_cache_meta'] = {'symmetric': bool(_State._symmetric)}
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        _json.dump(_to_jsonable(result), f, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_full_result(layout_hash: str) -> Optional[dict]:
    """加载完整解析结果, 失败返回 None"""
    import numpy as np
    from 数据模型 import State, Block
    path = os.path.join(ensure_layout_cache_dir(), f'result_{layout_hash}.json')
    if not os.path.exists(path):
        return None

    def _from_jsonable(obj):
        if isinstance(obj, dict):
            if obj.get('__ndarray__') is True:
                return np.array(obj['data'], dtype=obj.get('dtype', 'float64'))
            if obj.get('__frozenset__') is True:
                return frozenset(_from_jsonable(v) for v in obj['data'])
            if obj.get('__set__') is True:
                return set(_from_jsonable(v) for v in obj['data'])
            if obj.get('__state__') is True:
                blocks = [Block(b[0], [tuple(c) for c in b[1]]) for b in obj['blocks']]
                return State(blocks, tuple(obj['grid_shape']))
            if obj.get('__block__') is True:
                return Block(obj['id'], [tuple(c) for c in obj['coords']])
            # 处理序列化的 State/Block 作为 dict key 的情况
            result = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.startswith('__state__:'):
                    # 从序列化字符串重建 State: "__state__:rows,cols||id:r,c;r,c|id:r,c;r,c"
                    try:
                        rest = k[len('__state__:'):]
                        gs_part, blocks_part = rest.split('||', 1)
                        gs = tuple(int(x) for x in gs_part.split(','))
                        block_list = []
                        for b_str in blocks_part.split('|'):
                            bid, coords_str = b_str.split(':', 1)
                            coords = [tuple(int(x) for x in c.split(',')) for c in coords_str.split(';')]
                            block_list.append(Block(int(bid), coords))
                        result[State(block_list, gs)] = _from_jsonable(v)
                        continue
                    except Exception:
                        pass
                elif isinstance(k, str) and k.startswith('__block__:'):
                    # "__block__:id:r,c;r,c"
                    try:
                        rest = k[len('__block__:'):]
                        bid_str, coords_str = rest.split(':', 1)
                        coords = [tuple(int(x) for x in c.split(',')) for c in coords_str.split(';')]
                        result[Block(int(bid_str), coords)] = _from_jsonable(v)
                        continue
                    except Exception:
                        pass
                result[k] = _from_jsonable(v)
            return result
        if isinstance(obj, list):
            return [_from_jsonable(v) for v in obj]
        return obj

    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = _from_jsonable(_json.load(f))
        # 对称性模式不符 → 当作没有缓存 (宁可重算几分钟, 也不要秒级给出另一个
        # 状态空间的错误结果)。旧缓存没有 _cache_meta → 按默认 True 处理,
        # 所以默认(开启对称性)下的历史缓存照常命中。
        from 数据模型 import State as _State
        if bool((data.get('_cache_meta') or {}).get('symmetric', True)) != bool(_State._symmetric):
            return None
        return data
    except Exception:
        return None


# ============ 图序列化缓存 (跳过 BFS 生成) ============

def save_graph(layout_hash: str, graph) -> str:
    """序列化保存 networkx.Graph (节点为 State 对象)"""
    paths = layout_model_paths(layout_hash)
    path = paths['graph']
    import networkx as nx
    # 只保存边列表 + 节点哈希 → 重建图
    # State 对象不可变且可哈希, 直接 pickle
    # 原子写入: 先写 .tmp 再 os.replace
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        pickle.dump({'edges': list(graph.edges()), 'nodes': list(graph.nodes())}, f)
    os.replace(tmp_path, path)
    return path


def load_graph(layout_hash: str):
    """加载序列化的图, 返回 nx.Graph 或 None"""
    paths = layout_model_paths(layout_hash)
    path = paths['graph']
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
        G = nx.Graph()
        G.add_nodes_from(data['nodes'])
        G.add_edges_from(data['edges'])
        return G
    except Exception:
        return None
