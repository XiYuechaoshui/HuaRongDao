"""存档缓存管理器: 持久化保存解析结果

缓存结构:
  cache_dir/
    manifest.json          - 索引清单 (元数据, 不含大数据)
    {session_id}.json      - 单次会话完整数据 (布局/分析结果)

每次保存 = 一个 session, 包含:
  - session_id (时间戳)
  - name (用户可编辑)
  - layout (LayoutData 序列化)
  - basic_analysis (基本图统计)
  - topology_result (拓扑分析, 含谱特征值数组)
  - tda_result (持久同调)
  - community_result (社区检测)
  - created_at / updated_at
"""
import os
import json
import time
import shutil
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np


def _to_jsonable(obj):
    """递归将对象转为 JSON 可序列化 (主要处理 numpy 类型 + State/Block)"""
    # 延迟导入避免循环依赖
    from 数据模型 import State, Block
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, frozenset):
        return {'__frozenset__': True, 'data': [_to_jsonable(v) for v in obj]}
    if isinstance(obj, set):
        return {'__set__': True, 'data': [_to_jsonable(v) for v in obj]}
    if isinstance(obj, np.ndarray):
        return {'__ndarray__': True, 'data': obj.tolist(), 'dtype': str(obj.dtype)}
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, State):
        return {'__state__': True, 'blocks': [[b.id, [list(c) for c in b.coords]] for b in obj.blocks],
                'grid_shape': list(obj.grid_shape)}
    if isinstance(obj, Block):
        return {'__block__': True, 'id': obj.id, 'coords': [list(c) for c in obj.coords]}
    return obj


def _from_jsonable(obj):
    """递归从 JSON 还原 (恢复 numpy 数组 + State/Block)"""
    from 数据模型 import State, Block
    if isinstance(obj, dict):
        if obj.get('__ndarray__') is True:
            return np.array(obj['data'], dtype=obj.get('dtype', 'float64'))
        if obj.get('__state__') is True:
            blocks = [Block(b[0], [tuple(c) for c in b[1]]) for b in obj['blocks']]
            return State(blocks, tuple(obj['grid_shape']))
        if obj.get('__block__') is True:
            return Block(obj['id'], [tuple(c) for c in obj['coords']])
        if obj.get('__frozenset__') is True:
            return frozenset(_from_jsonable(v) for v in obj['data'])
        if obj.get('__set__') is True:
            return set(_from_jsonable(v) for v in obj['data'])
        return {k: _from_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_jsonable(v) for v in obj]
    return obj


class CacheManager:
    """存档缓存管理器

    使用方法:
        cm = CacheManager()              # 默认 ~/.huarongdao_cache
        cm = CacheManager('./my_cache')  # 自定义目录
        sid = cm.save_session(name, layout_data, basic, topo, tda, comm)
        sessions = cm.list_sessions()    # 返回元数据列表
        data = cm.load_session(sid)      # 加载完整数据
        cm.delete_session(sid)
    """

    MANIFEST = 'manifest.json'

    def __init__(self, cache_dir: Optional[str] = None):
        if cache_dir is None:
            home = Path.home()
            cache_dir = home / '.huarongdao_cache'
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.cache_dir / self.MANIFEST
        self._ensure_manifest()

    def _ensure_manifest(self):
        if not self.manifest_path.exists():
            self._write_manifest({'sessions': [], 'version': 1})

    def _read_manifest(self) -> dict:
        try:
            with open(self.manifest_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {'sessions': [], 'version': 1}

    def _write_manifest(self, data: dict):
        tmp = self.manifest_path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(self.manifest_path))

    def _session_file(self, session_id: str) -> Path:
        return self.cache_dir / f'{session_id}.json'

    def save_session(self, name: str, layout_data, basic_analysis: Optional[dict] = None,
                     topology_result: Optional[dict] = None,
                     tda_result: Optional[dict] = None,
                     community_result: Optional[dict] = None,
                     graph_stats: Optional[dict] = None) -> str:
        """保存一次解析会话, 返回 session_id"""
        session_id = time.strftime('%Y%m%d_%H%M%S')
        # 防止同一秒重复
        while self._session_file(session_id).exists():
            session_id = session_id + '_1'

        layout_dict = None
        if layout_data is not None:
            layout_dict = layout_data.to_dict()

        payload = {
            'session_id': session_id,
            'name': name or f'会话 {session_id}',
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'layout': layout_dict,
            'graph_stats': graph_stats or {},
            'basic_analysis': _to_jsonable(basic_analysis) if basic_analysis else None,
            'topology_result': _to_jsonable(topology_result) if topology_result else None,
            'tda_result': _to_jsonable(tda_result) if tda_result else None,
            'community_result': _to_jsonable(community_result) if community_result else None,
        }

        path = self._session_file(session_id)
        tmp = path.with_suffix('.tmp')
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(str(tmp), str(path))

        # 更新清单
        manifest = self._read_manifest()
        meta = {
            'session_id': session_id,
            'name': payload['name'],
            'created_at': payload['created_at'],
            'grid': f"{layout_data.rows}x{layout_data.cols}" if layout_data else '-',
            'n_blocks': len(layout_data.blocks) if layout_data else 0,
            'n_nodes': (graph_stats or {}).get('n_nodes', 0),
            'n_edges': (graph_stats or {}).get('n_edges', 0),
            'has_topo': topology_result is not None,
            'has_tda': tda_result is not None,
        }
        manifest['sessions'].insert(0, meta)
        # 限制最大数量 200
        if len(manifest['sessions']) > 200:
            old = manifest['sessions'][200:]
            manifest['sessions'] = manifest['sessions'][:200]
            for m in old:
                f = self._session_file(m['session_id'])
                if f.exists():
                    try:
                        f.unlink()
                    except Exception:
                        pass
        self._write_manifest(manifest)
        return session_id

    def list_sessions(self) -> List[dict]:
        """返回所有会话的元数据 (最新在前)"""
        return self._read_manifest().get('sessions', [])

    def load_session(self, session_id: str) -> Optional[dict]:
        """加载完整会话数据"""
        path = self._session_file(session_id)
        if not path.exists():
            return None
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return _from_jsonable(data)
        except Exception:
            return None

    def delete_session(self, session_id: str) -> bool:
        path = self._session_file(session_id)
        deleted = False
        if path.exists():
            try:
                path.unlink()
                deleted = True
            except Exception:
                pass
        manifest = self._read_manifest()
        manifest['sessions'] = [s for s in manifest['sessions'] if s['session_id'] != session_id]
        self._write_manifest(manifest)
        return deleted

    def rename_session(self, session_id: str, new_name: str) -> bool:
        path = self._session_file(session_id)
        if not path.exists():
            return False
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data['name'] = new_name
            tmp = path.with_suffix('.tmp')
            with open(tmp, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(str(tmp), str(path))
        except Exception:
            return False
        manifest = self._read_manifest()
        for s in manifest['sessions']:
            if s['session_id'] == session_id:
                s['name'] = new_name
                break
        self._write_manifest(manifest)
        return True

    def clear_all(self) -> int:
        """清空所有缓存, 返回删除数量"""
        manifest = self._read_manifest()
        n = len(manifest.get('sessions', []))
        for s in manifest.get('sessions', []):
            f = self._session_file(s['session_id'])
            if f.exists():
                try:
                    f.unlink()
                except Exception:
                    pass
        self._write_manifest({'sessions': [], 'version': 1})
        return n
