"""布局导入器: 从 .py / .json / 剪贴板字符串加载初始状态"""
import json
import importlib.util
from typing import Optional

from 数据模型 import Block, State


class LayoutImporter:

    @staticmethod
    def from_python_file(file_path: str) -> Optional[State]:
        spec = importlib.util.spec_from_file_location("layout_module", file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, 'get_initial_state'):
            state = module.get_initial_state()
            # 如果导入的是简化版State(无occupied等), 转换为完整版
            if not hasattr(state, 'occupied'):
                blocks = []
                for b in state.blocks:
                    if isinstance(b, Block):
                        blocks.append(b)
                    else:
                        blocks.append(Block(b.id, list(b.coords)))
                return State(blocks, state.grid_shape)
            return state

        if hasattr(module, 'get_layout_data'):
            data = module.get_layout_data()
            return LayoutImporter._from_dict(data)

        return None

    @staticmethod
    def from_json_file(file_path: str) -> Optional[State]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LayoutImporter._from_dict(data)

    @staticmethod
    def from_json_string(json_str: str) -> Optional[State]:
        data = json.loads(json_str)
        return LayoutImporter._from_dict(data)

    @staticmethod
    def _from_dict(data: dict) -> State:
        rows, cols = data.get('rows', 3), data.get('cols', 3)
        blocks = []
        for bid_str, coords in data.get('blocks', {}).items():
            bid = int(bid_str)
            blocks.append(Block(bid, coords))
        return State(blocks, (rows, cols))
