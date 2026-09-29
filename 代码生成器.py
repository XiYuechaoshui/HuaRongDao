"""代码生成器: 将 LayoutData 转为标准格式 Python 代码"""
from typing import List, Tuple
from 数据模型 import LayoutData


class CodeGenerator:
    @staticmethod
    def generate(layout_data: LayoutData) -> str:
        code = f'''"""
华容道布局 - 标准格式
棋盘大小: {layout_data.rows}x{layout_data.cols}
方块数量: {len(layout_data.blocks)}
"""

from typing import List, Tuple

ROWS = {layout_data.rows}
COLS = {layout_data.cols}


class Block:
    def __init__(self, id: int, coords: List[Tuple[int, int]]):
        self.id = id
        self.coords = tuple(sorted((r, c) for r, c in coords))

    def __repr__(self):
        return f"Block(id={{self.id}}, coords={{list(self.coords)}})"


class State:
    def __init__(self, blocks: List[Block], grid_shape: Tuple[int, int]):
        self.blocks = tuple(blocks)
        self.grid_shape = grid_shape

    def __repr__(self):
        return f"State(blocks={{len(self.blocks)}}, grid={{self.grid_shape}})"


def get_layout_data():
    return {{
        'rows': ROWS,
        'cols': COLS,
        'blocks': {{
'''
        for bid, coords in layout_data.blocks.items():
            coords_str = ', '.join([f"({r}, {c})" for r, c in coords])
            code += f'            {bid}: [{coords_str}],\n'

        code += f'''        }}
    }}


def get_initial_state():
    data = get_layout_data()
    blocks = []
    for bid, coords in data['blocks'].items():
        blocks.append(Block(bid, coords))
    return State(blocks, (ROWS, COLS))


if __name__ == "__main__":
    state = get_initial_state()
    print(f"棋盘大小: {{state.grid_shape[0]}}x{{state.grid_shape[1]}}")
    print(f"方块数量: {{len(state.blocks)}}")
    for block in state.blocks:
        print(f"  方块 {{block.id}}: {{list(block.coords)}}")
'''
        return code
