"""布局编辑器: 棋盘绘制 GUI

提供可视化绘制华容道棋盘布局的功能, 支持导出为 Python 代码或 JSON。
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Dict, List, Tuple, Optional

from 数据模型 import LayoutData
from 代码生成器 import CodeGenerator


class LayoutBoardEditor:
    COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A',
              '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
              '#F8C471', '#82E0AA', '#F1948A', '#85929E']

    def __init__(self, rows: int = 4, cols: int = 4):
        self.rows = rows
        self.cols = cols
        self.grid = [[-1 for _ in range(cols)] for _ in range(rows)]
        self.blocks: Dict[int, List[Tuple[int, int]]] = {}
        self.next_id = 0
        self.current_block: Optional[int] = None
        self.history = []

    def start_block(self, row: int, col: int) -> bool:
        if not self._is_valid(row, col) or self.grid[row][col] != -1:
            return False
        self._save_history()
        block_id = self.next_id
        self.next_id += 1
        self.blocks[block_id] = [(row, col)]
        self.grid[row][col] = block_id
        self.current_block = block_id
        return True

    def add_cell(self, row: int, col: int) -> bool:
        if self.current_block is None:
            return False
        if not self._is_valid(row, col) or self.grid[row][col] != -1:
            return False
        current_coords = self.blocks[self.current_block]
        if not any(abs(r - row) + abs(c - col) == 1 for r, c in current_coords):
            return False
        self._save_history()
        self.blocks[self.current_block].append((row, col))
        self.grid[row][col] = self.current_block
        return True

    def finish_block(self):
        if self.current_block is not None:
            if len(self.blocks.get(self.current_block, [])) == 0:
                del self.blocks[self.current_block]
            self.current_block = None

    def delete_block(self, row: int, col: int):
        if not self._is_valid(row, col):
            return
        block_id = self.grid[row][col]
        if block_id == -1:
            return
        self._save_history()
        for r, c in self.blocks[block_id]:
            self.grid[r][c] = -1
        del self.blocks[block_id]
        if self.current_block == block_id:
            self.current_block = None

    def clear_all(self):
        if not self.blocks:
            return
        self._save_history()
        self.grid = [[-1 for _ in range(self.cols)] for _ in range(self.rows)]
        self.blocks = {}
        self.current_block = None
        self.next_id = 0

    def undo(self) -> bool:
        if not self.history:
            return False
        last_state = self.history.pop()
        self.grid = last_state['grid']
        self.blocks = last_state['blocks']
        self.next_id = last_state['next_id']
        self.current_block = None
        return True

    def get_layout_data(self) -> LayoutData:
        return LayoutData(self.rows, self.cols, self.blocks.copy())

    def _save_history(self):
        state = {
            'grid': [row[:] for row in self.grid],
            'blocks': {k: v[:] for k, v in self.blocks.items()},
            'next_id': self.next_id
        }
        self.history.append(state)
        if len(self.history) > 30:
            self.history.pop(0)

    def _is_valid(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def get_block_color(self, block_id: int) -> str:
        return self.COLORS[block_id % len(self.COLORS)]


class LayoutBoardEditorGUI:

    def __init__(self, root, on_layout_saved=None):
        self.root = root
        self.root.title("华容道绘制工具")
        self.root.geometry("1000x800")

        self.board = LayoutBoardEditor(4, 4)
        self.cell_size = 60
        self.on_layout_saved = on_layout_saved
        self._create_widgets()
        self._draw_board()
        self._update_status()

    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        size_frame = ttk.LabelFrame(left_frame, text="棋盘大小", padding="10")
        size_frame.pack(fill=tk.X, pady=(0, 10))

        size_row = ttk.Frame(size_frame)
        size_row.pack()
        ttk.Label(size_row, text="行:").pack(side=tk.LEFT, padx=(0, 5))
        self.rows_var = tk.StringVar(value="4")
        ttk.Spinbox(size_row, from_=2, to=8, width=5, textvariable=self.rows_var).pack(side=tk.LEFT, padx=(0, 10))
        ttk.Label(size_row, text="列:").pack(side=tk.LEFT, padx=(0, 5))
        self.cols_var = tk.StringVar(value="4")
        ttk.Spinbox(size_row, from_=2, to=8, width=5, textvariable=self.cols_var).pack(side=tk.LEFT)
        ttk.Button(size_frame, text="创建新棋盘", command=self._create_new_board).pack(fill=tk.X, pady=5)

        op_frame = ttk.LabelFrame(left_frame, text="操作", padding="10")
        op_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(op_frame, text="完成当前方块", command=self._finish_block).pack(fill=tk.X, pady=2)
        ttk.Button(op_frame, text="撤销 (Ctrl+Z)", command=self._undo).pack(fill=tk.X, pady=2)
        ttk.Button(op_frame, text="清空所有", command=self._clear_all).pack(fill=tk.X, pady=2)

        export_frame = ttk.LabelFrame(left_frame, text="导出", padding="10")
        export_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(export_frame, text="导出为代码", command=self._export_code).pack(fill=tk.X, pady=2)
        ttk.Button(export_frame, text="保存为JSON", command=self._save_json).pack(fill=tk.X, pady=2)

        info_frame = ttk.LabelFrame(left_frame, text="操作说明", padding="10")
        info_frame.pack(fill=tk.X)
        ttk.Label(info_frame, text="左键: 创建/扩展\n右键: 删除\nCtrl+Z: 撤销",
                  justify=tk.LEFT).pack()

        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        status_frame = ttk.Frame(right_frame)
        status_frame.pack(fill=tk.X, pady=(0, 5))
        self.status_label = ttk.Label(status_frame, text="就绪")
        self.status_label.pack(side=tk.LEFT)
        self.info_label = ttk.Label(status_frame, text="")
        self.info_label.pack(side=tk.RIGHT)

        canvas_frame = ttk.Frame(right_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.root.bind("<Control-z>", lambda e: self._undo())

    def _create_new_board(self):
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            if rows < 2 or rows > 8 or cols < 2 or cols > 8:
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入2-8之间的数字")
            return
        self.board = LayoutBoardEditor(rows, cols)
        self.cell_size = min(500 // max(rows, cols), 80)
        self._draw_board()
        self._update_status()

    def _draw_board(self):
        self.canvas.delete("all")
        rows, cols = self.board.rows, self.board.cols
        cell = self.cell_size

        width = cols * cell + 2
        height = rows * cell + 2
        self.canvas.config(width=width, height=height)

        for i in range(rows + 1):
            self.canvas.create_line(1, i * cell + 1, cols * cell + 1, i * cell + 1, fill='black', width=1)
        for j in range(cols + 1):
            self.canvas.create_line(j * cell + 1, 1, j * cell + 1, rows * cell + 1, fill='black', width=1)

        for block_id, coords in self.board.blocks.items():
            if not coords:
                continue
            color = self.board.get_block_color(block_id)
            coords_set = set(coords)
            # 逐格填充: 每格单独画 1x1 矩形, 避免非矩形方块包围盒覆盖其他方块
            for (r, c) in coords:
                x1 = c * cell + 1
                y1 = r * cell + 1
                x2 = (c + 1) * cell + 1
                y2 = (r + 1) * cell + 1
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
            # 外轮廓: 仅画与异块/空格相邻的边
            for (r, c) in coords:
                x1 = c * cell + 1
                y1 = r * cell + 1
                x2 = (c + 1) * cell + 1
                y2 = (r + 1) * cell + 1
                if (r - 1, c) not in coords_set:
                    self.canvas.create_line(x1, y1, x2, y1, fill='black', width=2)
                if (r + 1, c) not in coords_set:
                    self.canvas.create_line(x1, y2, x2, y2, fill='black', width=2)
                if (r, c - 1) not in coords_set:
                    self.canvas.create_line(x1, y1, x1, y2, fill='black', width=2)
                if (r, c + 1) not in coords_set:
                    self.canvas.create_line(x2, y1, x2, y2, fill='black', width=2)
            # 编号标注在质心
            n = len(coords)
            avg_r = sum(r for r, c in coords) / n
            avg_c = sum(c for r, c in coords) / n
            cx = int((avg_c + 0.5) * cell + 1)
            cy = int((avg_r + 0.5) * cell + 1)
            self.canvas.create_text(cx, cy, text=str(block_id), font=('Arial', 14, 'bold'))

        if self.board.current_block is not None:
            coords = self.board.blocks.get(self.board.current_block, [])
            for r, c in coords:
                x1 = c * cell + 2
                y1 = r * cell + 2
                x2 = (c + 1) * cell
                y2 = (r + 1) * cell
                self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=3)

    def _get_cell(self, event):
        col = event.x // self.cell_size
        row = event.y // self.cell_size
        if 0 <= row < self.board.rows and 0 <= col < self.board.cols:
            return row, col
        return None, None

    def _on_click(self, event):
        row, col = self._get_cell(event)
        if row is None:
            return

        if self.board.grid[row][col] == -1:
            if self.board.current_block is not None:
                if self.board.add_cell(row, col):
                    self.status_label.config(text=f"添加格子 ({row}, {col})")
                    self._draw_board()
                    self._update_status()
                    return
            if self.board.start_block(row, col):
                self.status_label.config(text=f"开始方块 ({row}, {col})")
                self._draw_board()
                self._update_status()
        else:
            block_id = self.board.grid[row][col]
            self.board.current_block = block_id
            self.status_label.config(text=f"选中方块 {block_id}")
            self._draw_board()

    def _on_right_click(self, event):
        row, col = self._get_cell(event)
        if row is None:
            return
        if self.board.grid[row][col] != -1:
            self.board.delete_block(row, col)
            self.status_label.config(text="删除方块")
            self._draw_board()
            self._update_status()

    def _finish_block(self):
        self.board.finish_block()
        self._draw_board()
        self._update_status()
        self.status_label.config(text="完成方块")

    def _undo(self):
        if self.board.undo():
            self._draw_board()
            self._update_status()
            self.status_label.config(text="撤销成功")
        else:
            self.status_label.config(text="没有可撤销的操作")

    def _clear_all(self):
        if messagebox.askyesno("确认", "确定要清空所有方块吗？"):
            self.board.clear_all()
            self._draw_board()
            self._update_status()
            self.status_label.config(text="已清空")

    def _update_status(self):
        blocks = len(self.board.blocks)
        cells = sum(len(c) for c in self.board.blocks.values())
        current = f"当前: {self.board.current_block}" if self.board.current_block is not None else "无激活"
        self.info_label.config(text=f"{current} | 方块: {blocks} | 格子: {cells}")

    def _export_code(self):
        if not self.board.blocks:
            messagebox.showwarning("警告", "棋盘为空")
            return

        layout_data = self.board.get_layout_data()
        code = CodeGenerator.generate(layout_data)

        dialog = tk.Toplevel(self.root)
        dialog.title("导出代码")
        dialog.geometry("700x500")

        text_frame = ttk.Frame(dialog)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        text_widget = tk.Text(text_frame, wrap=tk.WORD, font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert('1.0', code)
        text_widget.config(state=tk.DISABLED)

        scrollbar = ttk.Scrollbar(text_frame, orient=tk.VERTICAL, command=text_widget.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text_widget.config(yscrollcommand=scrollbar.set)

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(fill=tk.X, padx=10, pady=10)

        def copy_code():
            self.root.clipboard_clear()
            self.root.clipboard_append(code)
            messagebox.showinfo("成功", "代码已复制到剪贴板")

        def save_file():
            file_path = filedialog.asksaveasfilename(
                defaultextension=".py",
                filetypes=[("Python files", "*.py"), ("All files", "*.*")]
            )
            if file_path:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                messagebox.showinfo("成功", f"已保存到 {file_path}")
                if self.on_layout_saved:
                    self.on_layout_saved(file_path)

        ttk.Button(btn_frame, text="复制到剪贴板", command=copy_code).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="保存为文件", command=save_file).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="关闭", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)

    def _save_json(self):
        if not self.board.blocks:
            messagebox.showwarning("警告", "棋盘为空")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            layout_data = self.board.get_layout_data()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(layout_data.to_json())
            messagebox.showinfo("成功", f"已保存到 {file_path}")
            if self.on_layout_saved:
                self.on_layout_saved(file_path)
