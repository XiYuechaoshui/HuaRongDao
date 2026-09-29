"""缓存查看面板: 嵌入主界面的存档管理组件

显示历史会话列表, 支持加载/删除/重命名/清空。
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Callable, Optional

from 缓存管理器 import CacheManager
from 主题 import Theme


class CachePanel(ttk.Frame):
    """缓存查看面板 (嵌入 Notebook 的一个 tab)"""

    def __init__(self, parent, cache_manager: CacheManager,
                 on_load_session: Optional[Callable[[dict], None]] = None,
                 **kwargs):
        super().__init__(parent, **kwargs)
        self.cm = cache_manager
        self.on_load_session = on_load_session
        self._sessions = []
        self._create_widgets()
        self.refresh()

    def _create_widgets(self):
        # 顶部工具栏
        # 用 grid 而不是"左标签 + 右按钮"的 pack 混排: 面板较窄时(约 313px)右对齐按钮会被裁掉,
        # 「清空全部」实测右边被切掉约 8px。grid 两列各自占满, 不再互相挤压。
        bar = ttk.Frame(self)
        bar.pack(fill=tk.X, padx=6, pady=(6, 4))
        bar.columnconfigure(0, weight=1)   # 左侧标题区(可伸缩)
        bar.columnconfigure(1, weight=0)   # 右侧按钮区(按需宽度)

        左 = ttk.Frame(bar)
        左.grid(row=0, column=0, sticky='w')
        ttk.Label(左, text="历史存档", font=("Microsoft YaHei UI", 10, "bold")).pack(side=tk.LEFT, padx=(4, 8))
        self.count_label = ttk.Label(左, text="(0)", foreground=Theme.MUTED)
        self.count_label.pack(side=tk.LEFT)

        右 = ttk.Frame(bar)
        右.grid(row=0, column=1, sticky='e')
        ttk.Button(右, text="刷新", command=self.refresh, padding=(6, 2)).pack(side=tk.LEFT, padx=2)
        ttk.Button(右, text="清空全部", command=self._clear_all, padding=(6, 2)).pack(side=tk.LEFT, padx=2)

        # 列表
        list_frame = ttk.Frame(self)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        columns = ('name', 'grid', 'blocks', 'nodes', 'edges', 'topo', 'tda', 'time')
        headers = {
            'name': '名称', 'grid': '棋盘', 'blocks': '方块',
            'nodes': '节点', 'edges': '边', 'topo': '拓扑', 'tda': 'TDA', 'time': '时间'
        }
        # 列宽按"实际内容 + 余量"定, 而不是拍脑袋给大值 (2026-09-17 用户反馈"名称再窄一点"):
        #   实测名称列内容最长 43px('4x4 4块'), 却给了 180px; 时间列内容 118px 给 140px。
        #   收窄后整表 680 -> 466px, 左侧面板加宽幅度也随之从 640 降到 560 (见 主窗口._on_left_tab_changed)。
        #   stretch=False 很关键: 默认 stretch=1 时 Tk 会按比例重分多余空间, 设的宽度会被改掉。
        widths = {'name': 72, 'grid': 46, 'blocks': 40, 'nodes': 48, 'edges': 48,
                  'topo': 40, 'tda': 40, 'time': 132}

        self.tree = ttk.Treeview(list_frame, columns=columns, show='headings', selectmode='browse')
        for c in columns:
            self.tree.heading(c, text=headers[c])
            self.tree.column(c, width=widths[c], minwidth=widths[c], stretch=False,
                             anchor=tk.W if c == 'name' else tk.CENTER)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        sb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.tree.yview)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=sb.set)

        self.tree.bind('<Double-1>', lambda e: self._load_selected())

        # 右键菜单
        self.ctx_menu = tk.Menu(self, tearoff=False, bg=Theme.PANEL, fg=Theme.TEXT,
                                activebackground=Theme.ACCENT, activeforeground='#ffffff')
        self.ctx_menu.add_command(label="加载", command=self._load_selected)
        self.ctx_menu.add_command(label="重命名...", command=self._rename_selected)
        self.ctx_menu.add_separator()
        self.ctx_menu.add_command(label="删除", command=self._delete_selected)
        self.tree.bind('<Button-3>', self._on_right_click)

        # 底部操作栏
        op = ttk.Frame(self)
        op.pack(fill=tk.X, padx=6, pady=(4, 8))
        ttk.Button(op, text="加载选中", style='Accent.TButton', command=self._load_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(op, text="重命名", command=self._rename_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(op, text="删除", command=self._delete_selected).pack(side=tk.LEFT, padx=2)

        # 详情预览
        detail_frame = ttk.LabelFrame(self, text="详情", padding=6)
        detail_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        self.detail_text = tk.Text(detail_frame, height=6, wrap=tk.WORD, relief='flat',
                                   bg=Theme.BG, fg=Theme.TEXT, font=('Consolas', 9))
        self.detail_text.pack(fill=tk.X)
        self.detail_text.configure(state=tk.DISABLED)
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._show_detail())

    def refresh(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._sessions = self.cm.list_sessions()
        for s in self._sessions:
            topo_mark = '✓' if s.get('has_topo') else '-'
            tda_mark = '✓' if s.get('has_tda') else '-'
            self.tree.insert('', tk.END, iid=s['session_id'], values=(
                s.get('name', ''),
                s.get('grid', '-'),
                s.get('n_blocks', 0),
                f"{s.get('n_nodes', 0):,}" if s.get('n_nodes') else '0',
                f"{s.get('n_edges', 0):,}" if s.get('n_edges') else '0',
                topo_mark, tda_mark,
                s.get('created_at', '')
            ))
        self.count_label.config(text=f"({len(self._sessions)})")

    def _on_right_click(self, event):
        item = self.tree.identify_row(event.y)
        if item:
            self.tree.selection_set(item)
            self.ctx_menu.tk_popup(event.x_root, event.y_root)

    def _selected_id(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0] if sel else None

    def _show_detail(self):
        sid = self._selected_id()
        if not sid:
            return
        data = self.cm.load_session(sid)
        self.detail_text.configure(state=tk.NORMAL)
        self.detail_text.delete('1.0', tk.END)
        if not data:
            self.detail_text.insert('1.0', "(无法加载该会话数据)")
            self.detail_text.configure(state=tk.DISABLED)
            return

        lines = []
        lines.append(f"名称: {data.get('name', '')}")
        lines.append(f"时间: {data.get('created_at', '')}")
        gs = data.get('graph_stats', {})
        if gs:
            lines.append(f"图: 节点 {gs.get('n_nodes', 0):,} | 边 {gs.get('n_edges', 0):,} | 生成 {gs.get('generation_time', 0):.2f}s")
        topo = data.get('topology_result') or {}
        hom = topo.get('homology') if isinstance(topo, dict) else None
        if hom:
            lines.append(f"拓扑: β0={hom.get('beta_0')} β1={hom.get('beta_1')} χ={hom.get('euler_characteristic')}")
        tda = data.get('tda_result') or {}
        stats = tda.get('stats') if isinstance(tda, dict) else None
        if stats:
            h0 = stats.get('0', {}).get('total_bars', 0) if '0' in stats else 0
            h1 = stats.get('1', {}).get('total_bars', 0) if '1' in stats else 0
            lines.append(f"TDA: H0={h0} H1={h1}")
        self.detail_text.insert('1.0', "\n".join(lines))
        self.detail_text.configure(state=tk.DISABLED)

    def _load_selected(self):
        sid = self._selected_id()
        if not sid:
            messagebox.showinfo("提示", "请先选择一个会话")
            return
        data = self.cm.load_session(sid)
        if not data:
            messagebox.showerror("错误", "无法加载该会话数据")
            return
        if self.on_load_session:
            self.on_load_session(data)

    def _rename_selected(self):
        sid = self._selected_id()
        if not sid:
            return
        current = ''
        for s in self._sessions:
            if s['session_id'] == sid:
                current = s.get('name', '')
                break
        dlg = tk.Toplevel(self)
        dlg.title("重命名")
        dlg.geometry("360x120")
        dlg.resizable(False, False)
        ttk.Label(dlg, text="新名称:").pack(pady=(12, 4))
        var = tk.StringVar(value=current)
        ttk.Entry(dlg, textvariable=var).pack(fill=tk.X, padx=12)
        btn_frame = ttk.Frame(dlg)
        btn_frame.pack(pady=8)

        def apply_rename():
            new_name = var.get().strip()
            if not new_name:
                return
            if self.cm.rename_session(sid, new_name):
                self.refresh()
                dlg.destroy()

        ttk.Button(btn_frame, text="确定", style='Accent.TButton', command=apply_rename).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_frame, text="取消", command=dlg.destroy).pack(side=tk.LEFT, padx=4)

    def _delete_selected(self):
        sid = self._selected_id()
        if not sid:
            return
        if messagebox.askyesno("确认", "确定删除该会话?"):
            self.cm.delete_session(sid)
            self.refresh()

    def _clear_all(self):
        if not self._sessions:
            return
        if messagebox.askyesno("确认", f"确定清空全部 {len(self._sessions)} 条缓存?"):
            n = self.cm.clear_all()
            self.refresh()
            messagebox.showinfo("完成", f"已清空 {n} 条缓存")
