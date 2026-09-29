"""主窗口 GUI: 华容道状态空间分析器

界面结构:
  顶部菜单栏 + 主工具栏
  ┌──────────┬─────────────────────┬──────────────┐
  │ 左侧控制  │   中央可视化区域     │  右侧分析     │
  │ 布局/生成 │  3D/谱图/条形码切换  │  Notebook分页 │
  └──────────┴─────────────────────┴──────────────┘
  底部状态栏 (状态/节点边数/进度)

功能: 加载布局 -> 生成状态空间 -> 拓扑分析 -> 可视化 -> 寻路 -> 导出报告
"""
import os
import time
import threading

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

from 数据模型 import State, LayoutData, Block
from 导入器 import LayoutImporter
from 状态空间 import StateSpaceGenerator
from 拓扑分析 import TopologyAnalyzer
from 持久同调 import TDAnalyzer
from 社区检测 import CommunityAnalyzer
from 可视化器 import HoverVisualizer
from 布局编辑器 import LayoutBoardEditor, LayoutBoardEditorGUI
from 缓存管理器 import CacheManager
from 缓存面板 import CachePanel
from 训练器面板 import TrainerPanel
from 模型管理面板 import ModelManagerPanel
from 字体工具 import CN_FONT, apply_cn_font
from AI模型 import TopologyFeatureML, SimpleGNN, TDADeepLearning
from 布局哈希 import (compute_layout_hash, check_layout_memory,
                     save_full_result, load_full_result,
                     layout_model_paths, global_model_paths,
                     set_model_base, get_model_base, get_model_dir,
                     scan_model_sources)


from 主题 import Theme


class StateSpaceAnalyzerGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("华容道状态空间分析器")
        self.root.geometry("1680x960")
        self.root.minsize(1200, 720)
        self.root.configure(bg=Theme.BG)

        self._apply_theme()

        # 核心组件
        self.generator = StateSpaceGenerator()
        self.visualizer = HoverVisualizer()
        self.cache = CacheManager()

        # 状态
        self.current_state = None
        self.generation_thread = None
        self._hovered_state = None
        self._last_display_key = None
        self._hover_after_id = None
        self._hover_mouse_pos = (0, 0)
        self._editor_redraw_after_id = None

        # 左侧页签状态 (用于「缓存」页自动加宽左侧面板, 见 _on_left_tab_changed)
        self._left_tab_last = 0            # 上一次选中的页签索引
        self._sash_left_default = None     # 非缓存页时的左侧面板宽度, 离开缓存页时还原

        # 3D 视角
        self.view_elev = 22
        self.view_azim = 35
        self.view_dist = 10.0
        self._dragging = False
        self._last_mouse_pos = None
        self._middle_button_pressed = False
        self._drag_redraw_after_id = None   # 拖拽视图重绘节流
        self._drag_view_pending = None       # 待应用的视角 (elev, azim)
        self._cached_edge_visibilities = []  # 拖拽时隐藏边的原 visibility 状态

        # 模型来源选择 (主程序可切换不同训练目录的模型)
        self.var_model_source = tk.StringVar(value=get_model_base())

        # 拓扑分析缓存
        self.topology_analyzer = None
        self.tda_analyzer = None
        self.community_analyzer = None
        self.topo_result = None
        self.tda_result = None
        self.community_result = None
        self._current_view = '3d'

        self._create_widgets()
        self._bind_events()
        self._bind_shortcuts()
        self._update_status("就绪")

    # ==================== 主题 ====================

    def _apply_theme(self):
        style = ttk.Style()
        try:
            if 'clam' in style.theme_names():
                style.theme_use('clam')
        except Exception:
            pass

        style.configure('.', background=Theme.BG, foreground=Theme.TEXT, font=Theme.FONT)
        style.configure('TFrame', background=Theme.BG)
        style.configure('Panel.TFrame', background=Theme.PANEL)
        style.configure('TLabel', background=Theme.BG, foreground=Theme.TEXT, font=Theme.FONT)
        style.configure('Title.TLabel', background=Theme.PANEL, foreground=Theme.ACCENT_TEXT,
                        font=Theme.FONT_TITLE)
        style.configure('Muted.TLabel', background=Theme.PANEL, foreground=Theme.MUTED, font=Theme.FONT)
        style.configure('Status.TLabel', background=Theme.PANEL, foreground=Theme.TEXT, font=Theme.FONT)
        style.configure('TLabelframe', background=Theme.PANEL, bordercolor=Theme.BORDER,
                        relief='solid', borderwidth=1)
        style.configure('TLabelframe.Label', background=Theme.PANEL, foreground=Theme.ACCENT_TEXT,
                        font=Theme.FONT_BOLD)

        # 次级按钮: 白底 + 淡紫描边 + 主色文字; 悬停时淡紫底
        style.configure('TButton', padding=(8, 3), font=Theme.FONT,
                        background=Theme.PANEL, foreground=Theme.ACCENT_TEXT,
                        bordercolor=Theme.BORDER, focuscolor=Theme.ACCENT_LIGHT,
                        relief='solid', borderwidth=1)
        style.map('TButton',
                  background=[('pressed', Theme.PANEL_ALT), ('active', Theme.PANEL_ALT),
                              ('disabled', Theme.BG)],
                  foreground=[('disabled', Theme.DISABLED)],
                  bordercolor=[('active', Theme.ACCENT_LIGHT), ('focus', Theme.ACCENT_LIGHT)])

        # 主按钮 (Accent): 深紫蓝底 + 白字 (实测 5.70:1)
        style.configure('Accent.TButton', background=Theme.ACCENT, foreground='#ffffff',
                        font=Theme.FONT_BOLD, bordercolor=Theme.ACCENT)
        style.map('Accent.TButton',
                  background=[('pressed', Theme.ACCENT_PRESS), ('active', Theme.ACCENT_HOVER),
                              ('disabled', Theme.PANEL_ALT)],
                  foreground=[('disabled', Theme.DISABLED)],
                  bordercolor=[('active', Theme.ACCENT_HOVER), ('pressed', Theme.ACCENT_PRESS)])

        # 危险按钮 (Danger): 蜜桃玫瑰加深 + 白字 (5.0:1)
        style.configure('Danger.TButton', background=Theme.DANGER, foreground='#ffffff',
                        font=Theme.FONT_BOLD, bordercolor=Theme.DANGER)
        style.map('Danger.TButton',
                  background=[('active', Theme.DANGER_HOVER), ('pressed', Theme.DANGER_HOVER),
                              ('disabled', Theme.PANEL_ALT)],
                  foreground=[('disabled', Theme.DISABLED)])

        style.configure('TNotebook', background=Theme.BG, bordercolor=Theme.BORDER,
                        tabmargins=(2, 3, 2, 0))
        style.configure('TNotebook.Tab', padding=(10, 4), font=Theme.FONT,
                        background=Theme.PANEL_ALT, foreground=Theme.MUTED,
                        bordercolor=Theme.BORDER)
        style.map('TNotebook.Tab',
                  background=[('selected', Theme.PANEL)],
                  foreground=[('selected', Theme.ACCENT_TEXT), ('active', Theme.ACCENT_TEXT)],
                  expand=[('selected', (0, 0, 0, 0))])

        style.configure('TProgressbar', background=Theme.ACCENT, troughcolor=Theme.BAR_BG,
                        bordercolor=Theme.BORDER, lightcolor=Theme.ACCENT_LIGHT,
                        darkcolor=Theme.ACCENT)
        style.configure('TEntry', fieldbackground=Theme.PANEL, bordercolor=Theme.BORDER,
                        lightcolor=Theme.BORDER, darkcolor=Theme.BORDER)
        style.map('TEntry', bordercolor=[('focus', Theme.ACCENT_LIGHT)])
        style.configure('TSpinbox', fieldbackground=Theme.PANEL, bordercolor=Theme.BORDER,
                        arrowcolor=Theme.ACCENT_TEXT, lightcolor=Theme.BORDER,
                        darkcolor=Theme.BORDER)
        style.configure('TSeparator', background=Theme.BORDER)
        style.configure('TCombobox', fieldbackground=Theme.PANEL, background=Theme.PANEL,
                        bordercolor=Theme.BORDER, arrowcolor=Theme.ACCENT_TEXT)
        style.map('TCombobox',
                  fieldbackground=[('readonly', Theme.PANEL)],
                  bordercolor=[('focus', Theme.ACCENT_LIGHT)])
        style.configure('TCheckbutton', background=Theme.PANEL, foreground=Theme.TEXT,
                        focuscolor=Theme.PANEL)
        style.map('TCheckbutton',
                  background=[('active', Theme.PANEL)],
                  foreground=[('disabled', Theme.DISABLED)])
        style.configure('Treeview', background=Theme.PANEL, fieldbackground=Theme.PANEL,
                        foreground=Theme.TEXT, bordercolor=Theme.BORDER, rowheight=22)
        style.configure('Treeview.Heading', background=Theme.PANEL_ALT, foreground=Theme.ACCENT_TEXT,
                        font=Theme.FONT_BOLD, relief='flat')
        style.map('Treeview.Heading', background=[('active', Theme.ACCENT_SOFT)])
        style.map('Treeview',
                  background=[('selected', Theme.ACCENT)],
                  foreground=[('selected', '#ffffff')])
        style.configure('TScrollbar', background=Theme.PANEL_ALT, troughcolor=Theme.BG,
                        bordercolor=Theme.BG, arrowcolor=Theme.MUTED)
        style.map('TScrollbar', background=[('active', Theme.ACCENT_LIGHT)])
        style.configure('Scale', background=Theme.PANEL, troughcolor=Theme.BAR_BG)
        style.configure('TSizegrip', background=Theme.BG)

    # ==================== 界面构建 ====================

    def _create_widgets(self):
        self._build_menubar()
        self._build_toolbar()

        body = ttk.Frame(self.root)
        body.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 0))

        main_paned = ttk.PanedWindow(body, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(main_paned, style='Panel.TFrame')
        main_paned.add(left_frame, weight=2)

        center_frame = ttk.Frame(main_paned, style='Panel.TFrame')
        main_paned.add(center_frame, weight=5)

        right_frame = ttk.Frame(main_paned, style='Panel.TFrame')
        main_paned.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_center_panel(center_frame)
        self._build_right_panel(right_frame)

        self._main_paned = main_paned
        self._sash_initialized = False

        self._build_statusbar()

        # 窗口显示后按 1:7:2 比例设置初始 sash 位置
        self.root.after(50, self._set_initial_sash)

    def _set_initial_sash(self):
        if self._sash_initialized:
            return
        try:
            w = self._main_paned.winfo_width()
            if w <= 1:
                self.root.after(50, self._set_initial_sash)
                return
            # 左 2 : 中 5 : 右 3 => sash0 在 2w/10, sash1 在 7w/10
            self._main_paned.sashpos(0, w * 2 // 10)
            self._main_paned.sashpos(1, w * 7 // 10)
            self._sash_initialized = True
        except Exception:
            pass


    def _build_menubar(self):
        menubar = tk.Menu(self.root, bg=Theme.PANEL, fg=Theme.TEXT,
                          activebackground=Theme.ACCENT, activeforeground='#ffffff',
                          borderwidth=0, font=Theme.FONT)
        m_file = tk.Menu(menubar, tearoff=False, bg=Theme.PANEL, fg=Theme.TEXT,
                         activebackground=Theme.ACCENT, activeforeground='#ffffff', font=Theme.FONT)
        m_file.add_command(label="绘制新布局...", command=self._open_layout_editor, accelerator="Ctrl+N")
        m_file.add_command(label="加载文件...", command=self._load_file)
        m_file.add_command(label="从剪贴板加载", command=self._load_clipboard)
        m_file.add_separator()
        m_file.add_command(label="导出报告...", command=self._export_report, accelerator="Ctrl+E")
        m_file.add_command(label="保存图片...", command=self._save_figure, accelerator="Ctrl+S")
        m_file.add_separator()
        m_file.add_command(label="退出", command=self.root.quit)
        menubar.add_cascade(label="文件", menu=m_file)

        m_analysis = tk.Menu(menubar, tearoff=False, bg=Theme.PANEL, fg=Theme.TEXT,
                             activebackground=Theme.ACCENT, activeforeground='#ffffff', font=Theme.FONT)
        m_analysis.add_command(label="生成状态空间", command=self._start_generation, accelerator="Ctrl+G")
        m_analysis.add_command(label="停止生成", command=self._stop_generation)
        m_analysis.add_separator()
        m_analysis.add_command(label="全拓扑分析", command=self._run_full_topology, accelerator="Ctrl+T")
        menubar.add_cascade(label="分析", menu=m_analysis)

        m_view = tk.Menu(menubar, tearoff=False, bg=Theme.PANEL, fg=Theme.TEXT,
                         activebackground=Theme.ACCENT, activeforeground='#ffffff', font=Theme.FONT)
        m_view.add_command(label="3D 状态图", command=lambda: self._switch_view('3d'), accelerator="F1")
        m_view.add_command(label="谱分布", command=lambda: self._switch_view('spectrum'), accelerator="F2")
        m_view.add_command(label="持久条形码", command=lambda: self._switch_view('barcode'), accelerator="F3")
        # AI增强图 原先只有左侧「视图」分组里的按钮, 菜单栏漏了它 ——
        # 而分析报告的文字写的是"点击顶栏「AI增强图」", 用户照做却找不到入口。
        m_view.add_command(label="AI增强图", command=lambda: self._switch_view('ai'), accelerator="F4")
        m_view.add_separator()
        m_view.add_command(label="重置视图", command=self._reset_view, accelerator="Ctrl+R")
        menubar.add_cascade(label="视图", menu=m_view)

        m_help = tk.Menu(menubar, tearoff=False, bg=Theme.PANEL, fg=Theme.TEXT,
                         activebackground=Theme.ACCENT, activeforeground='#ffffff', font=Theme.FONT)
        m_help.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=m_help)

        self.root.config(menu=menubar)

    def _build_toolbar(self):
        """精简工具栏: 对称性开关 + 信息条"""
        container = ttk.Frame(self.root, style='Panel.TFrame')
        container.pack(fill=tk.X, padx=6, pady=(4, 2))

        # 对称性快捷开关 (主界面常驻)
        self.toolbar_symmetry_var = tk.BooleanVar(value=True)
        self.toolbar_symmetry_chk = ttk.Checkbutton(
            container, text="对称性:开",
            variable=self.toolbar_symmetry_var,
            command=self._on_toolbar_symmetry_changed)
        self.toolbar_symmetry_chk.pack(side=tk.LEFT, padx=(6, 0))

        ttk.Separator(container, orient=tk.VERTICAL).pack(side=tk.LEFT, padx=8, fill=tk.Y)

        # 右侧信息
        self.info_label = ttk.Label(container, text="", style='Muted.TLabel')
        self.info_label.pack(side=tk.RIGHT, padx=6)

        # 分组定义: (key, 标题, 按钮构造函数)
        self._toolbar_groups = {}
        self._active_group = None

        groups = [
            ('layout', '布局', self._build_layout_group),
            ('generate', '分析', self._build_generate_group),
            ('view', '视图', self._build_view_group),
        ]
        for key, title, builder in groups:
            self._toolbar_groups[key] = {
                'title': title, 'builder': builder, 'frame': None
            }

    def _on_left_tab_changed(self, _event=None):
        """选中「缓存」页时自动加宽左侧面板, 离开时还原

        背景 (2026-09-17 用户反馈): 左侧面板按 2:5:3 初始比例只分到约 325px, 而缓存页里的
        表格要 684px、「历史存档」那一排(标签+计数+刷新+清空全部)也要约 280px ——
        结果"清空全部"右侧被裁掉约 8px。用户希望"点开缓存时让它展开"。

        做法: 切到缓存页时把第一道 sash 右移到 560px(上限 40% 窗口宽), 并把原宽度存起来,
        切走时还原。用户手动拖过的宽度会被这次还原覆盖, 属于可接受的取舍。
        (560 这个值来自缓存表收窄后的实际需求: 列宽合计 466px + 滚动条 + 边距 ≈ 500px,
         留一点余量; 收窄前表格要 680px, 当时必须开到 640。)
        """
        nb = getattr(self, '_left_nb', None)
        paned = getattr(self, '_main_paned', None)
        if nb is None or paned is None:
            return
        try:
            # 注意: ttk.Notebook.index('selected'/'current') 在本版本会报
            # "Invalid slave specification", 只能用 index(nb.select())
            当前 = nb.index(nb.select())
        except tk.TclError:
            return
        if 当前 == getattr(self, '_left_tab_last', 当前):
            return

        之前 = getattr(self, '_left_tab_last', 0)
        self._left_tab_last = 当前
        try:
            总宽 = paned.winfo_width()
            if 总宽 <= 1:
                return
            # 各页签需要的左侧宽度 (px)。训练页参数表+日志最宽, 需要最多空间。
            需要宽 = {2: 560, 3: 760, 4: 700}
            # 上限: 别把中间可视化区挤没了 (最多占窗口 60%, 训练页放宽到 70%)
            上限比例 = 0.7 if 当前 == 3 else 0.6
            目标宽 = min(需要宽.get(当前, 0), int(总宽 * 上限比例))
            在宽页 = 当前 in 需要宽
            之前在宽页 = 之前 in 需要宽
            if 在宽页 and not 之前在宽页:
                self._sash_left_default = paned.sashpos(0)   # 记住原始宽度, 离开时还原
            if 目标宽 > 0:
                paned.sashpos(0, max(self._sash_left_default or 0, 目标宽))
            elif 之前在宽页 and not 在宽页 and self._sash_left_default:
                paned.sashpos(0, self._sash_left_default)
        except tk.TclError:
            pass

    def _toggle_group(self, key):
        """切换激活的分组: 更新左侧面板操作区的按钮"""
        self._active_group = key
        # 更新组选择按钮样式
        for k, btn in getattr(self, '_group_buttons', {}).items():
            if k == key:
                btn.config(style='Accent.TButton')
            else:
                btn.config(style='TButton')
        # 在左侧面板操作区重建按钮
        self._rebuild_action_area(key)

    # ---- 模型来源选择回调 (三处联动的唯一入口) ----
    # 「模型来源」下拉 / 「训练」页的输出目录 / 「模型」页的当前库, 三者必须同步:
    #   否则会出现"下拉显示 训练数据_实验A, 但训练结果写进了 训练数据"这种静默错位。
    def _on_model_source_change(self):
        """左面板下拉切换模型来源"""
        base = self.var_model_source.get()
        set_model_base(base)
        self._同步模型来源(base, 来源='下拉')
        self._update_status(f"模型来源已切换: {base}")

    def _on_model_switched(self, 库名: str):
        """模型管理面板切换/新建/重命名模型库后回调"""
        self._同步模型来源(库名, 来源='模型页')
        self._update_status(f"模型来源已切换: {库名}")

    def _同步模型来源(self, 库名: str, 来源: str = ''):
        """把库名同步到三处 UI (不重复调用 set_model_base 以免递归)"""
        if not 库名:
            return
        # 1) 左面板下拉
        if hasattr(self, 'var_model_source') and self.var_model_source.get() != 库名:
            vals = list(self._model_source_combo['values']) if hasattr(self, '_model_source_combo') else []
            if 库名 not in vals:
                vals.append(库名)
                if hasattr(self, '_model_source_combo'):
                    self._model_source_combo['values'] = vals
            self.var_model_source.set(库名)
        # 2) 训练页输出目录
        训练器 = getattr(self, 'trainer_tab', None)
        if 训练器 is not None and 训练器.当前输出目录() != 库名:
            训练器.设置输出目录(库名)
        # 3) 模型页当前库 (刷新会按 get_model_base 重新高亮)
        模型页 = getattr(self, 'model_tab', None)
        if 模型页 is not None and 来源 != '模型页':
            模型页.refresh()
        set_model_base(库名)

    def _refresh_model_sources(self):
        """重新扫描可用训练目录"""
        sources = scan_model_sources()
        self._model_source_combo['values'] = sources
        # 若当前选择不在列表中, 回退到默认
        if self.var_model_source.get() not in sources:
            self.var_model_source.set(sources[0] if sources else '训练数据')
            set_model_base(self.var_model_source.get())
        self._update_status(f"已刷新可用模型来源 ({len(sources)} 个)")

    # ---- 各组按钮构造 (在左侧面板操作区内垂直排列) ----

    def _build_layout_group(self, parent):
        ttk.Button(parent, text="新建布局", command=self._open_layout_editor).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="加载文件", command=self._load_file).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="从剪贴板加载", command=self._load_clipboard).pack(fill=tk.X, pady=1)

    def _build_generate_group(self, parent):
        self.generate_btn = ttk.Button(parent, text="生成状态空间", style='Accent.TButton',
                                       command=self._start_generation)
        self.generate_btn.pack(fill=tk.X, pady=1)
        self.stop_btn = ttk.Button(parent, text="停止", style='Danger.TButton',
                                   command=self._stop_generation, state=tk.DISABLED)
        self.stop_btn.pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="全拓扑分析", style='Accent.TButton',
                   command=self._run_full_topology).pack(fill=tk.X, pady=1)

    def _build_view_group(self, parent):
        self.btn_view_3d = ttk.Button(parent, text="3D 状态图",
                                      command=lambda: self._switch_view('3d'))
        self.btn_view_3d.pack(fill=tk.X, pady=1)
        self.btn_view_spectrum = ttk.Button(parent, text="谱分布",
                                            command=lambda: self._switch_view('spectrum'))
        self.btn_view_spectrum.pack(fill=tk.X, pady=1)
        self.btn_view_barcode = ttk.Button(parent, text="持久条形码",
                                           command=lambda: self._switch_view('barcode'))
        self.btn_view_barcode.pack(fill=tk.X, pady=1)
        self.btn_view_ai = ttk.Button(parent, text="AI增强图",
                                      command=lambda: self._switch_view('ai'))
        self.btn_view_ai.pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="重置视图", command=self._reset_view).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="导出报告", command=self._export_report).pack(fill=tk.X, pady=1)
        ttk.Button(parent, text="保存图片", command=self._save_figure).pack(fill=tk.X, pady=1)
        self._update_view_buttons()

    def _build_left_panel(self, parent):
        nb = ttk.Notebook(parent)
        nb.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # ---- Tab 1: 控制 ----
        ctrl_tab = ttk.Frame(nb, style='Panel.TFrame')
        nb.add(ctrl_tab, text="控制")

        canvas_frame = ttk.Frame(ctrl_tab, style='Panel.TFrame')
        canvas_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(canvas_frame, bg=Theme.PANEL, highlightthickness=0, borderwidth=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        inner = ttk.Frame(canvas, style='Panel.TFrame')

        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw', width=248)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        canvas.bind('<Enter>', lambda e: canvas.bind_all('<MouseWheel>',
            lambda ev: canvas.yview_scroll(int(-1 * (ev.delta / 120)), 'units')))
        canvas.bind('<Leave>', lambda e: canvas.unbind_all('<MouseWheel>'))

        # 组选择按钮行 (布局/分析/视图)
        sel_frame = ttk.Frame(inner)
        sel_frame.pack(fill=tk.X, padx=4, pady=(6, 2))
        self._group_buttons = {}
        # 用 grid + uniform 保证三个按钮**永远等宽**:
        #   pack(side=LEFT, expand=True, fill=X) 只在"有剩余空间"时才均分 —— 左侧面板只有 248px 宽,
        #   而三个按钮按默认内边距要 279px, 放不下时最后打包的「视图」会被压到 48px (实测)。
        #   uniform 让三列宽度强制一致, 配合 sticky='ew' 各自撑满自己的列。
        for 列, (key, title) in enumerate([('layout', '布局'), ('generate', '分析'), ('view', '视图')]):
            btn = ttk.Button(sel_frame, text=title, padding=(2, 2),
                             command=lambda k=key: self._toggle_group(k))
            btn.grid(row=0, column=列, sticky='ew', padx=1)
            self._group_buttons[key] = btn
        for 列 in range(3):
            sel_frame.columnconfigure(列, weight=1, uniform='分组')

        # 模型来源选择器 (选择使用哪个训练目录的模型)
        src_frame = ttk.LabelFrame(inner, text="模型来源", padding="6")
        src_frame.pack(fill=tk.X, padx=6, pady=(2, 4))
        self._model_source_combo = ttk.Combobox(
            src_frame, textvariable=self.var_model_source, state='readonly',
            values=scan_model_sources())
        self._model_source_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self._model_source_combo.bind('<<ComboboxSelected>>',
            lambda e: self._on_model_source_change())
        ttk.Button(src_frame, text="刷新", padding=(4, 1),
                   command=self._refresh_model_sources).pack(side=tk.RIGHT, padx=(4, 0))

        # 操作按钮容器 (随组切换内容)
        self._action_area = ttk.Frame(inner)
        self._action_area.pack(fill=tk.X, padx=6, pady=(2, 4))

        # 当前布局信息 (只读展示, 导入功能在"绘制"Tab)
        info_frame = ttk.LabelFrame(inner, text="当前布局", padding="10")
        info_frame.pack(fill=tk.X, padx=6, pady=(6, 4))
        self.layout_info_label = ttk.Label(info_frame, text="未加载",
                                           style='Muted.TLabel', wraplength=222)
        self.layout_info_label.pack(fill=tk.X)

        # 生成控制
        gen_frame = ttk.LabelFrame(inner, text="生成", padding="10")
        gen_frame.pack(fill=tk.X, padx=6, pady=4)

        self.status_label = ttk.Label(gen_frame, text="状态: 就绪", style='Status.TLabel')
        self.status_label.pack(anchor=tk.W, pady=(0, 4))

        self.progress_label = ttk.Label(gen_frame, text="进度: -", style='Muted.TLabel')
        self.progress_label.pack(anchor=tk.W, pady=(0, 4))

        self.progress_bar = ttk.Progressbar(gen_frame, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 6))

        steps_frame = ttk.Frame(gen_frame)
        steps_frame.pack(fill=tk.X)
        ttk.Label(steps_frame, text="展示步数:", style='Status.TLabel').pack(side=tk.LEFT, padx=(0, 5))
        self.max_steps_var = tk.IntVar(value=1)
        ttk.Spinbox(steps_frame, from_=1, to=5, width=5, textvariable=self.max_steps_var).pack(side=tk.LEFT)

        # 对称性开关 (与工具栏同步, 这里保留引用但不重复显示)
        self.symmetry_var = self.toolbar_symmetry_var
        self.symmetry_chk = self.toolbar_symmetry_chk
        self._sym_hint = None

        # 状态预览
        preview_frame = ttk.LabelFrame(inner, text="状态预览", padding="6")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self.fig_hover = Figure(figsize=(3.2, 3.2), dpi=90)
        self.fig_hover.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.04)
        self.ax_hover = self.fig_hover.add_subplot(111)
        self.ax_hover.text(0.5, 0.5, "悬停或点击质点\n可查看状态",
                           ha='center', va='center', fontsize=10, alpha=0.5,
                           fontproperties=CN_FONT)
        self.ax_hover.axis('off')

        self.canvas_hover = FigureCanvasTkAgg(self.fig_hover, master=preview_frame)
        self.canvas_hover.draw()
        self.canvas_hover.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.state_info_label = ttk.Label(preview_frame, text="", wraplength=232, style='Muted.TLabel')
        self.state_info_label.pack(fill=tk.X, pady=(4, 0))

        # ---- Tab 2: 绘制 ----
        draw_tab = ttk.Frame(nb, style='Panel.TFrame')
        nb.add(draw_tab, text="绘制")
        self._build_embedded_editor(draw_tab)

        # ---- Tab 3: 缓存 ----
        self.cache_tab = CachePanel(parent, self.cache,
                                    on_load_session=self._on_load_cached_session,
                                    style='Panel.TFrame')
        nb.add(self.cache_tab, text="缓存")

        # ---- Tab 4: 训练 ----
        # 拓扑训练器嵌进来 (与独立窗口共用 训练器面板.TrainerPanel 同一份代码,
        # 启动器 拓扑训练器.py 也调它, 不会再有两套 UI 各自漂移)
        self.trainer_tab = TrainerPanel(parent, 独立模式=False,
                                        初始输出目录=get_model_base() or '训练数据',
                                        风格='Panel.TFrame')
        nb.add(self.trainer_tab, text="训练")

        # ---- Tab 5: 模型 ----
        # 模型库的新建/重命名/删除/切换 + 布局级存档清理
        self.model_tab = ModelManagerPanel(parent, on_switch=self._on_model_switched,
                                           style='Panel.TFrame')
        nb.add(self.model_tab, text="模型")

        self._left_nb = nb

        # 初始化默认操作区 (分析组)
        self._toggle_group('generate')

    def _rebuild_action_area(self, key):
        """重建左侧面板操作区的按钮 (根据当前工具栏分组)"""
        if not hasattr(self, '_action_area'):
            return
        # 清除旧按钮
        for child in self._action_area.winfo_children():
            child.destroy()
        # 销毁控件后必须把实例引用一起清掉: 否则 hasattr(self, 'generate_btn') 仍为 True,
        # 后面的 .config() 会抛 TclError "invalid command name" —— 而打包配置是 console=False,
        # 回溯无处可看, 表现就是"点了没反应"(生成完成不画图 / F2~F4 切不了视图)。
        for _attr in ('generate_btn', 'stop_btn', 'btn_view_3d',
                      'btn_view_spectrum', 'btn_view_barcode', 'btn_view_ai'):
            if hasattr(self, _attr):
                delattr(self, _attr)
        # 构建新按钮
        g = self._toolbar_groups.get(key)
        if g:
            g['builder'](self._action_area)

    def _load_file(self):
        """合并加载: 自动识别 .py / .json 文件格式"""
        file_path = filedialog.askopenfilename(
            title="选择布局文件",
            filetypes=[("布局文件", "*.py *.json"), ("Python", "*.py"), ("JSON", "*.json"), ("所有文件", "*.*")])
        if not file_path:
            return
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.py':
                state = LayoutImporter.from_python_file(file_path)
            elif ext == '.json':
                state = LayoutImporter.from_json_file(file_path)
            else:
                messagebox.showerror("错误", "不支持的文件格式")
                return
            self.current_state = state
            self._on_loaded(f"已加载: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", str(e))

    def _build_center_panel(self, parent):
        header = ttk.Frame(parent, style='Panel.TFrame')
        header.pack(fill=tk.X, padx=2, pady=(1, 1))
        ttk.Label(header, text="可视化视图", style='Title.TLabel').pack(side=tk.LEFT, padx=4)
        self.view_title_label = ttk.Label(header, text="3D 状态空间", style='Muted.TLabel')
        self.view_title_label.pack(side=tk.RIGHT, padx=6)

        chart_frame = ttk.Frame(parent, style='Panel.TFrame')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=1)

        # dpi=90 降低渲染像素量
        self.fig = Figure(figsize=(8, 8), dpi=90, facecolor=Theme.PANEL)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.02)
        self.ax_graph = self.fig.add_subplot(111, projection='3d')
        self.ax_graph.set_title("3D 状态空间视图", fontproperties=CN_FONT, fontsize=11, pad=8)
        self.ax_graph.set_facecolor(Theme.PANEL)
        self.ax_graph.set_axis_off()

        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, chart_frame)
        toolbar.update()
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_right_panel(self, parent):
        header = ttk.Frame(parent, style='Panel.TFrame')
        header.pack(fill=tk.X, padx=4, pady=(2, 1))
        ttk.Label(header, text="分析结果", style='Title.TLabel').pack(side=tk.LEFT, padx=6)

        nb_frame = ttk.Frame(parent, style='Panel.TFrame')
        nb_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        analysis_nb = ttk.Notebook(nb_frame)
        analysis_nb.pack(fill=tk.BOTH, expand=True)

        tab_basic = ttk.Frame(analysis_nb, style='Panel.TFrame', padding="8")
        analysis_nb.add(tab_basic, text="基本")
        self.analysis_text = scrolledtext.ScrolledText(
            tab_basic, font=('Consolas', 9), bg=Theme.PANEL, fg=Theme.TEXT,
            relief='flat', borderwidth=0, wrap=tk.WORD)
        self.analysis_text.pack(fill=tk.BOTH, expand=True)

        tab_topo = ttk.Frame(analysis_nb, style='Panel.TFrame', padding="8")
        analysis_nb.add(tab_topo, text="拓扑")
        self.topo_text = scrolledtext.ScrolledText(
            tab_topo, font=('Consolas', 9), bg=Theme.PANEL, fg=Theme.TEXT,
            relief='flat', borderwidth=0, wrap=tk.WORD)
        self.topo_text.pack(fill=tk.BOTH, expand=True)

        tab_tda = ttk.Frame(analysis_nb, style='Panel.TFrame', padding="8")
        analysis_nb.add(tab_tda, text="TDA")
        self.tda_text = scrolledtext.ScrolledText(
            tab_tda, font=('Consolas', 9), bg=Theme.PANEL, fg=Theme.TEXT,
            relief='flat', borderwidth=0, wrap=tk.WORD)
        self.tda_text.pack(fill=tk.BOTH, expand=True)

        self.analysis_nb = analysis_nb


    def _build_embedded_editor(self, parent):
        """嵌入式布局编辑器 (非弹窗)"""
        wrap = ttk.Frame(parent, style='Panel.TFrame')
        wrap.pack(fill=tk.BOTH, expand=True)

        # 棋盘大小
        size_frame = ttk.LabelFrame(wrap, text="棋盘大小", padding="8")
        size_frame.pack(fill=tk.X, padx=6, pady=(6, 4))
        row_sz = ttk.Frame(size_frame)
        row_sz.pack()
        ttk.Label(row_sz, text="行:", style='Status.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.ed_rows_var = tk.StringVar(value="4")
        ttk.Spinbox(row_sz, from_=2, to=8, width=4, textvariable=self.ed_rows_var).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Label(row_sz, text="列:", style='Status.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.ed_cols_var = tk.StringVar(value="4")
        ttk.Spinbox(row_sz, from_=2, to=8, width=4, textvariable=self.ed_cols_var).pack(side=tk.LEFT)
        ttk.Button(size_frame, text="创建新棋盘", command=self._editor_new_board).pack(fill=tk.X, pady=(4, 0))

        # 操作按钮
        op_frame = ttk.LabelFrame(wrap, text="操作", padding="8")
        op_frame.pack(fill=tk.X, padx=6, pady=4)
        ttk.Button(op_frame, text="完成当前方块", command=self._editor_finish_block).pack(fill=tk.X, pady=1)
        ttk.Button(op_frame, text="撤销", command=self._editor_undo).pack(fill=tk.X, pady=1)
        ttk.Button(op_frame, text="清空所有", command=self._editor_clear_all).pack(fill=tk.X, pady=1)

        # 状态行
        self.ed_status = ttk.Label(wrap, text="左键:创建/扩展  右键:删除", style='Muted.TLabel')
        self.ed_status.pack(fill=tk.X, padx=8, pady=(2, 0))
        self.ed_info = ttk.Label(wrap, text="", style='Status.TLabel')
        self.ed_info.pack(fill=tk.X, padx=8)

        # 画布
        canvas_wrap = ttk.Frame(wrap, style='Panel.TFrame')
        canvas_wrap.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)
        self.ed_canvas = tk.Canvas(canvas_wrap, bg='white', highlightthickness=0)
        self.ed_canvas.pack(fill=tk.BOTH, expand=True)
        self.ed_canvas.bind("<Button-1>", self._editor_on_click)
        self.ed_canvas.bind("<Button-3>", self._editor_on_right_click)
        self.ed_canvas.bind("<Configure>", self._on_editor_configure)

        # 导入按钮
        bottom = ttk.Frame(wrap, style='Panel.TFrame')
        bottom.pack(fill=tk.X, padx=6, pady=(0, 6))
        ttk.Button(bottom, text="导入到解析", style='Accent.TButton',
                   command=self._import_from_editor).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(bottom, text="保存JSON", command=self._editor_save_json).pack(side=tk.LEFT)

        # 编辑器数据模型
        self.editor = LayoutBoardEditor(4, 4)
        self._editor_cell_size = 48
        self._editor_redraw()
        self._editor_update_info()

    def _on_editor_configure(self, event):
        """编辑器尺寸变化时节流重绘 (100ms 内只重绘一次)"""
        if self._editor_redraw_after_id is not None:
            return
        self._editor_redraw_after_id = self.root.after(100, self._do_editor_redraw)

    def _do_editor_redraw(self):
        self._editor_redraw_after_id = None
        self._editor_redraw()

    # ==================== 嵌入式编辑器事件 ====================

    def _editor_new_board(self):
        try:
            rows = int(self.ed_rows_var.get())
            cols = int(self.ed_cols_var.get())
            if not (2 <= rows <= 8 and 2 <= cols <= 8):
                raise ValueError
        except ValueError:
            messagebox.showerror("错误", "请输入2-8之间的数字")
            return
        self.editor = LayoutBoardEditor(rows, cols)
        self._editor_redraw()
        self._editor_update_info()

    def _editor_redraw(self):
        c = self.ed_canvas
        c.delete("all")
        rows, cols = self.editor.rows, self.editor.cols
        w = max(c.winfo_width(), cols * 30 + 2)
        h = max(c.winfo_height(), rows * 30 + 2)
        cell = min((w - 2) // cols, (h - 2) // rows, 60)
        cell = max(cell, 20)
        self._editor_cell_size = cell

        for i in range(rows + 1):
            c.create_line(1, i * cell + 1, cols * cell + 1, i * cell + 1, fill='#888', width=1)
        for j in range(cols + 1):
            c.create_line(j * cell + 1, 1, j * cell + 1, rows * cell + 1, fill='#888', width=1)

        for block_id, coords in self.editor.blocks.items():
            if not coords:
                continue
            color = self.editor.get_block_color(block_id)
            coords_set = set(coords)
            # 逐格填充: 避免非矩形方块包围盒覆盖其他方块
            for (r, cc) in coords:
                x1 = cc * cell + 1
                y1 = r * cell + 1
                x2 = (cc + 1) * cell + 1
                y2 = (r + 1) * cell + 1
                c.create_rectangle(x1, y1, x2, y2, fill=color, outline='')
            # 外轮廓: 仅画与异块/空格相邻的边
            for (r, cc) in coords:
                x1 = cc * cell + 1
                y1 = r * cell + 1
                x2 = (cc + 1) * cell + 1
                y2 = (r + 1) * cell + 1
                if (r - 1, cc) not in coords_set:
                    c.create_line(x1, y1, x2, y1, fill='black', width=2)
                if (r + 1, cc) not in coords_set:
                    c.create_line(x1, y2, x2, y2, fill='black', width=2)
                if (r, cc - 1) not in coords_set:
                    c.create_line(x1, y1, x1, y2, fill='black', width=2)
                if (r, cc + 1) not in coords_set:
                    c.create_line(x2, y1, x2, y2, fill='black', width=2)
            # 编号标注在质心
            n = len(coords)
            avg_r = sum(r for r, _ in coords) / n
            avg_c = sum(cc for _, cc in coords) / n
            c.create_text(int((avg_c + 0.5) * cell + 1), int((avg_r + 0.5) * cell + 1),
                          text=str(block_id), font=('Arial', max(10, cell // 3), 'bold'))

        if self.editor.current_block is not None:
            coords = self.editor.blocks.get(self.editor.current_block, [])
            for r, cc in coords:
                x1 = cc * cell + 2
                y1 = r * cell + 2
                x2 = (cc + 1) * cell
                y2 = (r + 1) * cell
                c.create_rectangle(x1, y1, x2, y2, outline='red', width=3)

    def _editor_get_cell(self, event):
        col = event.x // self._editor_cell_size
        row = event.y // self._editor_cell_size
        if 0 <= row < self.editor.rows and 0 <= col < self.editor.cols:
            return row, col
        return None, None

    def _editor_on_click(self, event):
        row, col = self._editor_get_cell(event)
        if row is None:
            return
        if self.editor.grid[row][col] == -1:
            if self.editor.current_block is not None:
                if self.editor.add_cell(row, col):
                    self._editor_redraw()
                    self._editor_update_info()
                    return
            if self.editor.start_block(row, col):
                self._editor_redraw()
                self._editor_update_info()
        else:
            block_id = self.editor.grid[row][col]
            self.editor.current_block = block_id
            self._editor_redraw()
            self._editor_update_info()

    def _editor_on_right_click(self, event):
        row, col = self._editor_get_cell(event)
        if row is None:
            return
        if self.editor.grid[row][col] != -1:
            self.editor.delete_block(row, col)
            self._editor_redraw()
            self._editor_update_info()

    def _editor_finish_block(self):
        self.editor.finish_block()
        self._editor_redraw()
        self._editor_update_info()

    def _editor_undo(self):
        if self.editor.undo():
            self._editor_redraw()
            self._editor_update_info()

    def _editor_clear_all(self):
        if messagebox.askyesno("确认", "确定要清空所有方块吗？"):
            self.editor.clear_all()
            self._editor_redraw()
            self._editor_update_info()

    def _editor_update_info(self):
        blocks = len(self.editor.blocks)
        cells = sum(len(c) for c in self.editor.blocks.values())
        current = f"当前: {self.editor.current_block}" if self.editor.current_block is not None else "无激活"
        self.ed_info.config(text=f"{current} | 方块: {blocks} | 格子: {cells}")

    def _editor_save_json(self):
        if not self.editor.blocks:
            messagebox.showwarning("警告", "棋盘为空")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")])
        if file_path:
            layout_data = self.editor.get_layout_data()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(layout_data.to_json())
            messagebox.showinfo("成功", f"已保存到 {file_path}")

    def _import_from_editor(self):
        """将当前绘制棋盘直接导入解析"""
        if not self.editor.blocks:
            messagebox.showwarning("警告", "当前画布为空，请先绘制方块")
            return
        layout_data = self.editor.get_layout_data()
        blocks = [Block(int(bid), coords) for bid, coords in layout_data.blocks.items()]
        state = State(blocks, (layout_data.rows, layout_data.cols))
        if not state.is_valid():
            messagebox.showerror("错误", "当前布局不合法 (方块重叠或越界)")
            return
        self.current_state = state
        self._on_loaded(f"已导入绘制布局 ({layout_data.rows}x{layout_data.cols}, {len(blocks)}方块)")
        # 切回控制页
        self._left_nb.select(0)

    def _build_statusbar(self):
        bar = ttk.Frame(self.root, style='Panel.TFrame')
        bar.pack(fill=tk.X, padx=6, pady=(2, 4))

        self.sb_status = ttk.Label(bar, text="就绪", style='Status.TLabel')
        self.sb_status.pack(side=tk.LEFT, padx=6)

        ttk.Separator(bar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=4)

        self.sb_graph = ttk.Label(bar, text="节点: 0 | 边: 0", style='Muted.TLabel')
        self.sb_graph.pack(side=tk.LEFT, padx=6)

        # 视图可以用 F1~F4 直接切 (菜单「视图」里也列了同样的快捷键), 这里给个常驻提示
        self.sb_keys = ttk.Label(bar, text="视图快捷键: F1 3D / F2 谱 / F3 条形码 / F4 AI增强",
                                 style='Muted.TLabel')
        self.sb_keys.pack(side=tk.LEFT, padx=10)

        self.sb_view = ttk.Label(bar, text="视图: 3D", style='Muted.TLabel')
        self.sb_view.pack(side=tk.RIGHT, padx=6)

    # ==================== 快捷键 ====================

    def _bind_shortcuts(self):
        self.root.bind('<Control-n>', lambda e: self._open_layout_editor())
        self.root.bind('<Control-g>', lambda e: self._start_generation())
        self.root.bind('<Control-t>', lambda e: self._run_full_topology())
        self.root.bind('<Control-e>', lambda e: self._export_report())
        self.root.bind('<Control-s>', lambda e: self._save_figure())
        self.root.bind('<Control-r>', lambda e: self._reset_view())
        self.root.bind('<F1>', lambda e: self._switch_view('3d'))
        self.root.bind('<F2>', lambda e: self._switch_view('spectrum'))
        self.root.bind('<F3>', lambda e: self._switch_view('barcode'))
        self.root.bind('<F4>', lambda e: self._switch_view('ai'))
        # 窗口恢复显示时强制重绘，解决后台运行再打开出现黑框的问题
        self.root.bind('<Map>', self._on_window_map)
        self.root.bind('<Visibility>', self._on_window_visibility)
        # 绑定 canvas 的 Tk 组件暴露事件（部分Windows版本需要）
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.get_tk_widget().bind('<Expose>', self._on_canvas_expose)
        if hasattr(self, 'canvas_hover') and self.canvas_hover is not None:
            self.canvas_hover.get_tk_widget().bind('<Expose>', self._on_hover_canvas_expose)

    # ==================== 窗口重绘修复（后台恢复黑框问题） ====================

    def _on_window_map(self, event):
        """窗口从最小化/隐藏恢复显示时触发，强制重绘所有 matplotlib canvas"""
        if event.widget is not self.root:
            return
        # 使用 after 延迟一帧再重绘，给 Tk 窗口恢复缓冲时间
        self.root.after(10, self._force_redraw_all_canvases)

    def _on_window_visibility(self, event):
        """窗口可见性变化时触发（从其他遮挡后露出），部分 Windows 版本需要"""
        if not hasattr(event, 'state'):
            return
        # VisibilityUnobscured=0, VisibilityPartiallyObscured=1, VisibilityFullyObscured=2
        if event.state in (0, 1):
            self.root.after(10, self._force_redraw_all_canvases)

    def _on_canvas_expose(self, event):
        """主 canvas Tk 组件暴露时触发"""
        if hasattr(self, 'canvas') and self.canvas is not None:
            try:
                self.canvas.draw_idle()
            except Exception:
                pass

    def _on_hover_canvas_expose(self, event):
        """预览 canvas Tk 组件暴露时触发"""
        if hasattr(self, 'canvas_hover') and self.canvas_hover is not None:
            try:
                self.canvas_hover.draw_idle()
            except Exception:
                pass

    def _force_redraw_all_canvases(self):
        """强制重绘所有 matplotlib canvas，消除后台恢复的黑框"""
        canvases = []
        if hasattr(self, 'canvas') and self.canvas is not None:
            canvases.append(self.canvas)
        if hasattr(self, 'canvas_hover') and self.canvas_hover is not None:
            canvases.append(self.canvas_hover)
        for c in canvases:
            try:
                c.draw_idle()
            except Exception:
                pass
            # 额外 flush 一次 Tk 底层事件队列，确保重绘请求被处理
            try:
                c.flush_events()
            except Exception:
                pass

    # ==================== 事件绑定 ====================

    def _bind_events(self):
        self.fig.canvas.mpl_connect('pick_event', self._on_3d_pick)
        self.fig.canvas.mpl_connect('button_press_event', self._on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self._on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', self._on_mouse_scroll)
        # 左侧页签切换 -> 「缓存」页自动加宽左侧面板
        if hasattr(self, '_left_nb'):
            self._left_nb.bind('<<NotebookTabChanged>>', self._on_left_tab_changed)

    def _on_3d_pick(self, event):
        if not hasattr(event.artist, '_state_lookup'):
            return
        index = event.ind[0] if hasattr(event.ind, '__len__') else event.ind
        try:
            state = event.artist._state_lookup[index]
        except (IndexError, KeyError, TypeError):
            return
        if state is None:
            return
        self.current_state = state
        self._show_hover_state(state)
        self._update_display()

    def _hide_edges_for_drag(self):
        """拖拽开始: 隐藏所有边 (Line2D) 以加速重绘"""
        if self._cached_edge_visibilities:
            # 已经隐藏过一轮了。再进一次会把"已隐藏(False)"当成原状态缓存下来,
            # 松手后边就再也回不来 —— 中键+左键同按、或拖出画布松手都会触发。
            return
        cache = []
        for line in list(self.ax_graph.get_lines()):
            # 只处理坐标轴里的连线 (非网格线: 网格线是ax的grid, 不在get_lines里)
            cache.append((line, line.get_visible()))
            line.set_visible(False)
        self._cached_edge_visibilities = cache

    def _restore_edges_after_drag(self):
        """拖拽结束: 恢复边的 visibility, 并完整重绘一次"""
        for line, vis in self._cached_edge_visibilities:
            try:
                line.set_visible(vis)
            except Exception:
                pass
        self._cached_edge_visibilities = []
        self.canvas.draw_idle()

    def _on_mouse_press(self, event):
        if event.inaxes != self.ax_graph:
            return
        if event.button == 2:
            self._middle_button_pressed = True
            self._last_mouse_pos = (event.x, event.y)
            self._hide_edges_for_drag()
            return
        self._dragging = True
        self._last_mouse_pos = (event.x, event.y)
        # 按下时立刻隐藏连线 + 取消悬停节流定时器避免干扰
        self._hide_edges_for_drag()
        if self._hover_after_id is not None:
            self.root.after_cancel(self._hover_after_id)
            self._hover_after_id = None

    def _on_mouse_move(self, event):
        if event.inaxes != self.ax_graph:
            return
        if self._middle_button_pressed or self._dragging:
            if self._last_mouse_pos is None:
                self._last_mouse_pos = (event.x, event.y)
                return
            dx = event.x - self._last_mouse_pos[0]
            dy = event.y - self._last_mouse_pos[1]
            self.view_azim = (self.view_azim + dx * 0.35) % 360
            self.view_elev = max(-89, min(89, self.view_elev - dy * 0.35))
            self._last_mouse_pos = (event.x, event.y)
            self.ax_graph.view_init(elev=self.view_elev, azim=self.view_azim)
            # ---- 视图重绘节流: 30ms 一次, 事件期间只保留最新角度 ----
            self._drag_view_pending = (self.view_elev, self.view_azim)
            if self._drag_redraw_after_id is None:
                self._drag_redraw_after_id = self.root.after(30, self._do_drag_redraw)
            return
        # ---- 非拖拽模式: 悬停检测, 节点数>阈值时阈值变大 ----
        n_nodes = 0
        try:
            if self.visualizer.node_positions:
                n_nodes = len(self.visualizer.node_positions)
        except Exception:
            pass
        self._hover_mouse_pos = (event.x, event.y)
        # 节点数越多, 节流窗口越大: 500以下→60ms, 500-2000→120ms, 2000+→200ms
        if n_nodes < 500:
            delay = 60
        elif n_nodes < 2000:
            delay = 120
        else:
            delay = 200
        if self._hover_after_id is None:
            self._hover_after_id = self.root.after(delay, self._do_hover_highlight)

    def _do_drag_redraw(self):
        """拖拽时的节流重绘"""
        self._drag_redraw_after_id = None
        if self._drag_view_pending is not None:
            elev, azim = self._drag_view_pending
            self.ax_graph.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _do_hover_highlight(self):
        """节流后的悬停高亮执行: 向量化投影 + 提前退出"""
        self._hover_after_id = None
        if self._dragging or self._middle_button_pressed:
            return
        node_positions = getattr(self.visualizer, 'node_positions', None)
        if not node_positions:
            return
        # 节点数过多, 直接跳过悬停检测 (保证拖动跟手)
        if len(node_positions) > 3000:
            return

        mx, my = self._hover_mouse_pos
        # ---- 向量化计算所有位置: 一次性变换 ----
        # 注意: node_positions 可能被其他线程 (布局切换/AI分析) 并发修改,
        # 用 list(items()) 一次性取 (key, value) 成对快照, 保证 nodes 和 pos_arr 一一对应
        # 并发修改时 dict.items() 可能抛 RuntimeError/KeyError, 这里捕获后跳过本次检测
        try:
            items = list(node_positions.items())
        except (RuntimeError, KeyError):
            return
        if not items:
            return
        nodes = [k for k, _ in items]
        pos_arr = np.array([v for _, v in items], dtype=np.float64)
        # 只投影 x,y (忽略 z 轴像素差, 对距离搜索影响很小)
        try:
            t = self.ax_graph.transData.transform(pos_arr[:, :2])
            px = t[:, 0]
            py = t[:, 1]
        except Exception:
            return
        dists_sq = (mx - px) ** 2 + (my - py) ** 2
        i_min = int(np.argmin(dists_sq))
        # 距离阈值: 20px 半径 (阈值越大, 越容易命中)
        if dists_sq[i_min] < 400:
            # 二次验证: 确认节点仍在当前 node_positions 中 (防止并发删除导致引用过期)
            try:
                if nodes[i_min] in node_positions:
                    self._highlight_hovered_node(nodes[i_min])
                else:
                    self._restore_point_highlights()
            except (KeyError, RuntimeError):
                self._restore_point_highlights()
        else:
            self._restore_point_highlights()

    def _on_mouse_scroll(self, event):
        if event.inaxes != self.ax_graph:
            return
        if event.button == 'up':
            self.view_dist = max(1.0, self.view_dist - 0.8)
        elif event.button == 'down':
            self.view_dist = min(40.0, self.view_dist + 0.8)
        else:
            self.view_dist = max(1.0, min(40.0, self.view_dist - event.step * 0.5))
        self.ax_graph.set_box_aspect((1, 1, 0.8))
        self.canvas.draw_idle()

    def _on_mouse_release(self, event):
        if event.button == 2:
            self._middle_button_pressed = False
        self._dragging = False
        self._last_mouse_pos = None
        # 取消节流定时器, 立刻应用最后一次角度 + 恢复边重绘
        if self._drag_redraw_after_id is not None:
            try:
                self.root.after_cancel(self._drag_redraw_after_id)
            except Exception:
                pass
            self._drag_redraw_after_id = None
        self._drag_view_pending = None
        self._restore_edges_after_drag()

    def _restore_point_highlights(self):
        changed = False
        for coll in list(self.ax_graph.collections):
            if hasattr(coll, '_base_colors'):
                try:
                    original_colors = getattr(coll, '_base_colors', None)
                    if original_colors is not None:
                        coll.set_facecolors(np.asarray(original_colors, dtype=float))
                        changed = True
                except Exception:
                    pass
        if changed:
            self.canvas.draw_idle()

    def _highlight_hovered_node(self, node):
        if self._dragging or self._middle_button_pressed:
            return
        node_positions = getattr(self.visualizer, 'node_positions', None)
        if not node_positions:
            return
        changed = False
        for coll in list(self.ax_graph.collections):
            if not hasattr(coll, '_state_lookup') or not hasattr(coll, '_base_sizes'):
                continue
            base_colors = getattr(coll, '_base_colors', None)
            if base_colors is None:
                continue
            node_index = getattr(coll, '_node_index_map', {}).get(node)
            if node_index is None:
                continue
            try:
                current_colors = np.asarray(coll.get_facecolors(), dtype=float)
                if current_colors.ndim == 1:
                    current_colors = current_colors.reshape(1, -1)
            except Exception:
                continue
            # 直接替换 base_colors 副本中对应 index 那一行, 然后 set_facecolors
            new_colors = np.asarray(base_colors, dtype=float).copy()
            if node_index < len(new_colors):
                new_colors[node_index] = [1.0, 0.0, 0.0, 0.95]
                coll.set_facecolors(new_colors)
                changed = True
        if changed:
            self.canvas.draw_idle()

    # ==================== 状态预览 ====================

    def _show_hover_state(self, state: State):
        if state is None:
            self._clear_hover()
            return
        self._hovered_state = state
        self.ax_hover.clear()
        self.visualizer.draw_state_grid(state, ax=self.ax_hover)
        self.ax_hover.set_title(f"状态 (方块: {len(state.blocks)})",
                                fontproperties=CN_FONT, fontsize=10)
        self.canvas_hover.draw_idle()

        info = f"方块数: {len(state.blocks)}\n"
        for block in state.blocks:
            info += f"  方块 {block.id}: {len(block.coords)}格\n"
        self.state_info_label.config(text=info.strip())

    def _clear_hover(self):
        self._hovered_state = None
        for coll in list(self.ax_graph.collections):
            if hasattr(coll, '_base_colors'):
                try:
                    original_colors = getattr(coll, '_base_colors', None)
                    if original_colors is not None:
                        coll.set_facecolors(np.asarray(original_colors, dtype=float))
                except Exception:
                    pass
        self.ax_hover.clear()
        self.ax_hover.text(0.5, 0.5, "点击质点可查看状态",
                           ha='center', va='center', fontsize=10, alpha=0.5,
                           fontproperties=CN_FONT)
        self.ax_hover.axis('off')
        self.canvas_hover.draw_idle()
        self.state_info_label.config(text="")
        self.canvas.draw_idle()

    def _show_info_in_hover(self, title: str, text: str):
        self.ax_hover.clear()
        self.ax_hover.axis('off')
        self.ax_hover.text(0.05, 0.95, f"【{title}】", transform=self.ax_hover.transAxes,
                           fontproperties=CN_FONT, fontsize=11, fontweight='bold', va='top', color=Theme.ACCENT_TEXT)
        self.ax_hover.text(0.05, 0.88, text, transform=self.ax_hover.transAxes,
                           fontproperties=CN_FONT, fontsize=9, va='top', ha='left', linespacing=1.8)
        self.canvas_hover.draw_idle()
        self.state_info_label.config(text="")

    # ==================== 加载布局 ====================

    def _load_clipboard(self):
        try:
            text = self.root.clipboard_get()
            state = LayoutImporter.from_json_string(text)
            self.current_state = state
            self._on_loaded("已从剪贴板加载")
        except Exception as e:
            messagebox.showerror("错误", f"加载失败: {str(e)}")

    def _open_layout_editor(self):
        """切到嵌入式绘制面板"""
        if hasattr(self, '_left_nb'):
            self._left_nb.select(1)

    def _on_layout_saved(self, file_path: str):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.py':
                state = LayoutImporter.from_python_file(file_path)
            elif ext == '.json':
                state = LayoutImporter.from_json_file(file_path)
            else:
                return
            self.current_state = state
            self._on_loaded(f"已加载保存的布局: {os.path.basename(file_path)}")
        except Exception as e:
            messagebox.showerror("错误", f"加载保存的布局失败: {str(e)}")

    def _on_loaded(self, message):
        self._update_status(message)
        # 换布局必须把上一张图连同它的分析结果一起作废。只清画布是不够的:
        # 数据层留着旧图, 而「全拓扑分析」只判断"有没有图", 于是会拿**旧布局**的节点算出
        # 结果、再按**新布局**的哈希存盘 —— 新布局的 result_<hash>.json / gnn_<hash>.pt
        # 从此永久被旧图的结果污染 (切 3D 还会因节点不属于新状态而抛 NetworkXError)。
        self.generator.graph = None
        self.generator._analysis_cache = None
        self.topo_result = None
        self.tda_result = None
        self.community_result = None
        self._ai_enhancement = {}
        self._last_display_key = None
        self.ax_graph.clear()
        if self.current_state:
            self.ax_graph.text2D(0.5, 0.5, "已加载初始布局，开始生成后将显示 3D 状态空间",
                                 transform=self.ax_graph.transAxes,
                                 ha='center', va='center', fontsize=11, alpha=0.7,
                                 fontproperties=CN_FONT)
            self.ax_graph.set_axis_off()
        self.canvas.draw()
        # 同步布局信息到左侧控制面板
        if self.current_state:
            info = f"棋盘: {self.current_state.grid_shape[0]}x{self.current_state.grid_shape[1]}, 方块: {len(self.current_state.blocks)}"
            self.info_label.config(text=info)
            if hasattr(self, 'layout_info_label'):
                self.layout_info_label.config(text=info)
        # 生成组可能未展开, 按钮已被销毁; 用 hasattr 容错
        if hasattr(self, 'generate_btn'):
            try:
                self.generate_btn.config(state=tk.NORMAL)
            except tk.TclError:
                pass

    # ==================== 生成状态空间 ====================

    def _on_symmetry_changed(self):
        """左侧面板的对称性开关回调"""
        # 同步工具栏开关状态
        if self.toolbar_symmetry_var.get() != self.symmetry_var.get():
            self.toolbar_symmetry_var.set(self.symmetry_var.get())
        # 直接应用变化 (不走 toolbar 回调避免循环)
        State._symmetric = self.symmetry_var.get()
        if State._symmetric:
            self._update_status("对称性: 开 (同形状方块按编号区分)")
        else:
            self._update_status("对称性: 关 (同形状方块物理等价合并)")
        if self.generator.graph is not None:
            self._update_status("对称性已切换, 请重新生成状态空间")

    def _on_toolbar_symmetry_changed(self):
        """工具栏的对称性快捷开关回调"""
        # 同步左侧面板开关 + 更新显示文字
        checked = self.toolbar_symmetry_var.get()
        if hasattr(self, 'symmetry_var') and self.symmetry_var.get() != checked:
            self.symmetry_var.set(checked)
        # 切换显示文本
        try:
            self.toolbar_symmetry_chk.config(
                text="对称性:开 (编号区分)" if checked else "对称性:关 (物理合并)"
            )
        except Exception:
            pass
        # 应用到 State 类
        State._symmetric = checked
        if checked:
            self._update_status("对称性: 开 (同形状方块按编号区分)")
        else:
            self._update_status("对称性: 关 (同形状方块物理等价合并)")
        if self.generator.graph is not None:
            self._update_status("对称性已切换, 请重新生成状态空间")

    def _start_generation(self):
        if self.current_state is None:
            messagebox.showwarning("警告", "请先加载布局")
            return
        if self.generator.is_running:
            return

        # 同步对称性模式到 State 类
        State._symmetric = self.symmetry_var.get()

        self.generator.graph = None
        self.topo_result = None
        self.tda_result = None
        self.community_result = None

        # ============ AI 记忆: 尝试加载布局级图缓存 ============
        cached_graph = self.generator.try_load_cached_graph(self.current_state)
        if cached_graph is not None:
            n_nodes = cached_graph.number_of_nodes()
            self.root.after(0, lambda: self._update_status(
                f"✓ AI记忆命中: 加载已缓存的图 ({n_nodes:,} 节点), 跳过BFS"))
            # 直接进入生成完成流程
            self._generation_finished_from_cache()
            return

        # 自动展开生成组, 保证停止按钮可见
        if self._active_group != 'generate':
            self._toggle_group('generate')
        if hasattr(self, 'generate_btn'):
            self.generate_btn.config(state=tk.DISABLED)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.config(state=tk.NORMAL)
        self.progress_bar.start()
        self._update_status("生成中...")
        self.progress_label.config(text="进度: 初始化...")

        self.generation_thread = threading.Thread(target=self._generation_task, daemon=True)
        self.generation_thread.start()

    def _generation_finished_from_cache(self):
        """从缓存加载图后直接进入后续流程 (跳过 BFS)"""
        # 模拟 UI 状态恢复
        if hasattr(self, 'generate_btn'):
            self.generate_btn.config(state=tk.NORMAL)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.config(state=tk.DISABLED)
        self.progress_bar.stop()

        if self.generator.graph and len(self.generator.graph.nodes) > 0:
            info = f"状态: {len(self.generator.graph.nodes):,} | 转换: {len(self.generator.graph.edges):,} (缓存)"
            self.info_label.config(text=info)
            self.sb_graph.config(text=f"节点: {len(self.generator.graph.nodes):,} | 边: {len(self.generator.graph.edges):,}")
            # 后台线程做基本分析 + 可视化准备
            t = threading.Thread(target=self._post_generation_task, daemon=True)
            t.start()
        else:
            messagebox.showwarning("警告", "缓存图为空")

    def _generation_task(self):
        def progress_callback(states, transitions, queue_size, finished=False):
            self.root.after(0, lambda: self._update_progress(states, transitions, queue_size, finished))
        try:
            self.generator.generate(self.current_state, progress_callback=progress_callback)
        except Exception as e:
            err_msg = str(e)
            self.root.after(0, lambda m=err_msg: messagebox.showerror("错误", f"生成失败: {m}"))
        self.root.after(0, self._generation_finished)

    def _update_progress(self, states, transitions, queue_size, finished=False):
        self.progress_label.config(text=f"状态: {states:,} | 转换: {transitions:,} | 队列: {queue_size}")
        self.sb_graph.config(text=f"节点: {states:,} | 边: {transitions:,}")
        if finished:
            self.progress_bar.stop()

    def _generation_finished(self):
        if hasattr(self, 'generate_btn'):
            self.generate_btn.config(state=tk.NORMAL)
        if hasattr(self, 'stop_btn'):
            self.stop_btn.config(state=tk.DISABLED)
        self.progress_bar.stop()

        if self.generator.should_stop:
            self._update_status("已停止")
        else:
            self._update_status("生成完成, 正在准备可视化...")

        if self.generator.graph and len(self.generator.graph.nodes) > 0:
            info = f"状态: {len(self.generator.graph.nodes):,} | 转换: {len(self.generator.graph.edges):,}"
            self.info_label.config(text=info)
            self.sb_graph.config(text=f"节点: {len(self.generator.graph.nodes):,} | 边: {len(self.generator.graph.edges):,}")
            # ---- 关键: 可视化绘制 + 分析放到后台线程, 避免阻塞UI ----
            t = threading.Thread(target=self._post_generation_task, daemon=True)
            t.start()
        else:
            messagebox.showwarning("警告", "未生成任何状态")

    def _post_generation_task(self):
        """生成后耗时操作: 基本分析 + 3D 视图绘制准备 (后台线程执行)

        注意:
        - 分析 (纯数据计算) 可以直接在后台跑
        - 3D 布局计算 (纯 numpy/math) 也可以在后台跑
        - 真正的 matplotlib canvas.draw() 和 widget 更新必须回到主线程 via after(0, ...)
        """
        # 1) 先跑分析 (纯数据, 无UI) — 很快
        try:
            analysis = self.generator.analyze()
        except Exception:
            analysis = None

        # 2) 如果当前是 3D 视图, 预计算布局 + 准备绘制数据 (纯CPU, 不操作UI)
        need_3d = (self._current_view == '3d')
        draw_args = None
        if need_3d and self.current_state is not None:
            try:
                max_steps = self.max_steps_var.get()
                # 准备绘图参数 (在后台构建子图、布局这些纯计算部分)
                draw_args = self._prepare_3d_draw_args(max_steps)
            except Exception:
                draw_args = None

        # 3) 回到主线程执行实际的 UI 绘制 + 文本更新
        def apply_to_ui():
            try:
                if analysis is not None:
                    self._apply_analysis_result(analysis)
                if need_3d:
                    self._update_display_3d_from_prepared(draw_args)
                else:
                    # 非3D视图也尝试同步分析文本
                    self._update_analysis()
                if not self.generator.should_stop:
                    self._update_status("生成完成")
                # 异步自动保存缓存 (IO密集, 再开一个线程不阻塞UI)
                t2 = threading.Thread(target=self._auto_save_to_cache, daemon=True)
                t2.start()
            except Exception as e:
                print(f"[apply_to_ui] error: {e}")

        self.root.after(0, apply_to_ui)

    def _prepare_3d_draw_args(self, max_steps):
        """在后台线程预计算 3D 绘制需要的数据 (BFS 子图 + 去中心化布局)

        返回一个 dict, 包含所有绘图所需数据, 主线程只需直接塞进 matplotlib
        """
        from collections import deque
        graph = self.generator.graph
        center_state = self.current_state
        if center_state is None:
            center_state = next(iter(graph.nodes()))

        # BFS 子图 (与 visualizer.draw_state_space 中逻辑一致, 只是不画图)
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

        # 布局 (纯数学)
        pos_3d = self.visualizer._decentralized_layout(graph, subgraph_nodes, depths,
                                                       center_state, max_steps)
        return {
            'graph': graph,
            'center_state': center_state,
            'subgraph_nodes': subgraph_nodes,
            'depths': depths,
            'pos_3d': pos_3d,
            'max_steps': max_steps,
            'title': f"3D 状态空间图（以当前状态为中心，最多 {max_steps} 步）",
        }

    def _update_display_3d_from_prepared(self, prep):
        """用预计算数据直接绘制 (主线程)"""
        if prep is None:
            # 回退到原方法
            self._update_display()
            return
        view = self._current_view
        display_key = (
            '3d',
            id(self.generator.graph) if self.generator.graph is not None else None,
            id(self.current_state) if self.current_state is not None else None,
            prep['max_steps'],
        )
        if display_key == self._last_display_key:
            self.canvas.draw_idle()
            self._show_hover_state(self.current_state)
            return
        self._last_display_key = display_key

        self._restore_3d_axis()
        ax = self.ax_graph

        graph = prep['graph']
        center_state = prep['center_state']
        subgraph_nodes = prep['subgraph_nodes']
        depths = prep['depths']
        pos_3d = prep['pos_3d']
        subgraph = graph.subgraph(subgraph_nodes)

        x = [pos_3d[n][0] for n in subgraph_nodes]
        y = [pos_3d[n][1] for n in subgraph_nodes]
        z = [pos_3d[n][2] for n in subgraph_nodes]

        depths_list = [depths[n] for n in subgraph_nodes]
        max_d = max(depths_list) if depths_list else 1
        # 用主题的紫蓝单色阶替代 viridis (见 主题.Theme.深度色阶 的说明: 明度仍严格单调)
        _深度图 = Theme.深度色图()
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

        scatter = ax.scatter(x, y, z, c=colors_array, s=sizes,
                             alpha=0.95, picker=12, linewidths=0.4, edgecolors='white')
        scatter._state_lookup = state_lookup
        scatter._base_sizes = __import__('numpy').array(sizes)
        scatter._base_colors = __import__('numpy').array(colors_array)
        scatter._node_index_map = {node: idx for idx, node in enumerate(subgraph_nodes)}
        self.visualizer.node_positions = {node: pos_3d[node] for node in subgraph_nodes}
        self.visualizer.node_to_state = {node: node for node in subgraph_nodes}
        self.visualizer.graph = graph

        # 画边
        for u, v in subgraph.edges():
            if u in pos_3d and v in pos_3d:
                x0, y0, z0 = pos_3d[u]
                x1, y1, z1 = pos_3d[v]
                d_avg = (depths.get(u, 0) + depths.get(v, 0)) / 2
                alpha = max(0.15, 0.6 - d_avg * 0.12)
                ax.plot([x0, x1], [y0, y1], [z0, z1],
                        color='#4a5568', alpha=alpha, linewidth=0.8, zorder=0)

        self.visualizer._auto_focus(ax, pos_3d)
        ax.set_title(prep['title'], fontproperties=CN_FONT, fontsize=11, pad=8)
        ax.set_axis_off()
        ax.set_box_aspect((1, 1, 0.8))
        ax.view_init(elev=self.view_elev, azim=self.view_azim)
        ax.set_proj_type('persp')

        self.canvas.draw_idle()
        self._show_hover_state(self.current_state)
        self._update_analysis()

    def _apply_analysis_result(self, analysis: dict):
        """直接把 analysis dict 渲染到 UI, 不再重新计算"""
        if not analysis:
            return
        text = "状态空间分析报告\n"
        text += "=" * 50 + "\n\n"
        text += f"状态数: {analysis['num_states']:,}\n"
        text += f"转换数: {analysis['num_transitions']:,}\n"
        text += f"生成时间: {analysis['generation_time']:.2f} 秒\n"
        text += f"图密度: {analysis['density']:.6f}\n"
        text += f"连通性: {'是' if analysis['connected'] else '否'}\n"
        if analysis.get('diameter') is not None:
            text += f"直径: {analysis.get('diameter')}\n"
        text += f"\n【度统计】\n"
        text += f"  平均度: {analysis['avg_degree']:.2f}\n"
        text += f"  最大度: {analysis['max_degree']}\n"
        text += f"  最小度: {analysis['min_degree']}\n"
        if not analysis['connected'] and 'num_components' in analysis:
            text += f"\n【连通分量】\n"
            text += f"  分量数: {analysis['num_components']}\n"
            text += f"  分量大小: {analysis['component_sizes']}\n"
        if analysis.get('stopped_early'):
            text += "\n生成被中断\n"
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', text)

    def _stop_generation(self):
        if self.generator.is_running:
            self.generator.stop()
            self._update_status("正在停止...")

    # ==================== 视图更新 ====================

    def _update_display(self):
        view = self._current_view
        changed = False

        if view == '3d':
            display_key = (
                '3d',
                id(self.generator.graph) if self.generator.graph is not None else None,
                id(self.current_state) if self.current_state is not None else None,
                self.max_steps_var.get(),
            )
            if display_key != self._last_display_key:
                self._last_display_key = display_key
                changed = True
                self._restore_3d_axis()
                if self.generator.graph and self.current_state:
                    self.visualizer.draw_state_space(
                        self.generator.graph, current_state=self.current_state,
                        ax=self.ax_graph,
                        title=f"3D 状态空间图（以当前状态为中心，最多 {self.max_steps_var.get()} 步）",
                        max_steps=self.max_steps_var.get(), focus_state=self.current_state)
                    self.ax_graph.view_init(elev=self.view_elev, azim=self.view_azim)
                    self.ax_graph.set_box_aspect((1, 1, 0.8))
                    self.ax_graph.set_axis_off()
                else:
                    self.ax_graph.text2D(0.5, 0.5, "无状态 / 请先加载布局并生成",
                                         transform=self.ax_graph.transAxes,
                                         ha='center', va='center', fontsize=13, alpha=0.7,
                                         fontproperties=CN_FONT)
                    self.ax_graph.set_axis_off()

        elif view == 'spectrum':
            display_key = ('spectrum', id(self.topo_result))
            if display_key != self._last_display_key:
                self._last_display_key = display_key
                changed = True
                self._draw_spectrum()

        elif view == 'barcode':
            display_key = ('barcode', id(self.tda_result))
            if display_key != self._last_display_key:
                self._last_display_key = display_key
                changed = True
                self._draw_barcode()

        elif view == 'ai':
            ai_fig = (getattr(self, '_ai_enhancement', {}) or {}).get('figure')
            display_key = ('ai', id(ai_fig))
            if display_key != self._last_display_key:
                self._last_display_key = display_key
                changed = True
                if ai_fig is not None:
                    self._show_ai_figure(ai_fig)
                else:
                    self.fig.clear()
                    ax = self.fig.add_subplot(111)
                    ax.text(0.5, 0.5, "请先运行「全拓扑分析」\n生成AI增强图",
                            transform=ax.transAxes, ha='center', va='center',
                            fontsize=13, alpha=0.7, fontproperties=CN_FONT)
                    ax.set_axis_off()

        if changed:
            self.canvas.draw_idle()
            if view == '3d':
                self._show_hover_state(self.current_state)
            elif view == 'spectrum':
                self._show_info_in_hover("谱图理论说明",
                    "左: 拉普拉斯谱 L=D-A\n"
                    "  · λ0=0 重数=β0\n"
                    "  · λ2=Fiedler值=代数连通度\n"
                    "  · 越大 → 连通性/鲁棒性越好\n\n"
                    "右: 邻接谱 A\n"
                    "  · 检验 Wigner 半圆律\n"
                    "  · (随机图呈半圆分布)\n"
                    "  · 谱半径 ρ=最大绝对值特征值")
            elif view == 'barcode':
                self._show_info_in_hover("持久条形码说明",
                    "横轴: 过滤参数 ε = 图距离\n"
                    "每条横线 = 1 个拓扑特征\n"
                    "  · 起点 = 特征诞生(出现)\n"
                    "  · 终点 = 特征消亡(消失)\n"
                    "H0 蓝色: 连通分支/聚类\n"
                    "H1 红色: 独立环/洞\n"
                    "长横线 = 稳定/本质特征\n"
                    "短横线 = 噪声/短暂特征")
            elif view == 'ai':
                self._show_info_in_hover("AI增强图说明",
                    "PCA/t-SNE: 拓扑特征聚类降维\n"
                    "GNN嵌入: GraphSAGE节点嵌入\n"
                    "持久图像: 持久条形码转图像\n"
                    "\n"
                    "运行「全拓扑分析」自动生成")
            self._update_analysis()

    def _update_view_buttons(self):
        base_labels = {'3d': '3D 状态图', 'spectrum': '谱分布', 'barcode': '持久条形码', 'ai': 'AI增强图'}
        titles = {'3d': '3D 状态空间', 'spectrum': '拉普拉斯/邻接谱分布', 'barcode': '持久同调条形码', 'ai': 'AI增强分析图'}
        btn_map = {}
        for key in ('3d', 'spectrum', 'barcode', 'ai'):
            attr = f'btn_view_{key}'
            if hasattr(self, attr):
                btn_map[key] = getattr(self, attr)
        for key, btn in btn_map.items():
            prefix = '● ' if key == self._current_view else ''
            btn.configure(text=prefix + base_labels[key])
        if hasattr(self, 'view_title_label'):
            self.view_title_label.config(text=titles.get(self._current_view, ''))
        if hasattr(self, 'sb_view'):
            self.sb_view.config(text=f"视图: {titles.get(self._current_view, '')}")

    def _switch_view(self, view: str):
        if self._current_view == view:
            return
        self._current_view = view
        self._update_view_buttons()
        self._update_display()

    def _restore_3d_axis(self):
        fig = self.fig
        for a in fig.axes[:]:
            fig.delaxes(a)
        self.ax_graph = fig.add_subplot(111, projection='3d')
        self.ax_graph.set_title("3D 状态空间视图", fontproperties=CN_FONT, fontsize=11, pad=8)
        self.ax_graph.set_facecolor(Theme.PANEL)
        self.ax_graph.set_axis_off()

    # ==================== 谱图 / 条形码绘制 ====================

    def _draw_spectrum(self):
        fig = self.fig
        for a in fig.axes[:]:
            fig.delaxes(a)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        ax1.set_facecolor(Theme.PANEL)
        ax2.set_facecolor(Theme.PANEL)

        if not self.topo_result or 'spectrum' not in self.topo_result:
            ax1.text(0.5, 0.5, "请先运行「全拓扑分析」\n生成谱数据",
                     transform=ax1.transAxes, ha='center', va='center',
                     fontsize=12, alpha=0.7, fontproperties=CN_FONT)
            ax1.axis('off')
            ax2.axis('off')
            return

        spec = self.topo_result['spectrum']
        lap = spec.get('laplacian_eigenvalues', np.array([]))
        adj = spec.get('adjacency_eigenvalues', np.array([]))

        if len(lap) > 0:
            bins = min(40, max(5, len(lap) // 4))
            ax1.hist(lap, bins=bins, color=Theme.ACCENT, edgecolor='white', alpha=0.85)
            if len(lap) >= 2:
                lam2 = lap[1]
                ax1.axvline(lam2, color=Theme.DANGER, linestyle='--', linewidth=2,
                            label=f'λ2 (代数连通度) = {lam2:.3f}')
                ax1.legend(fontsize=9, loc='best')
        apply_cn_font(ax1, title="拉普拉斯谱 L = D - A\n(特征值分布)",
                      xlabel="λ (Laplacian 特征值)", ylabel="频数",
                      title_fontsize=10, label_fontsize=9, tick_fontsize=9)
        ax1.grid(alpha=0.3)

        if len(adj) > 0:
            bins = min(40, max(5, len(adj) // 4))
            ax2.hist(adj, bins=bins, color=Theme.SUCCESS, edgecolor='white', alpha=0.85)
            rho = spec.get('spectral_radius', 0)
            if rho > 0:
                ax2.axvline(rho, color=Theme.WARN, linestyle='--', linewidth=2, label=f'谱半径 = {rho:.3f}')
                ax2.axvline(-rho, color=Theme.WARN, linestyle='--', linewidth=2)
                ax2.legend(fontsize=9, loc='best')
        apply_cn_font(ax2, title="邻接谱 A (是否符合 Wigner 半圆律?)",
                      xlabel="λ (邻接矩阵特征值)", ylabel="频数",
                      title_fontsize=10, label_fontsize=9, tick_fontsize=9)
        ax2.grid(alpha=0.3)
        fig.tight_layout()

    def _draw_barcode(self):
        fig = self.fig
        for a in fig.axes[:]:
            fig.delaxes(a)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor(Theme.PANEL)

        if not self.tda_result or 'bars' not in self.tda_result:
            ax.text(0.5, 0.5, "请先运行「全拓扑分析」\n生成持久同调条形码",
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=12, alpha=0.7, fontproperties=CN_FONT)
            ax.axis('off')
            return

        if self.tda_analyzer is None:
            self.tda_analyzer = TDAnalyzer(self.generator.graph)
        self.tda_analyzer.plot_barcode(ax, self.tda_result, max_bars_per_dim=100)
        fig.tight_layout()

    # ==================== 拓扑分析 ====================

    def _run_full_topology(self):
        if self.generator.graph is None or len(self.generator.graph.nodes) == 0:
            messagebox.showwarning("警告", "请先生成状态空间")
            return

        n = len(self.generator.graph.nodes)
        self._update_status("拓扑分析中...")
        self.analysis_nb.select(1)
        self.topo_text.delete('1.0', tk.END)
        self.topo_text.insert('1.0', f"正在计算 (节点数 {n:,})...\n")
        self.tda_text.delete('1.0', tk.END)
        self.tda_text.insert('1.0', "等待拓扑分析完成后开始...\n")

        # 进度标签映射
        topo_labels = {
            'homology': '同调 (Betti数)',
            'cycles': '环结构',
            'spectrum': '谱图理论 (特征值)',
            'centrality': '拓扑中心性',
        }
        tda_labels = {
            'distance_matrix': '距离矩阵 (BFS)',
            'vietoris_rips': 'VR复形 (持久同调)',
            'stats': '统计量',
        }

        def update_topo_progress(stage):
            label = topo_labels.get(stage, stage)
            self.root.after(0, lambda: self.topo_text.insert('end', f"  ✓ {label}\n"))

        def update_tda_progress(stage):
            label = tda_labels.get(stage, stage)
            self.root.after(0, lambda: self.tda_text.delete('1.0', tk.END))
            self.root.after(0, lambda: self.tda_text.insert('1.0', f"TDA 计算: {label}...\n"))

        def worker():
            G = self.generator.graph
            n_nodes = G.number_of_nodes()
            is_large = n_nodes >= 800
            is_medium = n_nodes >= 300

            # ============ AI 记忆: 检查完整结果缓存 ============
            layout_hash = compute_layout_hash(self.current_state)
            self._current_layout_hash = layout_hash

            cached_full = load_full_result(layout_hash)
            if cached_full is not None:
                # 完整缓存命中! 跳过全部解析, 直接加载结果
                self.topo_result = cached_full.get('topology_result')
                self.tda_result = cached_full.get('tda_result')
                self.community_result = cached_full.get('community_result')
                self._ai_enhancement = cached_full.get('ai_enhancement', {}) or {}

                # 关键: 存盘时 figure 被剔除 (不可 JSON 序列化), 命中缓存必须**重建**,
                # 否则「AI增强图」永远是空画布 —— 明明 PCA/t-SNE/持久图像数据都在缓存里。
                # 在主线程重建 (不碰 GUI, 只是造 Figure 对象), 失败也不影响缓存命中的其它结果。
                self.root.after(0, lambda: self._重建AI图并记录('缓存命中'))

                self.root.after(0, lambda: self.topo_text.insert(
                    'end', f"\n✓✓✓ AI记忆命中完整结果缓存 (布局 {layout_hash}), 秒级完成!\n"))
                self.root.after(0, lambda: self.topo_text.insert(
                    'end', "  跳过: 拓扑分析 / GNN / TDA / 社区检测 / AI增强\n"))
                self.root.after(0, lambda: self._format_topo_text())
                self.root.after(0, lambda: self._format_tda_text())
                self.root.after(0, self._on_topology_finished)
                return

            # ============ AI 记忆: 检查各模块布局级记忆 ============
            mem_info = check_layout_memory(layout_hash, state=self.current_state)
            gnn_paths = layout_model_paths(layout_hash)
            gnn_path = gnn_paths['gnn']
            global_paths = global_model_paths()

            # ===== 1. GNN 先行训练 (与拓扑分析并行, 所有图都启动) =====
            gnn_embeddings = [None]  # 用 list 包装以便线程间共享
            gnn_nodes = [None]
            gnn_ready = threading.Event()

            def train_gnn():
                try:
                    gnn = SimpleGNN(input_dim=8, hidden_dim=64, output_dim=32)

                    # ============ AI 记忆: 三级加载策略 ============
                    # 1. 布局级模型 (最优, 完全匹配)
                    # 2. 全局预训练模型 (次优, 跨布局迁移)
                    # 3. 从头训练 (无记忆)
                    loaded_from = None
                    if os.path.exists(gnn_path) and gnn.load_model(gnn_path):
                        loaded_from = f"布局级 ({layout_hash})"
                    elif os.path.exists(global_paths['gnn']) and gnn.load_model(global_paths['gnn']):
                        loaded_from = "全局预训练"

                    if loaded_from:
                        # 加载成功: 少量 epochs 微调 (不是从零开始!)
                        finetune_epochs = 20 if is_large else 30
                        gnn.train(G, epochs=finetune_epochs, lr=0.005,
                                  progress_callback=lambda **kw: None)
                        # 保存微调结果 (累积记忆)
                        gnn.save_model(gnn_path)
                        self.root.after(0, lambda f=loaded_from, e=finetune_epochs: self.topo_text.insert(
                            'end', f"  ✓ GNN加载{f} + 微调{e} epochs (AI记忆)\n"))
                    else:
                        # 无任何记忆, 从零训练 + 保存全局预训练
                        epochs = 50 if is_large else 100
                        gnn.train(G, epochs=epochs, lr=0.01,
                                  progress_callback=lambda **kw: None)
                        # 同时保存布局级和全局 (供跨布局迁移)
                        gnn.save_model(gnn_path)
                        os.makedirs(os.path.dirname(global_paths['gnn']), exist_ok=True)
                        gnn.save_model(global_paths['gnn'])
                        self.root.after(0, lambda: self.topo_text.insert(
                            'end', "  ✓ GNN训练完成 (已保存布局级 + 全局记忆)\n"))

                    gnn_embeddings[0] = gnn.get_embeddings(G)
                    gnn_nodes[0] = list(G.nodes())
                except Exception as e:
                    err_msg = str(e)
                    self.root.after(0, lambda m=err_msg: self.topo_text.insert('end', f"  ✗ GNN训练失败: {m}\n"))
                finally:
                    gnn_ready.set()

            gnn_thread = threading.Thread(target=train_gnn, daemon=True)
            gnn_thread.start()
            # AI提速强化: 所有图都显示并行提示
            self.root.after(0, lambda: self.topo_text.insert('end', "— GNN训练与拓扑分析并行中 —\n"))

            # ===== 2. 拓扑分析 (与 GNN 并行) =====
            # AI提速强化: 大图跳过精确谱计算, 等GNN完成后用嵌入近似
            topo = TopologyAnalyzer(G)
            try:
                self.topo_result = topo.full_analysis(progress_callback=update_topo_progress)
            except Exception as e:
                self.topo_result = {'error': str(e)}
            self.topology_analyzer = topo

            # ===== 3. 等待 GNN 完成 =====
            gnn_ready.wait(timeout=120)
            embeddings = gnn_embeddings[0]
            emb_nodes = gnn_nodes[0]

            # ===== 3.5 AI提速强化: GNN嵌入快速近似图级特征 =====
            if embeddings is not None:
                try:
                    fast_feats = SimpleGNN.fast_graph_features(embeddings)
                    if fast_feats and 'error' not in fast_feats:
                        # 将近似特征补充到拓扑结果中
                        if isinstance(self.topo_result, dict):
                            self.topo_result.setdefault('ai_fast_features', fast_feats)
                        self.root.after(0, lambda: self.topo_text.insert(
                            'end', f"  ✓ GNN嵌入快速近似: 谱半径={fast_feats.get('spectral_radius', 0):.3f}, "
                                   f"代数连通度={fast_feats.get('algebraic_connectivity', 0):.3f}, "
                                   f"模块度={fast_feats.get('modularity_proxy', 0):.3f}\n"))
                    elif fast_feats:
                        # 失败原因由 fast_graph_features 带回来, 不静默跳过
                        _why = str(fast_feats.get('error', '未知原因'))
                        self.root.after(0, lambda m=_why: self.topo_text.insert(
                            'end', f"  ⚠ GNN快速近似未产出: {m}\n"))
                except Exception as e:
                    err_msg = str(e)
                    self.root.after(0, lambda m=err_msg: self.topo_text.insert(
                        'end', f"  ⚠ GNN快速近似失败: {m}\n"))

            # ===== 4. 社区检测: GNN可用时用KMeans加速 (扩大到中等图) =====
            comm = CommunityAnalyzer(G)
            if embeddings is not None and (is_large or is_medium):
                try:
                    self.community_result = comm.detect_communities_embedding(
                        embeddings, emb_nodes, n_clusters=min(5, n_nodes))
                    self.community_result['ai_accelerated'] = True
                    self.root.after(0, lambda: self.topo_text.insert('end', "  ✓ 社区检测: KMeans-GNN加速\n"))
                except Exception as e:
                    err_msg = str(e)
                    try:
                        self.community_result = comm.detect_communities(resolution=1.0)
                    except Exception as e2:
                        err2_msg = str(e2)
                        self.community_result = {'error': err2_msg}
                        self.root.after(0, lambda m=err2_msg: self.topo_text.insert('end', f"  ✗ greedy回退也失败: {m}\n"))
                    else:
                        self.root.after(0, lambda m=err_msg: self.topo_text.insert('end', f"  KMeans回退 greedy: {m}\n"))
            else:
                try:
                    self.community_result = comm.detect_communities(resolution=1.0)
                except Exception as e:
                    self.community_result = {'error': str(e)}
            self.community_analyzer = comm

            # ===== 5. TDA: GNN可用时用嵌入引导采样 (扩大到中等图, 大图进一步减半) =====
            tda = TDAnalyzer(G)
            # AI提速强化: 大图50采样, 中等图100采样, 小图200采样
            if embeddings is not None and is_large:
                tda_max_samples = 50
            elif embeddings is not None and is_medium:
                tda_max_samples = 100
            else:
                tda_max_samples = 200
            try:
                self.tda_result = tda.compute_persistence(
                    max_samples=tda_max_samples, max_dim=1,
                    progress_callback=update_tda_progress,
                    embeddings=embeddings if embeddings is not None else None,
                    nodes_list=emb_nodes if emb_nodes is not None else None)
                if self.tda_result.get('ai_sampling'):
                    self.root.after(0, lambda: self.topo_text.insert('end', "  ✓ TDA: AI引导采样加速\n"))
            except Exception as e:
                self.tda_result = {'error': str(e)}
            self.tda_analyzer = tda

            # ===== 6. AI 增强: ML特征 + TDA深度学习 (GNN已训练完毕) =====
            self.root.after(0, lambda: self.topo_text.insert('end', "\n— AI 增强分析 —\n"))
            self._ai_enhancement = {}
            self._run_ai_enhancement(G, embeddings, emb_nodes)

            # ============ AI 记忆: 保存完整结果到布局级缓存 ============
            try:
                full_result = {
                    'topology_result': self.topo_result,
                    'tda_result': self.tda_result,
                    'community_result': self.community_result,
                    'ai_enhancement': {
                        # 只保存可序列化的部分, 跳过 matplotlib figure 等不可序列化对象
                        k: v for k, v in self._ai_enhancement.items()
                        if k != 'figure' and k != 'gnn'  # gnn 的 embeddings 太大, 下次可以重算
                    },
                }
                save_full_result(layout_hash, full_result)
                self.root.after(0, lambda: self.topo_text.insert(
                    'end', f"\n✓ 完整结果已缓存 (布局 {layout_hash}), 下次运行秒级完成!\n"))
            except Exception:
                pass  # 缓存保存失败不影响主流程

            self.root.after(0, self._on_topology_finished)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _run_ai_enhancement(self, G, gnn_embeddings=None, gnn_nodes=None):
        """AI 增强分析: ML特征聚类 + (GNN已有嵌入) + TDA深度学习

        GNN 嵌入在 worker 中已训练完毕并传入, 此处不再重复训练。
        结果存入 self._ai_enhancement, 由 _format_topo_text / _format_tda_text 读取。
        """
        from collections import Counter
        ai = {}

        # ---- 1. 拓扑特征 + 传统 ML 聚类 ----
        try:
            ml = TopologyFeatureML()
            features = ml.extract_graph_features(G, self.topo_result, self.tda_result)
            n_nodes = features.shape[0]
            n_feat = features.shape[1]
            clusters = ml.cluster_nodes(G, n_clusters=min(5, n_nodes))
            sizes = Counter(clusters.values())
            pca_2d = ml.reduce_dimensions(features, method='pca', n_components=2)
            tsne_2d = (ml.reduce_dimensions(features, method='tsne', n_components=2)
                       if n_nodes <= 5000 else None)
            ai['ml'] = {
                'features_shape': (n_nodes, n_feat),
                'n_clusters': len(set(clusters.values())),
                'cluster_sizes': dict(sizes),
                'pca_2d': pca_2d,
                'tsne_2d': tsne_2d,
                'clusters': clusters,
            }
            self.root.after(0, lambda: self.topo_text.insert('end', "  ✓ 拓扑特征ML聚类\n"))
            # 保存 ML 模型 (记录当前图的特征, 供后续查阅)
            try:
                _md = get_model_dir()
                os.makedirs(_md, exist_ok=True)
                ml.save_model(os.path.join(_md, 'ml_global.pkl'))
            except Exception:
                pass
        except Exception as e:
            err_msg = str(e)
            ai['ml'] = {'error': err_msg}
            self.root.after(0, lambda m=err_msg: self.topo_text.insert('end', f"  ✗ ML聚类失败: {m}\n"))

        # ---- 2. GNN 嵌入 (复用已训练结果) ----
        if gnn_embeddings is not None:
            ai['gnn'] = {
                'embeddings_shape': gnn_embeddings.shape,
                'embeddings': gnn_embeddings,
                'nodes': gnn_nodes,
            }
            self.root.after(0, lambda: self.topo_text.insert('end', "  ✓ GNN节点嵌入 (已复用)\n"))
        else:
            try:
                gnn = SimpleGNN(input_dim=8, hidden_dim=64, output_dim=32)
                gnn.train(G, epochs=100, lr=0.01, progress_callback=lambda **kw: None)
                emb = gnn.get_embeddings(G)
                ai['gnn'] = {
                    'embeddings_shape': emb.shape,
                    'embeddings': emb,
                    'nodes': list(G.nodes()),
                }
                self.root.after(0, lambda: self.topo_text.insert('end', "  ✓ GNN节点嵌入\n"))
            except Exception as e:
                err_msg = str(e)
                ai['gnn'] = {'error': err_msg}
                self.root.after(0, lambda m=err_msg: self.topo_text.insert('end', f"  ✗ GNN失败: {m}\n"))

        # ---- 3. TDA 深度学习 (持久图像 + 自编码器) ----
        try:
            tda_dl = TDADeepLearning()
            bars = (self.tda_result or {}).get('bars', {})
            pimg_h0 = (tda_dl.barcode_to_persistence_image({0: bars.get(0, [])}, resolution=20)
                       if bars.get(0) else None)
            pimg_h1 = (tda_dl.barcode_to_persistence_image({1: bars.get(1, [])}, resolution=20)
                       if bars.get(1) else None)
            feat = tda_dl.build_feature_vector(self.tda_result)

            # 记忆功能: 尝试加载预训练自编码器 (维度匹配时复用)
            model_dir = get_model_dir()
            tda_path = os.path.join(model_dir, 'tda_ae_global.pt')
            loaded_ae = (os.path.exists(tda_path) and tda_dl.load_model(tda_path)
                         and getattr(tda_dl, '_autoencoder', None) is not None
                         and tda_dl._autoencoder.encoder[0].in_features == feat.shape[0])
            if not loaded_ae:
                tda_dl.train_autoencoder(feat.reshape(1, -1), epochs=50,
                                         progress_callback=lambda **kw: None)
                os.makedirs(model_dir, exist_ok=True)
                tda_dl.save_model(tda_path)
                self.root.after(0, lambda: self.topo_text.insert(
                    'end', "  ✓ TDA深度学习 (已保存)\n"))
            else:
                self.root.after(0, lambda: self.topo_text.insert(
                    'end', "  ✓ TDA深度学习 (加载记忆)\n"))

            ai['tda_dl'] = {
                'pimg_h0': pimg_h0,
                'pimg_h1': pimg_h1,
                'feat_shape': feat.shape,
            }
        except Exception as e:
            # 注意: Python 3 中 except 块结束后 e 会被自动清除,
            # 但 root.after 的 lambda 是延迟执行, 那时 e 已不存在 → NameError
            # 解决: 在块内立即转成字符串存到本地变量, lambda 引用该变量
            err_msg = str(e)
            ai['tda_dl'] = {'error': err_msg}
            self.root.after(0, lambda m=err_msg: self.topo_text.insert(
                'end', f"  ✗ TDA深度学习失败: {m}\n"))

        # ---- 4. AI 加速标记 ----
        comm_ai = (self.community_result or {}).get('ai_accelerated', False)
        tda_ai = (self.tda_result or {}).get('ai_sampling', False)
        topo_fast = isinstance(self.topo_result, dict) and 'ai_fast_features' in (self.topo_result or {})
        ai['acceleration'] = {
            'gnn_parallel': gnn_embeddings is not None,
            'gnn_fast_features': topo_fast,
            'community_kmeans': comm_ai,
            'tda_guided_sampling': tda_ai,
            'betweenness_skipped': G.number_of_nodes() >= 500,
            'medium_graph_accelerated': 300 <= G.number_of_nodes() < 800,
        }

        # ---- 生成 AI 综合可视化图 ----
        self._ai_enhancement = ai
        # try 保护: matplotlib 字体/字形或 sklearn PCA 在边界条件下可能抛异常,
        # 此处未保护会导致异常击穿 worker 线程, _on_topology_finished 无法执行 → UI 卡死
        try:
            ai['figure'] = self._build_ai_figure(ai, G)
        except Exception as e:
            err_msg = str(e)
            ai['figure'] = None
            self.root.after(0, lambda m=err_msg: self.topo_text.insert(
                'end', f"  ✗ AI可视化图构建失败: {m}\n"))
        # 注意: 这里**不要**把 figure 从 ai 里摘掉 —— self._ai_enhancement 就是 ai 本身,
        # 摘掉等于把刚建好的图从内存里删了, 于是首次分析后点「AI增强图」只会看到
        # "请先运行全拓扑分析"的空占位 (只有命中磁盘缓存那条路会重建, 所以看起来是好的)。
        # 落盘时本来就不用担心: 下面的 full_result 用字典推导显式排除了 figure。

    def _build_ai_figure(self, ai, G):
        """构建 AI 增强分析的综合可视化 Figure (线程安全, 无 GUI manager)

        注意: 入参 ai 可能来自**缓存反序列化**(json), 其中
          • 键可能整体缺失 (例如缓存里 gnn 存的是 None);
          • numpy 数组已变成 list。
        所以取值一律 `ai.get(k) or {}` —— 之前写 `ai.get('gnn', {})` 在值为 None 时
        会拿到 None, 后面 `in None` 直接 TypeError。
        调用方: _run_ai_enhancement (训练后首次) 与 _rebuild_ai_figure_from_cache (缓存命中)。
        """
        from 字体工具 import CN_FONT
        import numpy as _np
        ai = ai or {}
        ml = ai.get('ml') or {}
        gnn = ai.get('gnn') or {}
        tda_dl = ai.get('tda_dl') or {}

        # 计算需要的子图数: PCA + (t-SNE?) + GNN嵌入 + (H0持久图?) + (H1持久图?)
        plots = []
        if ml.get('pca_2d') is not None:
            plots.append('pca')
        if ml.get('tsne_2d') is not None:
            plots.append('tsne')
        if gnn.get('embeddings') is not None:
            plots.append('gnn')
        if tda_dl.get('pimg_h0') is not None:
            plots.append('ph0')
        if tda_dl.get('pimg_h1') is not None:
            plots.append('ph1')
        if not plots:
            return None

        fig = Figure(figsize=(5 * min(len(plots), 3), 4 * ((len(plots) + 2) // 3)))
        for i, kind in enumerate(plots):
            ax = fig.add_subplot((len(plots) + 2) // 3, min(len(plots), 3), i + 1)
            if kind == 'pca':
                pca_2d = _np.asarray(ml['pca_2d'])
                colors = self._聚类取色(ml.get('clusters'), G)
                ax.scatter(pca_2d[:, 0], pca_2d[:, 1],
                           c=colors if colors is not None else Theme.ACCENT,
                           cmap='tab10' if colors is not None else None, s=8, alpha=0.6)
                ax.set_title("PCA 降维 (拓扑特征聚类)", fontproperties=CN_FONT, fontsize=10)
                ax.set_xlabel("PC1", fontproperties=CN_FONT)
                ax.set_ylabel("PC2", fontproperties=CN_FONT)
            elif kind == 'tsne':
                tsne_2d = _np.asarray(ml['tsne_2d'])
                colors = self._聚类取色(ml.get('clusters'), G)
                ax.scatter(tsne_2d[:, 0], tsne_2d[:, 1],
                           c=colors if colors is not None else Theme.ACCENT,
                           cmap='tab10' if colors is not None else None, s=8, alpha=0.6)
                ax.set_title("t-SNE 降维", fontproperties=CN_FONT, fontsize=10)
            elif kind == 'gnn':
                emb = _np.asarray(gnn['embeddings'])
                from sklearn.decomposition import PCA as _PCA
                emb_2d = _PCA(n_components=2).fit_transform(emb)
                ax.scatter(emb_2d[:, 0], emb_2d[:, 1], s=8, alpha=0.6, c=Theme.ACCENT)
                ax.set_title("GNN 节点嵌入 (PCA)", fontproperties=CN_FONT, fontsize=10)
            elif kind in ('ph0', 'ph1'):
                dim = 0 if kind == 'ph0' else 1
                img = _np.asarray(tda_dl[f'pimg_h{dim}'])
                im = ax.imshow(img, cmap='hot', origin='lower', aspect='auto')
                ax.set_title(f"H{dim} 持久图像", fontproperties=CN_FONT, fontsize=10)
                ax.set_xlabel("出生时间", fontproperties=CN_FONT)
                ax.set_ylabel("持久值", fontproperties=CN_FONT)
                fig.colorbar(im, ax=ax)
        fig.tight_layout()
        return fig

    def _聚类取色(self, clusters, G):
        """按聚类给节点取色; 键对不上时回退 (返回 None 表示用单色)

        为什么需要兜底: 缓存里的 clusters 是 JSON 反序列化回来的, 键是
        `__state__:行,列||id:r,c|...` 这样的**字符串**, 而 G.nodes() 是真实的 State 对象,
        直接 `clusters.get(node)` 会全部查不到 → 所有点退化成同一类颜色,
        图看起来"有画但没信息"。这里发现命中率过低就明确回退, 不假装有聚类。
        """
        if not isinstance(clusters, dict) or not clusters:
            return None
        try:
            节点 = list(G.nodes())
        except Exception:
            return None
        if not 节点:
            return None
        命中 = sum(1 for n in 节点 if n in clusters)
        if 命中 < max(1, len(节点) // 10):      # 命中不足 10% 视为键不匹配
            # 退一步: 按"节点顺序"对齐 —— 缓存里的 clusters 往往是按节点顺序生成的,
            # 用其值序列按序上色仍能反映聚类结构 (数量一致时才用)
            值序列 = list(clusters.values())
            if len(值序列) == len(节点) and all(isinstance(v, (int, float)) for v in 值序列):
                return [int(v) for v in 值序列]
            return None
        return [clusters.get(n, 0) for n in 节点]

    def _重建AI图并记录(self, 来源: str):
        """从 _ai_enhancement 重建 AI 增强图 (缓存命中路径用)

        为什么要重建: 存盘时 figure 被明确剔除 (matplotlib Figure 不可 JSON 序列化),
        命中缓存后如果不管它, 「AI增强图」就永远是空画布 —— 明明分析结果都在缓存里。
        """
        ai = getattr(self, '_ai_enhancement', None) or {}
        if ai.get('figure') is not None:
            return True
        G = getattr(self.generator, 'graph', None)
        if G is None:
            return False
        try:
            fig = self._build_ai_figure(ai, G)
        except Exception as e:
            self.topo_text.insert('end', f"  ⚠ AI增强图重建失败 ({来源}): {e}\n")
            return False
        if fig is None:
            self.topo_text.insert('end', f"  · 缓存里没有可画的 AI 数据 ({来源}), AI增强图暂不可用\n")
            return False
        ai['figure'] = fig
        self._ai_enhancement = ai
        self.topo_text.insert('end', f"  ✓ AI增强图已从缓存重建 ({来源}), 可点左上「AI增强图」或按 F4 查看\n")
        return True

    def _on_topology_finished(self):
        self._update_status("拓扑分析完成")
        self._format_topo_text()
        self._format_tda_text()
        if self._current_view in ('spectrum', 'barcode', 'ai'):
            self._last_display_key = None  # 强制刷新
            self._update_display()
        self._auto_save_to_cache()

    def _auto_save_to_cache(self):
        """解析完成后自动保存到缓存

        注意: 本方法会在**工作线程**里被调用 (见 _post_generation_task),
        所以结尾刷新 Treeview 必须回到主线程 —— Tk 控件只能在主线程写。
        """
        try:
            if self.generator.graph is None:
                return
            layout_data = None
            if self.current_state is not None:
                blocks_dict = {b.id: list(b.coords) for b in self.current_state.blocks}
                layout_data = LayoutData(self.current_state.grid_shape[0],
                                         self.current_state.grid_shape[1], blocks_dict)
            graph_stats = {
                'n_nodes': self.generator.graph.number_of_nodes(),
                'n_edges': self.generator.graph.number_of_edges(),
                'generation_time': getattr(self.generator, 'generation_time', 0.0) or 0.0,
            }
            name = f"{self.current_state.grid_shape[0]}x{self.current_state.grid_shape[1]} {len(self.current_state.blocks)}块" if self.current_state else "未命名"
            self.cache.save_session(
                name=name,
                layout_data=layout_data,
                basic_analysis=self.generator.analyze(),
                topology_result=self.topo_result,
                tda_result=self.tda_result,
                community_result=self.community_result,
                graph_stats=graph_stats,
            )
            # Treeview 只能在主线程刷新 (本方法可能跑在工作线程里)
            self.root.after(0, self.cache_tab.refresh)
        except Exception:
            pass

    def _on_load_cached_session(self, data: dict):
        """从缓存加载会话到界面"""
        try:
            layout = data.get('layout')
            if layout:
                blocks = [Block(int(bid), [tuple(c) for c in coords])
                          for bid, coords in layout.get('blocks', {}).items()]
                self.current_state = State(blocks, (layout['rows'], layout['cols']))
                self._on_loaded(f"已从缓存加载: {data.get('name', '')}")

            self.topo_result = data.get('topology_result')
            self.tda_result = data.get('tda_result')
            self.community_result = data.get('community_result')

            self._format_topo_text()
            self._format_tda_text()

            # 如果有图统计但没有图对象, 至少更新基本分析文本
            gs = data.get('graph_stats') or {}
            if gs and self.generator.graph is None:
                text = "从缓存加载的会话 (无图对象, 仅显示已保存的统计)\n"
                text += "=" * 50 + "\n\n"
                text += f"节点数: {gs.get('n_nodes', 0):,}\n"
                text += f"边数:   {gs.get('n_edges', 0):,}\n"
                text += f"生成时间: {gs.get('generation_time', 0):.2f} 秒\n"
                self.analysis_text.delete('1.0', tk.END)
                self.analysis_text.insert('1.0', text)
            elif self.generator.graph is not None:
                self._update_analysis()

            if self._current_view in ('spectrum', 'barcode'):
                self._update_display()

            self._update_status(f"已加载缓存: {data.get('name', '')}")
            self._left_nb.select(0)
            messagebox.showinfo("成功", f"已加载缓存会话:\n{data.get('name', '')}")
        except Exception as e:
            messagebox.showerror("错误", f"加载缓存失败: {str(e)}")

    def _format_topo_text(self):
        lines = []
        lines.append("拓扑学研究报告")
        lines.append("=" * 50)
        lines.append("")

        hom = (self.topo_result or {}).get('homology', {})
        if hom:
            lines.append("【代数拓扑 · 单纯同调】")
            lines.append(f"  节点数 n = {hom.get('n_nodes'):,}")
            lines.append(f"  边数   m = {hom.get('n_edges'):,}")
            lines.append("")
            lines.append(f"  β0 (0维Betti数/连通分支数) = {hom.get('beta_0')}")
            lines.append(f"  β1 (1维Betti数/独立环数)   = {hom.get('beta_1')}")
            lines.append(f"  Euler示性数 χ = n - m     = {hom.get('euler_characteristic')}")
            lines.append(f"  (环秩 Cyclomatic Number = β1 = m-n+β0)")
            lines.append("")
            lines.append(f"  图类型判定:")
            lines.append(f"    森林(无环):    {'是' if hom.get('forest') else '否'}")
            lines.append(f"    树(连通无环): {'是' if hom.get('tree') else '否'}")
            lines.append(f"    单圈图:       {'是' if hom.get('unicyclic') else '否'}")
            lines.append("")

        cyc = (self.topo_result or {}).get('cycles', {})
        if cyc:
            lines.append("【环结构】")
            lines.append(f"  围长 (最短环长度)   = {cyc.get('girth')}")
            if cyc.get('skipped'):
                lines.append(f"  最小环基: 已跳过 (图过大或环数过多, minimum_cycle_basis 计算耗时过长)")
                lines.append(f"  β1 (独立环数)      = {cyc.get('cycle_basis_size', hom.get('beta_1', '?'))} (由公式 m-n+β0 直接给出)")
            else:
                lines.append(f"  最小环基大小(β1)    = {cyc.get('cycle_basis_size')}")
                if cyc.get('cycle_lengths'):
                    lens = cyc['cycle_lengths']
                    lines.append(f"  环长度分布 (前15项) = {lens[:15]}{'...' if len(lens) > 15 else ''}")
                if cyc.get('mean_cycle_length'):
                    lines.append(f"  环基平均长度       = {cyc['mean_cycle_length']:.2f}")
            lines.append("")

        spec = (self.topo_result or {}).get('spectrum')
        if isinstance(spec, dict):
            lines.append("【谱图理论 Spectral】")
            lines.append(f"  谱半径 λ_max(A)           = {spec.get('spectral_radius', 0):.6f}")
            lines.append(f"  代数连通度 λ2(L) (Fiedler) = {spec.get('algebraic_connectivity', 0):.6f}")
            if spec.get('effective_resistance'):
                lines.append(f"  总有效电阻 ΣR_eff         = {spec['effective_resistance']:.2f}")
            lap_vals = spec.get('laplacian_eigenvalues', np.array([]))
            if len(lap_vals) > 0:
                lines.append("")
                lines.append(f"  拉普拉斯谱 (λ0 ~ λ[n-1]) 分位数:")
                qs = np.quantile(lap_vals, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
                labels = ['min   ', 'P10   ', 'Q1    ', 'median', 'Q3    ', 'P90   ', 'max   ']
                for lb, q in zip(labels, qs):
                    lines.append(f"    {lb}: {q:.4f}")
            lines.append("")

        comm = self.community_result
        if isinstance(comm, dict) and comm.get('n_communities'):
            lines.append("【社区结构 (Louvain 贪心模块度)】")
            lines.append(f"  社区数量     = {comm.get('n_communities')}")
            lines.append(f"  模块度 Q     = {comm.get('modularity'):.4f}" if comm.get('modularity') is not None else "  模块度 Q     = N/A")
            lines.append(f"  大小分布熵   = {comm.get('size_entropy', 0):.3f} bits")
            sizes = comm.get('community_sizes', [])
            if sizes:
                preview = sizes[:10]
                lines.append(f"  社区大小(前10) = {preview}{'...' if len(sizes) > 10 else ''}")
                lines.append(f"  最大社区占比   = {comm.get('largest_community_ratio', 0)*100:.1f}%")
            lines.append("")

        cent = (self.topo_result or {}).get('centrality', {})
        if cent:
            lines.append("【拓扑中心性 (Top-5 关键状态)】")
            mapping = [
                ('度中心性 Degree', 'degree_centrality'),
                ('介数中心性 Betweenness (瓶颈)', 'betweenness_centrality'),
                ('接近中心性 Closeness', 'closeness_centrality'),
                ('特征向量中心性 Eigenvector (核心)', 'eigenvector_centrality'),
                ('PageRank', 'pagerank'),
            ]
            for label, key in mapping:
                if key in cent:
                    c = cent[key]
                    lines.append(f"  [{label}]")
                    lines.append(f"    均值 = {c.get('mean', 0):.4f} | 最大 = {c.get('max', 0):.4f}")
                    tops = c.get('top_nodes', [])
                    if tops:
                        tops_str = ", ".join(f"{s:.3g}" for _, s in tops)
                        lines.append(f"    Top-5 分数: {tops_str}")
            lines.append("")

        if 'error' in (self.topo_result or {}):
            lines.append(f"拓扑分析出错: {self.topo_result['error']}")

        # ===== AI 增强分析章节 =====
        ai = getattr(self, '_ai_enhancement', {})
        if ai:
            lines.append("")
            lines.append("【AI 增强分析】")
            # AI 加速报告
            accel = ai.get('acceleration', {})
            if accel:
                accel_items = []
                if accel.get('gnn_parallel'):
                    accel_items.append("GNN并行训练")
                if accel.get('community_kmeans'):
                    accel_items.append("KMeans社区检测(替代greedy)")
                if accel.get('tda_guided_sampling'):
                    accel_items.append("AI引导TDA采样")
                if accel.get('betweenness_skipped'):
                    accel_items.append("介数中心性跳过(嵌入近似)")
                if accel_items:
                    lines.append(f"  ⚡ AI加速: {', '.join(accel_items)}")
            ml = ai.get('ml', {})
            if 'error' in ml:
                lines.append(f"  拓扑特征ML: 失败 ({ml['error']})")
            elif ml:
                shape = ml.get('features_shape', (0, 0))
                lines.append(f"  特征矩阵: {shape[0]} x {shape[1]} (度/聚类/距离等)")
                lines.append(f"  KMeans 聚类: {ml.get('n_clusters', 0)} 簇")
                csizes = ml.get('cluster_sizes', {})
                if csizes:
                    cs = [f"簇{k}:{v}" for k, v in sorted(csizes.items())]
                    lines.append(f"  簇大小: {', '.join(cs)}")
                if ml.get('tsne_2d') is not None:
                    lines.append(f"  降维: PCA + t-SNE (2D)")
                else:
                    lines.append(f"  降维: PCA (2D, t-SNE 跳过)")
            gnn = ai.get('gnn', {})
            if 'error' in gnn:
                lines.append(f"  GNN嵌入: 失败 ({gnn['error']})")
            elif gnn:
                lines.append(f"  GNN 嵌入: {gnn.get('embeddings_shape', (0,0))} "
                             f"(GraphSAGE, hidden=64, out=32)")
            lines.append("  (TDA深度学习结果见 TDA Tab)")
            lines.append("  点击顶栏「AI增强图」查看综合可视化")
            lines.append("")

        self.topo_text.delete('1.0', tk.END)
        self.topo_text.insert('1.0', "\n".join(lines))

    def _format_tda_text(self):
        res = self.tda_result
        lines = []
        lines.append("拓扑数据分析 (TDA) 报告")
        lines.append("=" * 50)
        lines.append("")
        lines.append("方法: Vietoris-Rips 滤过 → 持久同调 (PH)")
        lines.append("距离度量: 图最短路径长度 (无权图 = 步数)")
        lines.append("")

        if not res or 'error' in (res if isinstance(res, dict) else {}):
            lines.append(f"TDA 计算失败: {res.get('error') if isinstance(res, dict) else '未知错误'}")
            self.tda_text.delete('1.0', tk.END)
            self.tda_text.insert('1.0', "\n".join(lines))
            return

        # 守卫: 从缓存加载时 generator.graph 可能为 None (仅重建了 current_state, 未重生成图)
        _g = self.generator.graph
        _n_total = _g.number_of_nodes() if _g is not None else '?'
        lines.append(f"采样节点数: {res.get('n_samples')} / 总节点 {_n_total}")
        shape = res.get('distance_matrix_shape')
        if shape:
            lines.append(f"距离矩阵 D ∈ R^{shape[0]}x{shape[1]}")
        lines.append("")

        stats = res.get('stats', {})
        for dim in sorted(stats.keys()):
            s = stats[dim]
            name_map = {0: 'H0 (0维同调: 连通分支 / 聚类)', 1: 'H1 (1维同调: 独立环 / 洞)'}
            lines.append(f"━━━━ {name_map.get(dim, f'H{dim}')} ━━━━")
            lines.append(f"  Bar 总数      = {s.get('total_bars')}")
            lines.append(f"  有限消亡 Bar  = {s.get('finite_bars')}")
            lines.append(f"  永恒特征 (∞)  = {s.get('infinite_bars')}")
            if s.get('mean_lifetime') is not None:
                lines.append(f"  平均寿命 (d-b) = {s['mean_lifetime']:.3f}")
                lines.append(f"  中位寿命       = {s['median_lifetime']:.3f}")
                lines.append(f"  最大寿命       = {s['max_lifetime']:.3f}")
                lines.append(f"  持久熵 H_pers  = {s['persistence_entropy']:.4f}")
            lines.append("")

        lines.append("研究解读:")
        lines.append("  · H0 的永恒 bar 数 = β0 (真实连通分支)")
        lines.append("  · H0 中有限寿命 bar = 虚假/尺度依赖的聚类")
        lines.append("  · H1 的长寿命 bar  = 状态空间中大尺度的环")
        lines.append("  · 持久熵越高，拓扑特征尺度分布越均匀")
        lines.append("")
        lines.append("点击顶栏「持久条形码」查看可视化。")

        # ===== TDA 深度学习章节 =====
        ai = getattr(self, '_ai_enhancement', {})
        tda_dl = ai.get('tda_dl', {})
        if tda_dl:
            lines.append("")
            lines.append("━━━━ TDA 深度学习 ━━━━")
            if 'error' in tda_dl:
                lines.append(f"  失败: {tda_dl['error']}")
            else:
                lines.append("  方法: 持久条形码 → 持久图像 (Persistence Image)")
                lines.append("        + 自编码器学习压缩表示")
                if tda_dl.get('pimg_h0') is not None:
                    lines.append(f"  H0 持久图像: {tda_dl['pimg_h0'].shape} (resolution=20)")
                if tda_dl.get('pimg_h1') is not None:
                    lines.append(f"  H1 持久图像: {tda_dl['pimg_h1'].shape} (resolution=20)")
                feat_shape = tda_dl.get('feat_shape')
                if feat_shape:
                    lines.append(f"  特征向量维度: {feat_shape}")
                lines.append("  自编码器: 50 epochs (特征压缩)")
                lines.append("  点击顶栏「AI增强图」查看持久图像可视化")
            lines.append("")

        self.tda_text.delete('1.0', tk.END)
        self.tda_text.insert('1.0', "\n".join(lines))

    # ==================== 基本分析 ====================

    def _update_analysis(self):
        if self.generator.graph is None or len(self.generator.graph.nodes) == 0:
            return
        analysis = self.generator.analyze()
        text = "状态空间分析报告\n"
        text += "=" * 50 + "\n\n"
        text += f"状态数: {analysis['num_states']:,}\n"
        text += f"转换数: {analysis['num_transitions']:,}\n"
        text += f"生成时间: {analysis['generation_time']:.2f} 秒\n"
        text += f"图密度: {analysis['density']:.6f}\n"
        text += f"连通性: {'是' if analysis['connected'] else '否'}\n"
        if analysis['connected']:
            text += f"直径: {analysis.get('diameter', 'N/A')}\n"
        text += f"\n【度统计】\n"
        text += f"  平均度: {analysis['avg_degree']:.2f}\n"
        text += f"  最大度: {analysis['max_degree']}\n"
        text += f"  最小度: {analysis['min_degree']}\n"
        if not analysis['connected'] and 'num_components' in analysis:
            text += f"\n【连通分量】\n"
            text += f"  分量数: {analysis['num_components']}\n"
            text += f"  分量大小: {analysis['component_sizes']}\n"
        if analysis['stopped_early']:
            text += "\n生成被中断\n"
        self.analysis_text.delete('1.0', tk.END)
        self.analysis_text.insert('1.0', text)

    # ==================== 导出 / 保存 ====================

    def _export_report(self):
        if self.generator.graph is None:
            messagebox.showwarning("警告", "请先生成状态空间")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not file_path:
            return

        analysis = self.generator.analyze()
        G = self.generator.graph
        n = G.number_of_nodes()
        m = G.number_of_edges()

        sections = []
        sections.append("华容道状态空间 完整研究报告")
        sections.append("=" * 60)
        sections.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        sections.append("")
        sections.append("【1. 基本图统计】")
        sections.append(f"  状态数 n  = {analysis['num_states']:,}")
        sections.append(f"  转换数 m  = {analysis['num_transitions']:,}")
        sections.append(f"  生成时间  = {analysis['generation_time']:.2f} 秒")
        sections.append(f"  图密度 δ  = {analysis['density']:.6f}  (= 2m / [n(n-1)])")
        sections.append(f"  连通性    = {'是' if analysis['connected'] else '否'}")
        if analysis['connected']:
            sections.append(f"  图直径 D  = {analysis.get('diameter', 'N/A')}")
        sections.append(f"  平均度    = {analysis['avg_degree']:.2f}")
        sections.append(f"  最大度    = {analysis['max_degree']}")
        sections.append(f"  最小度    = {analysis['min_degree']}")
        if not analysis['connected'] and 'num_components' in analysis:
            sections.append(f"  连通分量数 = {analysis['num_components']}")
            sections.append(f"  分量大小   = {analysis['component_sizes']}")
        sections.append("")

        if self.topo_result and 'homology' in self.topo_result:
            hom = self.topo_result['homology']
            sections.append("=" * 60)
            sections.append("【2. 代数拓扑 · 单纯同调不变量】")
            sections.append(f"    β0 (H0秩, 连通分支)   = {hom.get('beta_0')}")
            sections.append(f"    β1 (H1秩, 独立环数)   = {hom.get('beta_1')}  (= m - n + β0)")
            sections.append(f"    Euler 示性数 χ        = {hom.get('euler_characteristic')}  (= n - m = β0 - β1)")
            sections.append(f"    森林 β1=0          : {'是' if hom.get('forest') else '否'}")
            sections.append(f"    树  β1=0 ∧ β0=1    : {'是' if hom.get('tree') else '否'}")
            sections.append("")

        if self.topo_result and 'cycles' in self.topo_result:
            cyc = self.topo_result['cycles']
            sections.append("=" * 60)
            sections.append("【3. 环结构 (Cycle Space)】")
            sections.append(f"  围长 girth (最短环)     = {cyc.get('girth')}")
            sections.append(f"  最小环基 |C|            = {cyc.get('cycle_basis_size')}  (应=β1)")
            if cyc.get('cycle_lengths'):
                ls = cyc['cycle_lengths']
                sections.append(f"  环长度 Top10            = {ls[:10]}")
            sections.append("")

        if self.topo_result and isinstance(self.topo_result.get('spectrum'), dict):
            spec = self.topo_result['spectrum']
            sections.append("=" * 60)
            sections.append("【4. 谱图理论 Spectral Graph Theory】")
            sections.append(f"  邻接谱半径 ρ(A)         = {spec.get('spectral_radius', 0):.6f}")
            sections.append(f"  代数连通度 λ2(L)        = {spec.get('algebraic_connectivity', 0):.6f}  (Fiedler值)")
            if spec.get('effective_resistance') is not None:
                sections.append(f"  总有效电阻 ΣR_eff       = {spec['effective_resistance']:.2f}")
            sections.append("")

        if self.community_result and self.community_result.get('n_communities'):
            comm = self.community_result
            sections.append("=" * 60)
            sections.append("【5. 社区结构检测 (Louvain, 模块度最优)】")
            sections.append(f"  社区数量       = {comm.get('n_communities')}")
            Q = comm.get('modularity')
            if Q is not None:
                sections.append(f"  模块度 Q       = {Q:.4f}")
            sizes = comm.get('community_sizes', [])
            if sizes:
                sections.append(f"  社区大小 (降序 Top10): {sizes[:10]}{'...' if len(sizes) > 10 else ''}")
            sections.append("")

        if self.tda_result and 'stats' in self.tda_result:
            stats = self.tda_result['stats']
            n_s = self.tda_result.get('n_samples', 0)
            sections.append("=" * 60)
            sections.append("【6. 拓扑数据分析 TDA (持久同调 PH)】")
            sections.append(f"  VR 滤过 | 样本 {n_s} 节点 (采样最多200)")
            for dim in sorted(stats.keys()):
                s = stats[dim]
                name_map = {0: 'H0 (0维: 连通分支/聚类)', 1: 'H1 (1维: 独立环/洞)'}
                sections.append(f"  ▶ {name_map.get(dim, f'H{dim}')}:")
                sections.append(f"      Bar 总数      = {s.get('total_bars')}")
                sections.append(f"      有限消亡      = {s.get('finite_bars')}")
                sections.append(f"      永恒特征(∞)   = {s.get('infinite_bars')}")
                if s.get('mean_lifetime') is not None:
                    sections.append(f"      平均寿命      = {s['mean_lifetime']:.3f}")
                    sections.append(f"      持久熵 H_pers = {s['persistence_entropy']:.4f}")
            sections.append("")

        report = "\n".join(sections)
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(report)
            messagebox.showinfo("成功", f"拓扑研究报告已保存到:\n{file_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def _save_figure(self):
        if not self.generator.graph:
            messagebox.showwarning("警告", "没有可保存的图")
            return
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png", filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")])
        if file_path:
            self.fig.savefig(file_path, dpi=300, bbox_inches='tight')
            messagebox.showinfo("成功", f"已保存到 {file_path}")

    def _reset_view(self):
        """把 3D 相机复位到初始视角

        之前这里只调 _update_display(): 而 display_key 没变时它会直接早退,
        所以这个按钮/菜单/Ctrl+R 三处入口全都是空操作 —— 相机不复位, 还把
        工具栏右侧那行"棋盘: 4x4, 方块: 10"统计覆盖成了"视图已重置"。
        """
        self.view_elev = 22
        self.view_azim = 35
        self.view_dist = 10.0
        if getattr(self, 'ax_graph', None) is not None:
            try:
                self.ax_graph.view_init(elev=self.view_elev, azim=self.view_azim)
            except Exception:
                pass
        self._last_display_key = None   # 不置空的话 _update_display 会认为"无变化"而早退
        self._update_display()
        self._update_status("视图已重置")

    # ==================== 状态栏 / 关于 ====================

    def _update_status(self, message):
        self.status_label.config(text=f"状态: {message}")
        self.sb_status.config(text=message)

    def _show_about(self):
        about = tk.Toplevel(self.root)
        about.title("关于")
        about.geometry("420x300")
        about.configure(bg=Theme.PANEL)
        about.resizable(False, False)

        tk.Label(about, text="华容道状态空间分析器", font=("Microsoft YaHei UI", 14, "bold"),
                 bg=Theme.PANEL, fg=Theme.ACCENT).pack(pady=(24, 8))
        tk.Label(about, text="拓扑学与持久同调研究工具", font=Theme.FONT,
                 bg=Theme.PANEL, fg=Theme.MUTED).pack(pady=(0, 16))

        info = (
            "功能模块:\n"
            "  · 状态空间 BFS 生成\n"
            "  · 代数拓扑(β0/β1/χ/谱/中心性)\n"
            "  · TDA 持久同调(Vietoris-Rips)\n"
            "  · Louvain 社区检测\n"
            "  · 3D 可视化 / 布局绘制\n"
        )
        tk.Label(about, text=info, font=Theme.FONT, bg=Theme.PANEL, fg=Theme.TEXT,
                 justify=tk.LEFT).pack(padx=20)

        ttk.Button(about, text="确定", command=about.destroy).pack(pady=16)

    # ==================== AI 分析 ====================

    def _ai_clear(self, msg):
        """清空拓扑文本框并写入首行"""
        w = getattr(self, 'topo_text', None)
        if w is not None:
            w.delete('1.0', tk.END)
            w.insert('1.0', msg)
            w.see('end')
            w.update_idletasks()

    def _ai_append(self, msg):
        """向拓扑文本框追加文本并自动滚动到底部"""
        w = getattr(self, 'topo_text', None)
        if w is not None:
            w.insert('end', msg)
            w.see('end')
            w.update_idletasks()

    def _ai_replace_line(self, line, msg):
        """替换拓扑文本框指定行并滚动"""
        w = getattr(self, 'topo_text', None)
        if w is not None:
            w.delete(f'{line}.0', f'{line}.end')
            w.insert(f'{line}.0', msg)
            w.see('end')
            w.update_idletasks()

    def _show_ai_figure(self, figure):
        """在中心画布显示AI分析生成的matplotlib Figure

        AI worker 在后台线程用 Figure 直接创建图形 (无 GUI manager),
        此处在主线程将源 figure 整体渲染为 RGBA 图像后用 imshow 显示,
        避免逐 collection 复制带来的类型/axes 兼容性问题 (如 LineCollection、
        colorbar axes), 同时不触发跨线程 Tk 操作。

        注意: 2026-09-17 这个方法曾被"切除 RL 残留"时的截断误删 (它紧跟在 _run_ai_rl 之后),
        导致 _switch_view('ai') 直接 AttributeError —— AI增强图整条链断掉。
        已恢复并加了自检断言 (自检.py 的「GUI 主窗口」会检查 _show_ai_figure /
        _build_ai_figure / _重建AI图并记录 / _聚类取色 四个方法齐全, 且 AI 视图真能出图)。
        """
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        # 清除当前 axes
        self.fig.clear()
        try:
            # 为源 figure 附加 Agg canvas 并渲染到 numpy 数组
            agg = FigureCanvasAgg(figure)
            agg.draw()
            rgba = np.asarray(agg.buffer_rgba())
        except Exception:
            rgba = None

        ax = self.fig.add_subplot(111)
        if rgba is not None:
            ax.imshow(rgba, aspect='equal')
            ax.set_title("AI 分析结果", fontproperties=CN_FONT, fontsize=11)
        else:
            ax.text(0.5, 0.5, "无法显示图像", transform=ax.transAxes,
                    ha='center', va='center', fontproperties=CN_FONT, fontsize=12)
        ax.set_axis_off()
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
        figure.clear()
        self.canvas.draw_idle()
