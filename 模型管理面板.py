# -*- coding: utf-8 -*-
"""模型管理面板 (模型库 + 布局级存档)

概念对齐 (与 布局哈希.py 一致):
  • 「模型库」= 一个**含 模型\\ 子目录**的目录, 例如 训练数据 / 训练数据_实验A。
    主界面左面板的「模型来源」下拉 (scan_model_sources) 列出的就是这些库, 当前选中的库
    决定了 AI 分析 / GNN 微调读写哪个目录。
  • 库内 模型\\ 下的文件分两层:
      - 全局模型: gnn_global.pt / tda_ae_global.pt / ml_global.pkl (跨布局迁移用)
      - 布局级存档: graph_<hash>.pkl / gnn_<hash>.pt / tda_ae_<hash>.pt / ml_<hash>.pkl /
        embeddings_<hash>.npy / result_<hash>.json (每个哈希 = 一个布局)
    布局级存档由 拓扑训练器 或主程序「全拓扑分析」产生, 这里做查看与清理。

对外接口:
  on_switch(库名)   切换当前模型来源时回调 (主界面用来同步下拉与 布局哈希.set_model_base)
"""
import os
import shutil
import tkinter as tk
from tkinter import ttk, messagebox

from 布局哈希 import scan_model_sources, get_model_base, set_model_base
from 主题 import Theme

# 布局级存档的前缀 (与 布局哈希.layout_model_paths 对应)
布局前缀 = ('graph_', 'gnn_', 'tda_ae_', 'ml_', 'embeddings_', 'result_')
全局文件 = ('gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl')


def _库基目录(库名: str) -> str:
    """模型库目录的完整路径 (库名可为相对 基准目录 的名字或绝对路径)"""
    if os.path.isabs(库名):
        return 库名
    from 布局哈希 import _BASE_DIR
    return os.path.join(_BASE_DIR, 库名)


def _目录体积(路径: str, 文件数上限: int = 200000):
    """递归统计 (字节数, 文件数, mtime)"""
    总 = 0
    个数 = 0
    最新 = 0.0
    if not os.path.isdir(路径):
        return 0, 0, 0.0
    for 根, _目录们, 文件们 in os.walk(路径):
        for f in 文件们:
            p = os.path.join(根, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            总 += st.st_size
            个数 += 1
            最新 = max(最新, st.st_mtime)
            if 个数 > 文件数上限:
                return 总, 个数, 最新
    return 总, 个数, 最新


def _人类体积(n: float) -> str:
    for 单位 in ('B', 'KB', 'MB', 'GB'):
        if n < 1024 or 单位 == 'GB':
            return '%.1f %s' % (n, 单位)
        n /= 1024
    return '%.1f GB' % n


def _拆分哈希(文件名: str):
    """把 'graph_91be1fcc.pkl' 拆成 ('graph', '91be1fcc')；不匹配返回 (None, None)"""
    for 前缀 in 布局前缀:
        if 文件名.startswith(前缀):
            后半 = 文件名[len(前缀):]
            主名, _, 扩展 = 后半.rpartition('.')
            if 主名:
                return 前缀.rstrip('_'), 主名
    return None, None


class ModelManagerPanel(ttk.Frame):
    """模型管理面板"""

    def __init__(self, parent, on_switch=None, 风格: str = 'Panel.TFrame', **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style=风格)   # 用 configure 而不是构造参数, 避免和子控件的 style 选项撞名
        self.on_switch = on_switch
        self._库行 = {}          # tree item -> 库名
        self._布局行 = {}        # tree item -> (哈希, 该哈希下的文件列表)
        self._build_ui()
        self.refresh()

    # ---------- UI ----------
    def _build_ui(self):
        # 顶部: 当前模型来源 + 操作
        顶 = ttk.Frame(self)
        顶.pack(fill='x', padx=8, pady=(8, 4))

        self.var_current = tk.StringVar(value=get_model_base() or '训练数据')
        ttk.Label(顶, text="当前模型来源:").pack(side='left')
        self.lbl_current = ttk.Label(顶, textvariable=self.var_current,
                                     foreground=Theme.ACCENT_TEXT, font=('Microsoft YaHei UI', 9, 'bold'))
        self.lbl_current.pack(side='left', padx=(4, 12))

        self.btn_new = ttk.Button(顶, text="＋ 新建模型库", command=self._on_new)
        self.btn_new.pack(side='left', padx=2)
        self.btn_switch = ttk.Button(顶, text="设为当前", command=self._on_switch)
        self.btn_switch.pack(side='left', padx=2)
        self.btn_refresh = ttk.Button(顶, text="刷新", command=self.refresh)
        self.btn_refresh.pack(side='left', padx=2)

        # 中部: 模型库列表
        库框 = ttk.LabelFrame(self, text="模型库 (含 模型\\ 子目录的目录)", padding=4)
        库框.pack(fill='both', expand=True, padx=8, pady=4)

        列 = ('name', 'size', 'files', 'global', 'layouts', 'time')
        self.tree = ttk.Treeview(库框, columns=列, show='headings', height=7)
        表头 = {'name': '名称', 'size': '体积', 'files': '文件数',
                'global': '全局模型', 'layouts': '布局存档', 'time': '最近修改'}
        宽度 = {'name': 190, 'size': 80, 'files': 70, 'global': 100, 'layouts': 80, 'time': 130}
        for c in 列:
            self.tree.heading(c, text=表头[c])
            self.tree.column(c, width=宽度[c], minwidth=宽度[c], stretch=False,
                             anchor=tk.W if c == 'name' else tk.CENTER)
        sb = ttk.Scrollbar(库框, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side='left', fill='both', expand=True)
        sb.pack(side='right', fill='y')
        self.tree.bind('<<TreeviewSelect>>', lambda e: self._on_库选中())
        self.tree.bind('<Double-1>', lambda e: self._on_switch())

        # 库操作
        库按钮 = ttk.Frame(self)
        库按钮.pack(fill='x', padx=8)
        ttk.Button(库按钮, text="重命名", command=self._on_rename).pack(side='left', padx=2)
        ttk.Button(库按钮, text="删除模型库", command=self._on_delete).pack(side='left', padx=2)
        ttk.Button(库按钮, text="打开目录", command=self._open_dir).pack(side='left', padx=2)

        # 下部: 布局级存档
        布局框 = ttk.LabelFrame(self, text="布局级存档 (选中模型库内, 按布局哈希分组)", padding=4)
        布局框.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        列2 = ('hash', 'files', 'size', 'time')
        self.tree2 = ttk.Treeview(布局框, columns=列2, show='headings', height=8)
        表头2 = {'hash': '布局哈希', 'files': '文件', 'size': '体积', 'time': '最近修改'}
        宽度2 = {'hash': 150, 'files': 380, 'size': 80, 'time': 130}
        for c in 列2:
            self.tree2.heading(c, text=表头2[c])
            self.tree2.column(c, width=宽度2[c], minwidth=60, stretch=(c == 'files'),
                              anchor=tk.W if c in ('hash', 'files') else tk.CENTER)
        sb2 = ttk.Scrollbar(布局框, orient='vertical', command=self.tree2.yview)
        self.tree2.configure(yscrollcommand=sb2.set)
        self.tree2.pack(side='left', fill='both', expand=True)
        sb2.pack(side='right', fill='y')

        布局按钮 = ttk.Frame(self)
        布局按钮.pack(fill='x', padx=8, pady=(0, 8))
        ttk.Button(布局按钮, text="删除选中布局存档", command=self._on_delete_layout).pack(side='left', padx=2)
        ttk.Button(布局按钮, text="打开所在目录", command=self._open_layout_dir).pack(side='left', padx=2)

    # ---------- 数据刷新 ----------
    def 选中库名(self):
        sel = self.tree.selection()
        return self._库行.get(sel[0]) if sel else None

    def refresh(self):
        self.var_current.set(get_model_base() or '训练数据')
        self.tree.delete(*self.tree.get_children())
        self._库行.clear()
        for 库 in scan_model_sources():
            路径 = _库基目录(库)
            模型目录 = os.path.join(路径, '模型')
            总, 个数, 最新 = _目录体积(模型目录)
            全局 = sum(1 for f in 全局文件 if os.path.exists(os.path.join(模型目录, f)))
            存档 = 0
            if os.path.isdir(模型目录):
                for f in os.listdir(模型目录):
                    主, 哈希 = _拆分哈希(f)
                    if 主 == 'graph' and 哈希:
                        存档 += 1
            当前 = '● ' if 库 == (get_model_base() or '训练数据') else ''
            item = self.tree.insert('', 'end', values=(
                当前 + 库, _人类体积(总), '%d' % 个数,
                '%d / %d' % (全局, len(全局文件)), '%d' % 存档,
                self._时间(最新) if 最新 else '-',
            ))
            self._库行[item] = 库
        # 默认选中当前库
        当前库 = get_model_base() or '训练数据'
        for item, 库 in self._库行.items():
            if 库 == 当前库:
                self.tree.selection_set(item)
                break
        self._刷新布局表()

    @staticmethod
    def _时间(ts: float) -> str:
        import time
        return time.strftime('%Y-%m-%d %H:%M', time.localtime(ts))

    def _刷新布局表(self):
        self.tree2.delete(*self.tree2.get_children())
        self._布局行.clear()
        库 = self.选中库名()
        if not 库:
            return
        模型目录 = os.path.join(_库基目录(库), '模型')
        if not os.path.isdir(模型目录):
            return
        分组 = {}
        for f in sorted(os.listdir(模型目录)):
            完整 = os.path.join(模型目录, f)
            if not os.path.isfile(完整):
                continue
            # 全局模型 (gnn_global.pt 等) 不是布局级存档, 跳过 —— 否则会被当成哈希 'global' 列出来
            if f in 全局文件:
                continue
            主, 哈希 = _拆分哈希(f)
            if 哈希:
                分组.setdefault(哈希, []).append((f, 完整))
        for 哈希, 文件们 in sorted(分组.items()):
            总 = 0
            最新 = 0.0
            for _f, p in 文件们:
                try:
                    st = os.stat(p)
                    总 += st.st_size
                    最新 = max(最新, st.st_mtime)
                except OSError:
                    pass
            摘要 = ', '.join(sorted(f.replace(哈希, '…') for f, _p in 文件们))
            item = self.tree2.insert('', 'end', values=(
                哈希, 摘要, _人类体积(总), self._时间(最新) if 最新 else '-'))
            self._布局行[item] = (哈希, [p for _f, p in 文件们])

    def _on_库选中(self):
        self._刷新布局表()

    # ---------- 库操作 ----------
    def _on_new(self):
        from tkinter import simpledialog
        名字 = simpledialog.askstring(
            "新建模型库",
            "输入模型库名称 (将在工程目录下新建同名目录, 内含 模型\\ 子目录):\n\n"
            "例如: 训练数据_实验A",
            parent=self)
        if not 名字:
            return
        名字 = 名字.strip()
        if not 名字 or os.path.sep in 名字 or '/' in 名字:
            messagebox.showerror("错误", "名称不能为空, 且不能包含路径分隔符", parent=self)
            return
        路径 = _库基目录(名字)
        if os.path.exists(路径):
            messagebox.showerror("错误", "同名目录已存在:\n%s" % 路径, parent=self)
            return
        try:
            os.makedirs(os.path.join(路径, '模型'), exist_ok=True)
            os.makedirs(os.path.join(路径, '布局缓存'), exist_ok=True)
        except OSError as e:
            messagebox.showerror("错误", "创建失败: %s" % e, parent=self)
            return
        self.refresh()
        # 新建后直接设为当前, 并选中它
        for item, 库 in self._库行.items():
            if 库 == 名字:
                self.tree.selection_set(item)
                self.tree.see(item)
                break
        self._切换(名字)
        messagebox.showinfo("完成",
                            "已新建模型库: %s\n\n目录结构:\n  %s\\模型\\\n  %s\\布局缓存\\\n\n"
                            "已设为当前模型来源, 训练器的输出目录也会跟着切过去。"
                            % (名字, 名字, 名字), parent=self)

    def _切换(self, 库: str):
        set_model_base(库)
        self.var_current.set(库)
        if callable(self.on_switch):
            self.on_switch(库)

    def _on_switch(self):
        库 = self.选中库名()
        if not 库:
            messagebox.showinfo("提示", "请先在上表选择一个模型库", parent=self)
            return
        self._切换(库)
        self.refresh()

    def _on_rename(self):
        from tkinter import simpledialog
        库 = self.选中库名()
        if not 库:
            messagebox.showinfo("提示", "请先在上表选择一个模型库", parent=self)
            return
        原名 = os.path.basename(库)
        新名 = simpledialog.askstring("重命名模型库", "新名称:", initialvalue=原名, parent=self)
        if not 新名 or 新名.strip() == 原名:
            return
        新名 = 新名.strip()
        if os.path.sep in 新名 or '/' in 新名:
            messagebox.showerror("错误", "名称不能包含路径分隔符", parent=self)
            return
        旧路径, 新路径 = _库基目录(库), _库基目录(新名)
        if os.path.exists(新路径):
            messagebox.showerror("错误", "目标已存在:\n%s" % 新路径, parent=self)
            return
        try:
            os.rename(旧路径, 新路径)
        except OSError as e:
            messagebox.showerror("错误", "重命名失败: %s" % e, parent=self)
            return
        if (get_model_base() or '') == 原名:
            set_model_base(新名)
            if callable(self.on_switch):
                self.on_switch(新名)
        self.refresh()

    def _on_delete(self):
        库 = self.选中库名()
        if not 库:
            messagebox.showinfo("提示", "请先在上表选择一个模型库", parent=self)
            return
        路径 = _库基目录(库)
        总, 个数, _ = _目录体积(路径)
        if not messagebox.askyesno(
                "确认删除",
                "将**整个目录**删除, 不可恢复:\n\n%s\n\n体积 %s, %d 个文件\n\n确定继续?"
                % (路径, _人类体积(总), 个数), parent=self):
            return
        try:
            shutil.rmtree(路径)
        except OSError as e:
            messagebox.showerror("错误", "删除失败: %s" % e, parent=self)
            return
        if (get_model_base() or '') == os.path.basename(库):
            set_model_base('训练数据')
            if callable(self.on_switch):
                self.on_switch('训练数据')
        self.refresh()

    def _open_dir(self):
        库 = self.选中库名()
        if not 库:
            return
        self._打开(_库基目录(库))

    def _open_layout_dir(self):
        库 = self.选中库名()
        if not 库:
            return
        self._打开(os.path.join(_库基目录(库), '模型'))

    @staticmethod
    def _打开(路径: str):
        import sys
        try:
            os.makedirs(路径, exist_ok=True)
            if sys.platform.startswith('win'):
                os.startfile(路径)
            elif sys.platform == 'darwin':
                os.system('open "%s"' % 路径)
            else:
                os.system('xdg-open "%s"' % 路径)
        except Exception as e:
            messagebox.showerror("错误", "无法打开目录: %s" % e)

    # ---------- 布局级存档 ----------
    def _on_delete_layout(self):
        sel = self.tree2.selection()
        if not sel:
            messagebox.showinfo("提示", "请先在下表选择一行布局存档", parent=self)
            return
        哈希, 文件们 = self._布局行[sel[0]]
        if not messagebox.askyesno(
                "确认删除",
                "删除布局 %s 的 %d 个存档文件?\n\n%s"
                % (哈希, len(文件们), '\n'.join(os.path.basename(p) for p in 文件们)),
                parent=self):
            return
        失败 = []
        for p in 文件们:
            try:
                os.remove(p)
            except OSError as e:
                失败.append('%s: %s' % (os.path.basename(p), e))
        self._刷新布局表()
        if 失败:
            messagebox.showerror("部分失败", '\n'.join(失败), parent=self)
        else:
            messagebox.showinfo("完成", "已删除布局 %s 的存档" % 哈希, parent=self)
