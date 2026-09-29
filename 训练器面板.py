# -*- coding: utf-8 -*-
"""拓扑训练器面板 (可嵌入主界面页签, 也可独立开窗)

拆分背景 (2026-09-17):
  原来 GUI 和训练流水线都挤在 拓扑训练器.py 里, 而且面板把 self.root 当成"自己的窗口":
  构造时就 set title/geometry, 所有 UI 都 pack 到 root 上, 还用 self.root.after() 做跨线程调度。
  于是它只能独立开窗, 主界面里塞不进去。

  现在:
    • 训练流水线 (DynamicLayoutGenerator / topology_train_pipeline / CPULimiter / 资源工具)
      仍留在 拓扑训练器.py, 作为共享核心;
    • 本文件只负责 GUI, 做成一个 ttk.Frame 子类 TrainerPanel(parent, ...);
    • 跨线程调度统一走 self.after() (Frame 自己的 after), 不再依赖 root;
    • 独立模式由 拓扑训练器.run_standalone() 建一个 Tk 窗口把这个面板放进去。

  模型的"库"概念: 面板的「输出目录」指向一个模型库 (含 模型\\ 子目录的目录),
  主界面左面板的「模型来源」下拉与它共享同一份状态 —— 在训练页新建/切换模型库,
  主程序的 AI 分析与 GNN 微调就会读写同一个目录。
"""
import os
import sys
import threading
import time
from collections import Counter
from typing import List, Optional

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np

from 拓扑训练器 import (
    CPULimiter, DynamicLayoutGenerator, topology_train_pipeline,
    _release_memory, _lower_process_priority, _n_cpu,
)
from 数据模型 import State, Block


class TrainerPanel(ttk.Frame):
    """拓扑训练器面板 (嵌主界面「训练」页 或 独立窗口内)"""

    def __init__(self, parent, 独立模式: bool = False, 初始输出目录: str = '训练数据',
                 风格: str = 'Panel.TFrame', **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style=风格)   # 用 configure 而不是构造参数, 避免和子控件的 style 选项撞名
        self.独立模式 = 独立模式
        self._win = parent.winfo_toplevel()      # 仅用于独立模式设置窗口标题/图标

        # 运行控制
        self._running = False
        self._stop_requested = False
        self._worker_thread: Optional[threading.Thread] = None
        self._results: List[dict] = []

        # UI 日志节流缓冲 (工作线程高频日志合并写入, 降低 Tk 主线程压力)
        self._log_buffer: List[str] = []
        self._log_flush_pending = False

        # 降低本进程优先级 (不抢其他程序 CPU; 自己仍能跑满空闲 CPU, 不影响 AI 性能)
        _lower_process_priority()

        self._build_ui(初始输出目录)
        self._log("拓扑学训练器就绪。\n"
                  "功能: 代数拓扑(Betti/χ) + 谱图理论(λ₂/谱半径) + TDA持久同调 + "
                  "社区检测 + 拓扑特征ML + GNN嵌入 + TDA自编码器\n"
                  "完全不含 RL 求解。\n")

    # ---------- UI 构建 ----------
    def _build_ui(self, 初始输出目录: str):
        # 顶部参数面板
        param_frame = ttk.LabelFrame(self, text="训练参数", padding=8)
        param_frame.pack(fill='x', padx=8, pady=(8, 4))

        self.var_n = tk.IntVar(value=50)
        self.var_min_dim = tk.IntVar(value=2)
        self.var_max_dim = tk.IntVar(value=6)
        self.var_gnn_epochs = tk.IntVar(value=200)
        self.var_max_nodes = tk.IntVar(value=200000)
        self.var_seed = tk.IntVar(value=42)
        self.var_resume = tk.BooleanVar(value=True)
        self.var_output = tk.StringVar(value=初始输出目录 or '训练数据')
        # CPU 占用率上限 (默认 50%, 不跑满)
        self.var_cpu_limit = tk.IntVar(value=50)

        r = 0
        ttk.Label(param_frame, text="布局数量:").grid(row=r, column=0, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_n, width=8).grid(row=r, column=1, padx=4)
        ttk.Label(param_frame, text="棋盘尺寸范围:").grid(row=r, column=2, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_min_dim, width=4).grid(row=r, column=3)
        ttk.Label(param_frame, text="~").grid(row=r, column=4)
        ttk.Entry(param_frame, textvariable=self.var_max_dim, width=4).grid(row=r, column=5, padx=4)

        r = 1
        ttk.Label(param_frame, text="GNN训练轮数:").grid(row=r, column=0, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_gnn_epochs, width=8).grid(row=r, column=1, padx=4)
        ttk.Label(param_frame, text="状态空间节点上限:").grid(row=r, column=2, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_max_nodes, width=10).grid(
            row=r, column=3, columnspan=3, sticky='w', padx=4)

        r = 2
        ttk.Label(param_frame, text="随机种子:").grid(row=r, column=0, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_seed, width=8).grid(row=r, column=1, padx=4)
        ttk.Checkbutton(param_frame, text="启用自我迭代 (加载已有模型继续微调)",
                        variable=self.var_resume).grid(row=r, column=2, columnspan=3,
                                                       sticky='w', padx=4)

        r = 3
        ttk.Label(param_frame, text="输出目录:").grid(row=r, column=0, sticky='w', padx=4)
        ttk.Entry(param_frame, textvariable=self.var_output, width=25).grid(
            row=r, column=1, columnspan=3, sticky='w', padx=4)

        r = 4
        ttk.Label(param_frame, text="CPU占用上限:").grid(row=r, column=0, sticky='w', padx=4)
        cpu_scale = ttk.Scale(param_frame, from_=10, to=90, variable=self.var_cpu_limit,
                              orient='horizontal', length=120,
                              command=lambda v: self.var_cpu_limit.set(int(float(v))))
        cpu_scale.set(50)
        cpu_scale.grid(row=r, column=1, columnspan=2, sticky='ew', padx=4)
        self.lbl_cpu_value = ttk.Label(param_frame, text="50%", width=6)
        self.var_cpu_limit.trace_add('write', lambda *_: self.lbl_cpu_value.config(
            text=f"{self.var_cpu_limit.get()}%"))
        ttk.Label(param_frame, text="(不跑满)").grid(row=r, column=3, sticky='w')

        # 控制按钮
        btn_frame = ttk.Frame(param_frame)
        btn_frame.grid(row=0, column=6, rowspan=5, padx=16, sticky='ns')
        self.btn_start = ttk.Button(btn_frame, text="▶ 开始训练", width=16, command=self._on_start)
        self.btn_start.pack(pady=2)
        self.btn_stop = ttk.Button(btn_frame, text="■ 停止", width=16, command=self._on_stop,
                                   state='disabled')
        self.btn_stop.pack(pady=2)
        self.btn_clear = ttk.Button(btn_frame, text="清空日志", width=16, command=self._clear_log)
        self.btn_clear.pack(pady=2)
        self.btn_open_dir = ttk.Button(btn_frame, text="打开输出目录", width=16,
                                       command=self._open_output)
        self.btn_open_dir.pack(pady=2)

        # 进度条
        prog_frame = ttk.LabelFrame(self, text="进度", padding=8)
        prog_frame.pack(fill='x', padx=8, pady=4)
        self.var_progress = tk.DoubleVar(value=0)
        self.var_progress_text = tk.StringVar(value="0 / 0")
        self.progress = ttk.Progressbar(prog_frame, variable=self.var_progress, maximum=100)
        self.progress.pack(side='left', fill='x', expand=True, padx=(0, 8))
        ttk.Label(prog_frame, textvariable=self.var_progress_text, width=16).pack(side='right')

        # 中部: 日志 + 汇总 (垂直分栏)
        main_paned = ttk.PanedWindow(self, orient='vertical')
        main_paned.pack(fill='both', expand=True, padx=8, pady=4)

        log_frame = ttk.LabelFrame(main_paned, text="运行日志", padding=4)
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap='word',
                                                  font=('Consolas', 9), height=12)
        self.log_text.pack(fill='both', expand=True)
        main_paned.add(log_frame, weight=3)

        sum_frame = ttk.LabelFrame(main_paned, text="结果汇总 (最近50条)", padding=4)
        cols = ("idx", "size", "n_nodes", "b0", "b1", "euler", "lambda2",
                "modularity", "communities", "tda_h0", "tda_h1", "time", "resumed")
        self.tree = ttk.Treeview(sum_frame, columns=cols, show='headings', height=8)
        # 列宽按"实际内容 + 余量"定, 合计约 694px, 要能塞进训练页宽度 (左侧面板展开到 760px 时可用约 740px)。
        # 原先是每列 50~80 的均分写法, 13 列合计 830px, 右侧会被裁掉 (2026-09-17 实测)。
        列宽 = {'idx': 40, 'size': 54, 'n_nodes': 62, 'b0': 40, 'b1': 40, 'euler': 44,
                'lambda2': 58, 'modularity': 62, 'communities': 70, 'tda_h0': 50,
                'tda_h1': 50, 'time': 56, 'resumed': 58}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=列宽[c], minwidth=32, stretch=False, anchor='center')
        # 面板被拖窄时兜底: 给个横向滚动条, 免得右边几列直接看不到
        xsb = ttk.Scrollbar(sum_frame, orient='horizontal', command=self.tree.xview)
        vsb = ttk.Scrollbar(sum_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=xsb.set)
        vsb.pack(side='right', fill='y')
        xsb.pack(side='bottom', fill='x')
        self.tree.pack(side='left', fill='both', expand=True)
        main_paned.add(sum_frame, weight=2)

        # 底部统计栏
        self.var_status = tk.StringVar(value="准备就绪")
        status = ttk.Label(self, textvariable=self.var_status, anchor='w',
                           relief='sunken', padding=(8, 3))
        status.pack(fill='x', padx=8, pady=(0, 8))

    # ---------- 对外接口 ----------
    def 设置输出目录(self, 目录: str):
        """切换训练输出目录 (模型库), 供主界面「模型来源」联动"""
        if 目录:
            self.var_output.set(目录)

    def 当前输出目录(self) -> str:
        return self.var_output.get().strip() or '训练数据'

    def 正在训练(self) -> bool:
        return self._running

    def 请求停止(self):
        """供主界面关闭窗口时调用: 让工作线程尽快收尾"""
        if self._running:
            self._stop_requested = True

    # ---------- 工具方法 ----------
    def _log(self, msg: str):
        """批量合并 + 节流刷新: 每 80ms 至多刷新一次 UI"""
        self._log_buffer.append(msg)
        if not self._log_flush_pending:
            self._log_flush_pending = True
            self.after(80, self._flush_log)      # 用 Frame 自己的 after, 不再依赖 root

    def _flush_log(self):
        self._log_flush_pending = False
        if not self._log_buffer:
            return
        batch = '\n'.join((m if m.endswith('\n') else m + '\n') for m in self._log_buffer)
        self._log_buffer.clear()
        try:
            self.log_text.insert('end', batch)
            ln = int(self.log_text.index('end-1c').split('.')[0])
            if ln > 2000:
                self.log_text.delete('1.0', f'{ln - 2000}.0')
            self.log_text.see('end')
        except tk.TclError:
            pass   # 控件已销毁 (面板被关掉), 丢弃这批日志即可

    def _clear_log(self):
        self.log_text.delete('1.0', 'end')

    def _open_output(self):
        out = self.当前输出目录()
        os.makedirs(out, exist_ok=True)
        try:
            if sys.platform.startswith('win'):
                os.startfile(os.path.abspath(out))
            elif sys.platform == 'darwin':
                os.system(f'open "{os.path.abspath(out)}"')
            else:
                os.system(f'xdg-open "{os.path.abspath(out)}"')
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录: {e}")

    def _set_status(self, s: str):
        self.after(0, lambda: self._safe(lambda: self.var_status.set(s)))

    def _update_progress(self, cur: int, total: int):
        def _do():
            self._safe(lambda: self.var_progress.set(100.0 * cur / max(1, total)))
            self._safe(lambda: self.var_progress_text.set(f"{cur} / {total}"))
        self.after(0, _do)

    def _add_result_row(self, r: dict):
        """插入结果行 (字段名照 拓扑训练器.topology_train_pipeline 的返回键, 与原版一致)"""
        def _do():
            def _ins():
                row = (
                    r['idx'],
                    f"{r['rows']}x{r['cols']}",
                    f"{r['n_nodes']:,}" + ('*' if r.get('was_partial') else ''),
                    str(r['beta_0']),
                    str(r['beta_1']),
                    str(r['euler_char']),
                    f"{r['algebraic_connectivity']:.3f}"
                    if isinstance(r.get('algebraic_connectivity'), (int, float)) else "-",
                    f"{r['modularity']:.3f}",
                    str(r['n_communities']),
                    str(r['tda_h0']),
                    str(r['tda_h1']),
                    f"{r['total_time']:.1f}s",
                    ("G" if r.get('resumed_gnn') else ".") + ("D" if r.get('resumed_tda_dl') else "."),
                )
                self.tree.insert('', 0, values=row)
                children = self.tree.get_children()
                if len(children) > 50:
                    self.tree.delete(children[-1])
            self._safe(_ins)
        self.after(0, _do)

    def _safe(self, fn):
        """面板可能已被销毁 (用户切走/关窗), 任何 TclError 都吞掉"""
        try:
            return fn()
        except tk.TclError:
            return None

    # ---------- 训练控制 ----------
    def _on_start(self):
        if self._running:
            return
        try:
            n = int(self.var_n.get())
            min_dim = int(self.var_min_dim.get())
            max_dim = int(self.var_max_dim.get())
            assert n > 0
            assert 2 <= min_dim <= max_dim <= 6
        except Exception:
            messagebox.showerror("参数错误",
                                 "请检查:\n布局数量>0\n2 ≤ min_dim ≤ max_dim ≤ 6")
            return

        self._running = True
        self._stop_requested = False
        self.btn_start.configure(state='disabled')
        self.btn_stop.configure(state='normal')
        self._results.clear()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _on_stop(self):
        if not self._running:
            return
        self._log("用户请求停止, 当前布局完成后将终止...")
        self._stop_requested = True
        self.btn_stop.configure(state='disabled')

    # ---------- 训练工作线程 ----------
    def _worker(self):
        n = int(self.var_n.get())
        min_dim = int(self.var_min_dim.get())
        max_dim = int(self.var_max_dim.get())
        gnn_epochs = int(self.var_gnn_epochs.get())
        max_nodes = int(self.var_max_nodes.get())
        seed = int(self.var_seed.get())
        resume = bool(self.var_resume.get())
        output = self.当前输出目录()
        cpu_limit_pct = max(10, min(95, int(self.var_cpu_limit.get())))

        os.makedirs(output, exist_ok=True)
        model_dir = os.path.join(output, '模型')
        os.makedirs(model_dir, exist_ok=True)

        cpu_limiter = CPULimiter(target_ratio=cpu_limit_pct / 100.0, check_interval=0.3)

        self._log("=" * 60)
        self._log(f"  开始拓扑训练 | {n} 布局 | 棋盘 {min_dim}x{min_dim}~{max_dim}x{max_dim}")
        self._log(f"  GNN epochs={gnn_epochs} | max_nodes={max_nodes} | seed={seed}")
        self._log(f"  CPU占用上限: {cpu_limit_pct}% (不跑满) | 底层线程数: {_n_cpu}/{os.cpu_count()}核")
        self._log(f"  自我迭代: {'启用' if resume else '禁用'} | 输出: {output}/")
        if resume:
            existing = []
            for fn in ['gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl']:
                if os.path.exists(os.path.join(model_dir, fn)):
                    existing.append(fn)
            self._log(f"  已有全局模型: {existing if existing else '无'}")
        self._log("=" * 60)
        self._set_status(f"训练中: 0/{n}")
        self._update_progress(0, n)

        gen = DynamicLayoutGenerator(max_nodes=max_nodes, min_dim=min_dim, max_dim=max_dim)
        t_start = time.time()
        failed = 0
        succeeded = 0

        for i in range(n):
            if self._stop_requested:
                self._log(f"\n停止信号已接收, 已完成 {i}/{n} 个布局。")
                break

            cpu_limiter.throttle()

            seed_i = seed + i * 1000
            layout = gen.generate(seed=seed_i)
            if layout is None:
                failed += 1
                continue
            try:
                blks = [Block(bid, coords) for bid, coords in layout.blocks.items()]
                if not State(blks, (layout.rows, layout.cols)).is_valid():
                    failed += 1
                    continue
            except Exception:
                failed += 1
                continue

            try:
                result = topology_train_pipeline(
                    layout, i, output,
                    gnn_epochs=gnn_epochs, resume=resume,
                    max_nodes=max_nodes, log_fn=self._log,
                    cpu_limiter=cpu_limiter,
                )
                if result:
                    self._results.append(result)
                    self._add_result_row(result)
                    succeeded += 1
                else:
                    failed += 1
            except Exception as e:
                import traceback
                self._log(f"  ✗ 布局{i}异常: {e}\n{traceback.format_exc()}")
                failed += 1

            # 每个布局完成后显式释放大对象 (模型权重已存盘, 这里只放运行时中间量)
            _release_memory()
            time.sleep(0.005)

            self._update_progress(min(i + 1, n), n)
            self._set_status(f"训练中: {min(i+1,n)}/{n}  成功{succeeded}  失败{failed}")

        total_t = time.time() - t_start
        self._log("\n" + "=" * 60)
        self._log("  训练结束汇总")
        self._log("=" * 60)
        self._log(f"  成功: {succeeded}/{n}   失败: {failed}")
        self._log(f"  总耗时: {total_t:.1f}s"
                  + (f"  平均: {total_t/max(1, succeeded):.1f}s/布局" if succeeded else ""))

        if self._results:
            total_nodes = sum(r['n_nodes'] for r in self._results)
            total_edges = sum(r['n_edges'] for r in self._results)
            partials = sum(1 for r in self._results if r.get('was_partial'))
            res_gnn = sum(1 for r in self._results if r.get('resumed_gnn'))
            res_dl = sum(1 for r in self._results if r.get('resumed_tda_dl'))
            size_counter = Counter((r['rows'], r['cols']) for r in self._results)
            b0_list = [r['beta_0'] for r in self._results if isinstance(r.get('beta_0'), int)]
            b1_list = [r['beta_1'] for r in self._results if isinstance(r.get('beta_1'), int)]
            l2_list = [r['algebraic_connectivity'] for r in self._results
                       if isinstance(r.get('algebraic_connectivity'), (int, float))]
            mods = [r['modularity'] for r in self._results]
            comms = [r['n_communities'] for r in self._results
                     if isinstance(r.get('n_communities'), int)]

            self._log(f"  总状态节点: {total_nodes:,}")
            self._log(f"  总边数: {total_edges:,}")
            self._log(f"  部分图(超限截断): {partials}/{len(self._results)} "
                      f"({100*partials/len(self._results):.0f}%)")
            self._log(f"  GNN 自我迭代: {res_gnn}/{len(self._results)}")
            self._log(f"  TDA-DL 自我迭代: {res_dl}/{len(self._results)}")
            self._log(f"  棋盘尺寸分布: {dict(size_counter)}")
            if b0_list:
                self._log(f"  β₀ 连通分支数: 范围 {min(b0_list)}-{max(b0_list)}, "
                          f"均值 {np.mean(b0_list):.1f}")
            if b1_list:
                self._log(f"  β₁ 独立环数: 范围 {min(b1_list)}-{max(b1_list)}, "
                          f"均值 {np.mean(b1_list):.1f}")
            if l2_list:
                self._log(f"  λ₂ 代数连通度 (Fiedler): 范围 {min(l2_list):.4f}-{max(l2_list):.4f}, "
                          f"均值 {np.mean(l2_list):.4f}")
            if mods:
                self._log(f"  模块度 Q: 范围 {min(mods):.3f}-{max(mods):.3f}, "
                          f"均值 {np.mean(mods):.3f}")
            if comms:
                self._log(f"  社区数: 范围 {min(comms)}-{max(comms)}, 均值 {np.mean(comms):.1f}")

            import json
            summary = {
                'n_puzzles': n, 'n_succeeded': succeeded, 'n_failed': failed,
                'total_time': round(total_t, 2),
                'min_dim': min_dim, 'max_dim': max_dim,
                'gnn_epochs': gnn_epochs, 'max_nodes': max_nodes,
                'resume_enabled': resume,
                'results': self._results,
            }
            sp = os.path.join(output, 'topology_summary.json')
            tmp_sp = sp + '.tmp'
            try:
                with open(tmp_sp, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                os.replace(tmp_sp, sp)
                self._log(f"\n  汇总已保存: {sp}")
            except OSError as e:
                self._log(f"\n  ⚠ 汇总保存失败: {e}")

        self._log(f"  输出目录: {os.path.abspath(output)}")
        self._log("=" * 60)
        self._set_status(f"完成 | 成功{succeeded} 失败{failed} | {total_t:.1f}s")
        self.after(0, self._finish)

    def _finish(self):
        self._running = False
        self._safe(lambda: self.btn_start.configure(state='normal'))
        self._safe(lambda: self.btn_stop.configure(state='disabled'))


def run_standalone(初始输出目录: str = '训练数据'):
    """独立窗口模式: 建一个 Tk 窗口, 把同一个面板放进去

    代码与主界面「训练」页共用同一份 TrainerPanel, 不再有第二套 UI 需要同步维护。
    """
    root = tk.Tk()
    root.title("华容道 拓扑学训练器 (独立版本 · 无RL)")
    root.geometry("1100x760")
    try:
        style = ttk.Style()
        if 'vista' in style.theme_names():
            style.theme_use('vista')
    except Exception:
        pass

    面板 = TrainerPanel(root, 独立模式=True, 初始输出目录=初始输出目录)
    面板.pack(fill='both', expand=True)

    def _on_close():
        面板.请求停止()      # 让训练线程有机会收尾 (守护线程, 不强杀)
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', _on_close)
    root.mainloop()
    return 面板

