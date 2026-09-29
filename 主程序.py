"""华容道状态空间分析器 - 主入口

模块化架构:
  字体工具.py     - 中文字体探测与应用
  数据模型.py     - Block/State/LayoutData 核心数据结构
  代码生成器.py   - 布局代码生成器
  导入器.py       - 布局导入器 (.py/.json/剪贴板)
  状态空间.py     - 状态空间 BFS 生成器
  拓扑分析.py     - 代数拓扑分析 (谱/同调/中心性)
  持久同调.py     - 拓扑数据分析 (持久同调)
  社区检测.py     - 社区检测 (Louvain)
  可视化器.py     - 3D/2D 可视化器
  布局编辑器.py   - 布局编辑器 GUI
  主窗口.py       - 主窗口 GUI
"""
import os
import sys
import ctypes

# 设置 Windows AppUserModel ID (让任务栏正确显示自定义图标)
try:
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('huarongdao.app.v1')
except (AttributeError, OSError):
    pass

import tkinter as tk
from tkinter import ttk

# ---- 打包自检入口 (普通启动行为完全不变; 详见 自检.py 与 打包exe.ps1 第 7 步) ----
# 设 HRD_SELFTEST=1 时只跑 自检.py 的冻结检查, 把 JSON 报告写到 exe 同级目录并退出,
# 不开主窗口。打包脚本第 7 步靠它拿到"包内 torch / 训练数据 / GUI 真能用"的证据 ——
# 窗口版 exe 没有 stdout, 只能靠落盘报告。
# 位置刻意放在 import tkinter 之后、from 主窗口 import ... 之前:
#   既让自检能测到 tkinter 与 matplotlib(经 主窗口 传入) 的导入链,
#   又保证普通启动多出来的开销只有一次 os.environ 查询。
if os.environ.get('HRD_SELFTEST'):
    try:
        import 自检
        sys.exit(自检.main())
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001 - 自检自身出错也不能把主程序拖崩
        import traceback
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(sys.executable)),
                                   '自检_启动失败.txt'), 'w', encoding='utf-8') as _f:
                _f.write(traceback.format_exc())
        except OSError:
            pass
        sys.exit(3)

from 主窗口 import StateSpaceAnalyzerGUI


def _set_window_icon(root):
    """为窗口设置图标 (exe 模式和源码模式都支持)"""
    _icon_candidates = []
    if getattr(sys, 'frozen', False):
        # exe 模式: 优先 _MEIPASS, 再找 exe 同级目录
        _base = getattr(sys, '_MEIPASS', '') or os.path.dirname(sys.executable)
        _icon_candidates.append(os.path.join(_base, 'app.ico'))
        _icon_candidates.append(os.path.join(os.path.dirname(sys.executable), 'app.ico'))
    else:
        _icon_candidates.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app.ico'))
    for _p in _icon_candidates:
        if os.path.isfile(_p):
            try:
                root.iconbitmap(default=_p)
                return
            except tk.TclError:
                continue


def main():
    root = tk.Tk()
    _set_window_icon(root)
    # 使用更现代的 ttk 主题
    try:
        style = ttk.Style()
        if 'clam' in style.theme_names():
            style.theme_use('clam')
    except Exception:
        pass

    app = StateSpaceAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
