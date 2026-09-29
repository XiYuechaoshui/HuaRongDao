# -*- coding: utf-8 -*-
"""冻结产物自检 (既能在源码模式手工跑, 也会被打进 exe 用于打包后验证)

为什么需要它:
  旧打包脚本第 6 步的判据只有"进程 12 秒后还在" —— 而 PyInstaller 窗口版一旦
  在导入期抛异常, 会弹一个 "Unhandled exception in script" 模态对话框, 进程**不会退出**,
  于是 2026-09-17 那份坏包被误判成"启动成功", 实际一打开就崩
  (excludes 去掉 unittest → matplotlib → pyparsing → pyparsing.testing → No module named 'unittest')。

  所以判据必须换成"窗口标题对"+"冻结包里的依赖真能用", 后者只能由 exe 自己来测:
  冻结后的 import 解析、_internal 的 DLL 搜索路径、布局哈希.py 的 frozen 分支
  (_BASE_DIR = exe 同级) 在源码模式下都复现不出来。

打印注意:
  打包配置是 console=False, sys.stdout/stderr 是 None, print 会直接抛 AttributeError。
  因此所有输出必须落盘成 JSON 报告, 由调用方(打包脚本)读出来展示。

用法:
  源码模式:  python 自检.py [报告路径]
  exe 模式 :  set HRD_SELFTEST=1 && dist\\华容道\\华容道.exe     (报告写到 exe 同级)

退出码: 0 = 全过; 2 = 有失败项; 3 = 自检脚本自身异常
"""
import json
import os
import sys
import time
import traceback

# 期望的主窗口标题 —— 打包脚本第 6 步用它做断言, 两处必须一致
期望窗口标题 = '华容道状态空间分析器'
报告文件名 = '自检报告.json'


def _基准目录() -> str:
    """frozen 模式下与 布局哈希.py 完全一致的基准目录 (exe 同级)"""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _临时目录():
    """在工程/exe 同级建临时目录; 建不出来返回 None

    为什么不用 %TEMP%: 受限环境 (沙箱/受控令牌) 下系统临时目录与工程目录都可能只允许建目录、
    不允许在里面创建文件, 这时宁可**标注跳过**也不要把环境问题记成 exe 的缺陷。
    """
    import tempfile
    try:
        return tempfile.mkdtemp(prefix='_自检临时_', dir=_基准目录())
    except OSError:
        return None


def _能写文件吗(目录: str) -> bool:
    if not 目录:
        return False
    try:
        p = os.path.join(目录, '_写权限探测.tmp')
        with open(p, 'wb') as f:
            f.write(b'x')
        os.remove(p)
        return True
    except OSError:
        return False


def _清理临时目录(路径: str) -> None:
    if not 路径:
        return
    import shutil
    try:
        shutil.rmtree(路径, ignore_errors=True)
    except Exception:  # noqa: BLE001
        pass


def _run_check(报告: dict, 名称: str, fn) -> None:
    """跑一项检查; 任何异常都记成失败, 不中断后续检查"""
    t0 = time.time()
    item = {'名称': 名称}
    try:
        结果 = fn()
        item['通过'] = True
        item['详情'] = 结果 if isinstance(结果, str) else json.dumps(
            结果, ensure_ascii=False, default=str)
    except BaseException as e:  # noqa: BLE001 - SystemExit 也要抓住, 否则自检中途静默退出
        item['通过'] = False
        item['详情'] = '{0}: {1}'.format(type(e).__name__, e)
        item['回溯'] = traceback.format_exc()[-1500:]
    item['耗时秒'] = round(time.time() - t0, 2)
    报告['检查'].append(item)


# ============ 各项检查 ============

def _检查_冻结布局():
    if not getattr(sys, 'frozen', False):
        raise AssertionError('sys.frozen 未设置: 这份自检不是被打包后的 exe 运行的')
    exe目录 = os.path.dirname(os.path.abspath(sys.executable))
    internal = os.path.join(exe目录, '_internal')
    torch_lib = os.path.join(internal, 'torch', 'lib')
    if not os.path.isdir(internal):
        raise AssertionError('缺少 _internal: ' + internal)
    if not os.path.isdir(torch_lib):
        raise AssertionError('缺少 torch\\lib: ' + torch_lib)
    return {
        'exe': sys.executable,
        'meipass': getattr(sys, '_MEIPASS', ''),
        '无控制台': sys.stdout is None and sys.stderr is None,
        'torch库内DLL': {n: os.path.exists(os.path.join(torch_lib, n))
                        for n in ('c10.dll', 'torch_cpu.dll', 'torch_python.dll', 'libiomp5md.dll')},
    }


_工程模块 = ['字体工具', '数据模型', '代码生成器', '导入器', '状态空间', '拓扑分析',
             '持久同调', '社区检测', '可视化器', '布局编辑器', '布局哈希',
             '缓存管理器', '缓存面板', 'AI模型', '自动训练', '拓扑训练器', '主窗口']


def _检查_模块导入():
    import importlib
    失败 = {}
    for m in _工程模块:
        try:
            importlib.import_module(m)
        except BaseException as e:  # noqa: BLE001
            失败[m] = '{0}: {1}'.format(type(e).__name__, e)
    if 失败:
        raise AssertionError('导入失败: ' + json.dumps(失败, ensure_ascii=False))
    return '全部 {0} 个工程模块导入成功'.format(len(_工程模块))


def _检查_unittest可用():
    """专门盯 2026-09-17 那次事故的回归: pyparsing -> unittest 是启动硬依赖"""
    import unittest
    import unittest.mock  # noqa: F401
    import pyparsing
    import matplotlib
    if not hasattr(pyparsing, 'testing'):
        raise AssertionError('pyparsing.testing 未随包 (matplotlib 会导入它)')
    return 'unittest {0} / pyparsing {1} / matplotlib {2}'.format(
        getattr(unittest, '__name__', 'unittest'), pyparsing.__version__, matplotlib.__version__)


def _检查_torch():
    import torch
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    标量 = float(torch.sigmoid(x).sum().item())
    稀疏 = 'ok'
    try:
        i = torch.tensor([[0, 1], [1, 0]])
        v = torch.tensor([1.0, 2.0])
        torch.sparse.mm(torch.sparse_coo_tensor(i, v, (2, 2)), x)
    except BaseException as e:  # noqa: BLE001
        稀疏 = '{0}: {1}'.format(type(e).__name__, e)
    网络 = 'ok'
    try:
        import torch.nn as tnn
        net = tnn.Sequential(tnn.Linear(4, 8), tnn.ReLU(), tnn.Linear(8, 2))
        net(torch.randn(3, 4))
        torch.optim.Adam(net.parameters(), lr=1e-3)
    except BaseException as e:  # noqa: BLE001
        网络 = '{0}: {1}'.format(type(e).__name__, e)
    # 存盘往返: 用工程同级临时目录 (不用 %TEMP%, 见 _临时目录 注释),
    # 同时验证 torch.save / torch.load 两个方向; 环境不让写盘就标注跳过而不是记失败
    d = _临时目录()
    try:
        if not _能写文件吗(d):
            往返 = '跳过 (当前环境不允许写文件, 非 exe 缺陷)'
        else:
            p = os.path.join(d, '_自检往返.pt')
            torch.save({'w': torch.zeros(3)}, p)
            back = torch.load(p, map_location='cpu', weights_only=False)
            往返 = 'ok ({0})'.format(list(back))
    except BaseException as e:  # noqa: BLE001
        往返 = '{0}: {1}'.format(type(e).__name__, e)
    finally:
        _清理临时目录(d)
    return {'torch': torch.__version__, 'sigmoid求和': 标量, '稀疏矩阵乘': 稀疏,
            'nn与优化器': 网络, '存盘往返': 往返, '线程数': torch.get_num_threads()}


def _检查_ML流水线():
    """跑一遍工程自己的 AI 训练链路 —— 这是"用户点 AI 分析不会炸"的真实证据

    为什么不能只测 `import torch`: 2026-09-17 那次打包, import torch 挂在
    torch.distributed / torchgen 这类**传递性**依赖上, 而顶层 API 看起来完全正常。
    这里用一个 12 节点的合成环图把 GNN 前向/反向/优化器、TopologyFeatureML 特征、
    TDADeepLearning 自编码器、以及模型存盘往返全部走一遍, 只花几秒。
    """
    import numpy as np
    import networkx as nx
    from AI模型 import SimpleGNN, TopologyFeatureML, TDADeepLearning

    图 = nx.cycle_graph(12)
    for i in 图.nodes():
        图.nodes[i]['degree'] = float(图.degree(i))

    gnn = SimpleGNN(input_dim=1, hidden_dim=8, output_dim=4)
    gnn.train(图, epochs=3, lr=0.01)
    嵌入 = gnn.get_embeddings(图)
    if 嵌入 is None or len(嵌入) != 图.number_of_nodes():
        raise AssertionError('GNN 嵌入形状异常: {0}'.format(getattr(嵌入, 'shape', None)))

    特征 = TopologyFeatureML().extract_graph_features(图)
    if 特征.shape[0] != 图.number_of_nodes():
        raise AssertionError('特征矩阵行数异常: {0}'.format(特征.shape))

    tda = TDADeepLearning()
    tda.train_autoencoder(np.random.RandomState(0).rand(24, 6), epochs=3)

    # 模型存盘往返 (工程里 保存模型/加载模型 按钮走的就是这两个方法)
    d = _临时目录()
    try:
        if not _能写文件吗(d):
            存盘结论 = '跳过 (当前环境不允许写文件, 非 exe 缺陷)'
        else:
            p = os.path.join(d, '_自检_gnn.pt')
            gnn.save_model(p)
            gnn2 = SimpleGNN(input_dim=1, hidden_dim=8, output_dim=4)
            if not gnn2.load_model(p):
                raise AssertionError('SimpleGNN.load_model 返回 False')
            存盘结论 = '通过'
    finally:
        _清理临时目录(d)

    return {'图': '{0} 节点 / {1} 边'.format(图.number_of_nodes(), 图.number_of_edges()),
            'GNN嵌入': str(嵌入.shape), '特征矩阵': str(特征.shape),
            'TDA自编码器': '3 epochs 通过', '模型存盘往返': 存盘结论}


def _检查_训练数据():
    """校验训练数据目录与模型来源

    判据的取舍 (2026-09-19 修正):
      必须存在 —— 基准目录解析到 exe 同级、模型目录存在、scan_model_sources 认得出「训练数据」、
                  **全局模型 (gnn_global.pt / tda_ae_global.pt / ml_global.pkl) 齐全**。
                 全局模型是跨布局迁移用的, 缺了 AI 分析会退化成每次从零训练。
      不要求   —— 布局级存档 (graph_<hash>.pkl 等)。
                 它们是"用户自己算过的那个布局"的缓存, 换台机器/换个布局自然没有,
                 缺了只是要重算一次, 不是故障。发布包特意不带这些 (实测某次交付里
                 三个布局图占了 200MB / 整包 26%), 所以这里不能把它们当硬指标。
    """
    import 布局哈希 as lh
    exe目录 = os.path.dirname(os.path.abspath(sys.executable))
    源码目录 = os.path.dirname(os.path.abspath(__file__))
    模型目录 = lh.get_model_dir()
    来源 = lh.scan_model_sources()
    问题 = []
    # frozen 模式下 布局哈希.py 必须把基准目录解析到 exe 同级, 否则训练数据会找错地方
    期望基准 = exe目录 if getattr(sys, 'frozen', False) else 源码目录
    if os.path.normcase(lh._BASE_DIR) != os.path.normcase(期望基准):
        问题.append('_BASE_DIR 不是 {0} 同级: {1}'.format(
            'exe' if getattr(sys, 'frozen', False) else '源码', lh._BASE_DIR))
    if not os.path.isdir(模型目录):
        问题.append('模型目录不存在: {0}'.format(模型目录))
    if '训练数据' not in 来源:
        问题.append('scan_model_sources 未含 训练数据: {0}'.format(来源))

    文件 = sorted(os.listdir(模型目录)) if os.path.isdir(模型目录) else []
    全局模型 = [f for f in 文件 if f in ('gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl')]
    缺失全局 = [f for f in ('gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl')
                if f not in 文件]
    if 缺失全局:
        问题.append('缺少全局模型 (发布包应包含它们): {0}'.format(缺失全局))
    # 注意: 不能只用 startswith 判断 —— 三个全局模型自己也以 gnn_/tda_ae_/ml_ 开头,
    # 早先的写法会把它们算成"布局级存档", 于是一份**全新安装**被报成"有 3 个存档",
    # 反而让人以为包里带了布局缓存。这里显式排除全局模型名。
    布局存档 = [f for f in 文件 if f.startswith(('graph_', 'gnn_', 'tda_ae_', 'ml_'))
                and f not in ('gnn_global.pt', 'tda_ae_global.pt', 'ml_global.pkl')]
    if 问题:
        raise AssertionError('; '.join(问题))
    return {'基准目录': lh._BASE_DIR, '模型目录': 模型目录, '可用来源': 来源,
            '全局模型': 全局模型,
            '布局级存档': '%d 个 (%s)' % (
                len(布局存档),
                '用户算过的布局缓存, 可选' if 布局存档 else '无 —— 全新安装属正常, 首次分析会现算'),
            '模型文件': [f for f in 文件 if f.endswith(('.pkl', '.pt'))][:14]}


def _检查_模型载入():
    """真的把 pickle / torch.load 跑一遍, 而不是只看文件在不在"""
    import pickle
    import torch
    import 布局哈希 as lh
    结果 = {}
    for 键, 路径 in lh.global_model_paths().items():
        if not os.path.exists(路径):
            结果[键] = '缺失(跳过)'
            continue
        if 路径.endswith('.pt'):
            对象 = torch.load(路径, map_location='cpu', weights_only=False)
            结果[键] = '已载入 {0} 项'.format(len(对象) if hasattr(对象, '__len__') else type(对象).__name__)
        else:
            with open(路径, 'rb') as f:
                对象 = pickle.load(f)
            结果[键] = '已载入 {0}'.format(type(对象).__name__)
    # 取最大的布局图, 验证大文件也能读 (训练数据里有 100 MB 级的)
    模型目录 = lh.get_model_dir()
    最大, 大小 = None, 0
    for f in os.listdir(模型目录):
        if f.startswith('graph_') and f.endswith('.pkl'):
            大小_ = os.path.getsize(os.path.join(模型目录, f))
            if 大小_ > 大小:
                最大, 大小 = f, 大小_
    if 最大:
        t0 = time.time()
        with open(os.path.join(模型目录, 最大), 'rb') as f:
            图 = pickle.load(f)
        结果['最大布局图'] = '{0} ({1} MB) -> {2}, {3}s'.format(
            最大, round(大小 / 1048576, 1), type(图).__name__, round(time.time() - t0, 1))
    return 结果


def _检查_GUI():
    """真建窗口并核对标题, 再验四个视图入口都能用 —— 这是"用户打开它能看到界面"的最直接证据"""
    import tkinter as tk
    from 主窗口 import StateSpaceAnalyzerGUI
    t0 = time.time()
    root = tk.Tk()
    root.withdraw()
    try:
        应用 = StateSpaceAnalyzerGUI(root)
        root.update_idletasks()
        root.update()
        标题 = root.title()
        if 标题 != 期望窗口标题:
            raise AssertionError('主窗口标题不符: 期望 {0!r}, 实际 {1!r}'.format(期望窗口标题, 标题))

        # 「视图」菜单必须四个入口齐全 —— 曾经漏掉过「AI增强图」,
        # 而分析报告的文字写的是"点击顶栏「AI增强图」", 用户照做找不到入口
        菜单 = root.nametowidget(root.cget('menu'))
        视图项 = []
        for i in range(菜单.index('end') + 1):
            if 菜单.type(i) == 'cascade' and 菜单.entrycget(i, 'label') == '视图':
                子 = root.nametowidget(菜单.entrycget(i, 'menu'))
                for j in range(子.index('end') + 1):
                    if 子.type(j) == 'command':
                        视图项.append(子.entrycget(j, 'label'))
        缺失 = [v for v in ('3D 状态图', '谱分布', '持久条形码', 'AI增强图') if v not in 视图项]
        if 缺失:
            raise AssertionError('「视图」菜单缺少入口: {0} (现有: {1})'.format(缺失, 视图项))

        # F1~F4 必须都绑上
        绑定 = set(root.bind())
        缺键 = [k for k in ('<Key-F1>', '<Key-F2>', '<Key-F3>', '<Key-F4>') if k not in 绑定]
        if 缺键:
            raise AssertionError('缺少快捷键绑定: {0}'.format(缺键))

        # 「路径搜索」「双棋盘寻路」已按用户要求整体移除 (2026-09-17):
        # 菜单入口 / 左面板按钮 / 两个方法 / current_path 状态一起清掉。这里做回归断言, 防止又被加回来。
        残留方法 = [n for n in ('_show_path_search', '_show_dual_board_search')
                    if hasattr(应用, n)]
        if hasattr(应用, 'current_path'):
            残留方法.append('current_path')
        # RL (Q-learning 求解器) 已整体移除: 它的评测指标是伪信号 (目标判定不看初始布局,
        # solve() 会返回长度为 1 的"解"; 7 万节点上图 Q 表无法收敛)。
        # 这里做回归断言 —— 该功能当初是靠"截断到文件尾"切掉的, 顺手把紧随其后的
        # _show_ai_figure 一起删了, 正是 AI增强图整条链断掉的根因。
        import AI模型 as _ai
        for 名 in ('RLSolver', '_run_ai_rl', 'rl_qtable'):
            if hasattr(_ai, 名) or hasattr(应用, 名):
                残留方法.append(名)
        if 残留方法:
            raise AssertionError('已移除的功能又出现了: {0}'.format(残留方法))
        分析菜单 = []
        for i in range(菜单.index('end') + 1):
            if 菜单.type(i) == 'cascade' and 菜单.entrycget(i, 'label') == '分析':
                子分析 = root.nametowidget(菜单.entrycget(i, 'menu'))
                for j in range(子分析.index('end') + 1):
                    if 子分析.type(j) == 'command':
                        分析菜单.append(子分析.entrycget(j, 'label'))
        旧入口 = [t for t in ('路径搜索...', '双棋盘寻路...') if t in 分析菜单]
        if 旧入口:
            raise AssertionError('「分析」菜单仍残留已移除入口: {0}'.format(旧入口))

        # 左侧页签必须齐全 (控制/绘制/缓存/训练/模型) —— 2026-09-17 新增了「训练」「模型」两页:
        # 训练器面板从独立窗口搬进主界面, 模型库的新建/管理也在这里。
        页签 = [应用._left_nb.tab(i, 'text') for i in range(应用._left_nb.index('end'))]
        for 必需 in ('控制', '绘制', '缓存', '训练', '模型'):
            if 必需 not in 页签:
                raise AssertionError('左侧页签缺少「%s」: 现有 %s' % (必需, 页签))
        if getattr(应用, 'trainer_tab', None) is None or \
                应用.trainer_tab.__class__.__name__ != 'TrainerPanel':
            raise AssertionError('「训练」页没接上 TrainerPanel')
        if getattr(应用, 'model_tab', None) is None or \
                应用.model_tab.__class__.__name__ != 'ModelManagerPanel':
            raise AssertionError('「模型」页没接上 ModelManagerPanel')
        # 三处模型来源联动: 训练页输出目录必须等于当前模型来源
        if 应用.trainer_tab.当前输出目录() != (应用.var_model_source.get() or '训练数据'):
            raise AssertionError('训练页输出目录与模型来源不一致: %r vs %r'
                                 % (应用.trainer_tab.当前输出目录(), 应用.var_model_source.get()))

        # 三个分组按钮必须等宽 —— 2026-09-17 用户反馈"视图比布局/分析窄"。
        # 根因: 左侧面板只有 248px, 而 pack(expand=True,fill=X) 只在有剩余空间时才均分,
        # 放不下时最后打包的「视图」被压到 48px。已改为 grid + uniform 强制等宽。
        分组宽 = [b.winfo_width() for b in 应用._group_buttons.values()]
        if len(分组宽) != 3:
            raise AssertionError('分组按钮数量异常: %d' % len(分组宽))
        if max(分组宽) - min(分组宽) > 1:
            raise AssertionError('分组按钮宽度不一致: {0}'.format(分组宽))

        # 切到「缓存」页必须自动加宽左侧面板, 切走要还原 —— 缓存页表格收窄后仍要 466px、
        # 顶部那排也要约 280px, 而左侧默认只有约 333px (用户反馈"清空全部"被裁掉)
        # 注意: withdraw 状态下窗口几何是 1px, sashpos 恒为 0, 必须先把窗口真实映射出来,
        #       否则这条断言会误报 (实测 "0 -> 0")。测完立刻 withdraw 回去。
        sash变化 = None
        列宽信息 = '(未测)'
        if hasattr(应用, '_main_paned') and hasattr(应用, '_left_nb'):
            paned, nb = 应用._main_paned, 应用._left_nb
            root.deiconify()
            root.update()
            root.update_idletasks()
            try:
                sash原 = paned.sashpos(0)
                nb.select(2)
                for _ in range(12):
                    root.update()
                    root.update_idletasks()
                sash缓存 = paned.sashpos(0)
                nb.select(0)
                for _ in range(12):
                    root.update()
                    root.update_idletasks()
                sash还原 = paned.sashpos(0)
                if nb.index(nb.select()) != 0:
                    raise AssertionError('未能切回「控制」页')
                if not (sash缓存 > sash原):
                    raise AssertionError('切到缓存页没有加宽左侧面板: {0} -> {1}'.format(sash原, sash缓存))
                if abs(sash还原 - sash原) > 4:
                    raise AssertionError('离开缓存页未还原宽度: 原 {0}, 现在 {1}'.format(sash原, sash还原))
                sash变化 = '{0} -> 缓存页 {1} -> 还原 {2}'.format(sash原, sash缓存, sash还原)

                # 缓存表列宽必须按内容收窄过 —— 用户反馈"名称再窄一点":
                # 名称列原 180px 而内容只要 43px; 收窄后整表 680 -> 466px, 加宽幅度也随之降低。
                表 = getattr(应用.cache_tab, 'tree', None)
                列宽信息 = '(无表格)'
                if 表 is not None:
                    名称宽 = 表.column('name', 'width')
                    列宽合计 = sum(表.column(c, 'width') for c in 表['columns'])
                    if 名称宽 > 96:
                        raise AssertionError('名称列仍过宽: %d px (应 <= 96)' % 名称宽)
                    if 列宽合计 > 560:
                        raise AssertionError('缓存表总宽仍过大: %d px (应 <= 560)' % 列宽合计)
                    列宽信息 = '名称 %d px, 合计 %d px' % (名称宽, 列宽合计)
            finally:
                root.withdraw()

        # 真的切一遍 AI 视图 (空图上应显示占位提示而不是抛异常)
        应用._switch_view('ai')
        root.update_idletasks()
        root.update()
        if 应用._current_view != 'ai':
            raise AssertionError('切到 ai 视图后 _current_view={0!r}'.format(应用._current_view))
        应用._switch_view('3d')

        # ---- AI 增强图这条链的回归断言 (2026-09-17 补) ----
        # 背景: 「AI增强图」曾经整条链是断的 ——
        #   1) _show_ai_figure 在切除 RL 残留时被"截断到文件尾"误删 → _switch_view('ai') 直接 AttributeError;
        #   2) 命中完整结果缓存时没有重建 figure (figure 不可 JSON 序列化, 存盘时被剔除) → 空画布;
        #   3) _build_ai_figure 里 `ai.get('gnn', {})` 在值为 None 时拿到 None, `in None` 抛 TypeError。
        # 这三条任何一条复发, 用户看到的都是"AI增强图没生效"。
        for 名 in ('_show_ai_figure', '_build_ai_figure', '_重建AI图并记录', '_聚类取色'):
            if not hasattr(应用, 名):
                raise AssertionError('AI增强图链路缺少方法: %s' % 名)
        # 造一个最小 figure 走一遍真实显示路径, 确认 canvas 上真有图像对象
        from matplotlib.figure import Figure as _Figure
        假图 = _Figure(figsize=(2, 2))
        假图.add_subplot(111).plot([0, 1], [1, 0])
        try:
            应用._ai_enhancement = {'figure': 假图}
            应用._last_display_key = None
            应用._switch_view('ai')
            root.update_idletasks()
            root.update()
            有图 = any(getattr(a, 'images', None) for a in 应用.fig.axes)
            if not 有图:
                raise AssertionError('AI 视图没有渲染出图像 (canvas 上找不到 imshow 对象)')
        finally:
            应用._ai_enhancement = {}
            应用._last_display_key = None
            应用._switch_view('3d')
            root.update_idletasks()
        # 聚类取色兜底: 键对不上时应返回 None (单色) 而不是静默全 0
        if 应用._聚类取色({'不存在的键': 1}, 应用.generator.graph if 应用.generator.graph else
                          __import__('networkx').cycle_graph(5)) == [0, 0, 0, 0, 0]:
            raise AssertionError('_聚类取色 键不匹配时未回退为单色')

        return {'标题': 标题, '构建秒': round(time.time() - t0, 2),
                '顶层子控件': len(root.winfo_children()), 'tk': root.tk.call('info', 'patchlevel'),
                '类': 应用.__class__.__name__, '视图菜单': 视图项, '分析菜单': 分析菜单,
                '分组按钮宽度': 分组宽, '缓存页加宽': sash变化, '缓存表列宽': 列宽信息,
                '左侧页签': 页签, '模型来源联动': '训练页输出目录 ' + 应用.trainer_tab.当前输出目录(),
                'AI增强图链路': '渲染方法齐全 + 视图可显示图像',
                '视图快捷键': 'F1~F4 已绑定', 'AI视图切换': '通过', '已移除功能': '路径搜索/双棋盘寻路 无残留'}
    finally:
        try:
            root.destroy()
        except Exception:  # noqa: BLE001
            pass


def _相对明度(hex色: str) -> float:
    """WCAG 相对明度"""
    h = hex色.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def _对比度(a: str, b: str) -> float:
    la, lb = _相对明度(a), _相对明度(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _检查_主题对比度():
    """配色的可读性门槛 —— 主题是照参考图取色后"实用化"改造的, 别改回去

    参考图《汐》那套配色很仙, 但它是**极低对比**的: 实测次要文字 #8A87A8 在白底上
    只有 3.44:1 (正文标准 4.5), 主色 #8E93D8 配白字只有 2.87:1。
    所以本工程把"浅紫蓝"限定在容器装饰层, 承载文字/图形的一律加深到通过对比度。
    这条检查保证以后调色时不会悄悄把可读性调没。
    """
    from 主题 import Theme
    项 = [
        ('正文 TEXT/PANEL', Theme.TEXT, Theme.PANEL, 4.5),
        ('正文 TEXT/BG', Theme.TEXT, Theme.BG, 4.5),
        ('次要 MUTED/PANEL', Theme.MUTED, Theme.PANEL, 4.5),
        ('次要 MUTED/PANEL_ALT', Theme.MUTED, Theme.PANEL_ALT, 4.5),
        ('标题 ACCENT_TEXT/PANEL', Theme.ACCENT_TEXT, Theme.PANEL, 4.5),
        ('主按钮白字/ACCENT', '#FFFFFF', Theme.ACCENT, 4.5),
        ('危险按钮白字/DANGER', '#FFFFFF', Theme.DANGER, 4.5),
        ('表格选中白字/ACCENT', '#FFFFFF', Theme.ACCENT, 4.5),
        ('禁用 DISABLED/PANEL (仅非文本)', Theme.DISABLED, Theme.PANEL, 3.0),
        ('图表主色 ACCENT/PANEL (非文本)', Theme.ACCENT, Theme.PANEL, 3.0),
        ('图表成功色 SUCCESS/PANEL (非文本)', Theme.SUCCESS, Theme.PANEL, 3.0),
    ]
    结果 = {}
    失败 = []
    for 名, 前, 后, 门槛 in 项:
        r = _对比度(前, 后)
        结果[名] = '%.2f' % r
        if r < 门槛:
            失败.append('%s = %.2f (需 >= %.1f)' % (名, r, 门槛))
    if 失败:
        raise AssertionError('对比度不足: ' + '; '.join(失败))
    # 3D 深度色阶必须明度严格单调 —— 这是"色盲也能靠亮度区分深度"的前提
    亮度 = [_相对明度(c) for c in Theme.深度色阶]
    if any(亮度[i] >= 亮度[i + 1] for i in range(len(亮度) - 1)):
        raise AssertionError('深度色阶明度非单调: %s' % ['%.3f' % v for v in 亮度])
    结果['深度色阶明度跨度'] = '%.3f -> %.3f' % (亮度[0], 亮度[-1])
    return 结果


def run_all(报告路径: str = None) -> dict:
    """跑完整自检; 返回报告字典 (同时按需写盘)"""
    报告 = {'开始时间': time.strftime('%Y-%m-%d %H:%M:%S'),
            'frozen': bool(getattr(sys, 'frozen', False)),
            'exe': sys.executable, '检查': []}
    _run_check(报告, '冻结布局', _检查_冻结布局)
    _run_check(报告, '工程模块导入', _检查_模块导入)
    _run_check(报告, 'unittest/pyparsing 回归', _检查_unittest可用)
    _run_check(报告, 'torch 计算', _检查_torch)
    _run_check(报告, 'AI 训练流水线', _检查_ML流水线)
    _run_check(报告, '训练数据发现', _检查_训练数据)
    _run_check(报告, '模型文件载入', _检查_模型载入)
    _run_check(报告, 'GUI 主窗口', _检查_GUI)
    _run_check(报告, '主题对比度', _检查_主题对比度)
    报告['通过数'] = sum(1 for c in 报告['检查'] if c['通过'])
    报告['失败数'] = sum(1 for c in 报告['检查'] if not c['通过'])
    报告['失败项'] = [c['名称'] for c in 报告['检查'] if not c['通过']]

    if 报告路径 is None:
        报告路径 = os.path.join(_基准目录(), 报告文件名)
    try:
        with open(报告路径, 'w', encoding='utf-8') as f:
            json.dump(报告, f, ensure_ascii=False, indent=2)
        报告['报告路径'] = 报告路径
    except OSError as e:
        报告['报告路径'] = '写盘失败: {0}'.format(e)
    return 报告


def main(报告路径: str = None) -> int:
    报告 = run_all(报告路径)
    # 窗口版 exe 的 sys.stderr 是 None, 必须防御性判断
    流 = sys.stderr
    if 流 is not None:
        for c in 报告['检查']:
            流.write('[{0}] {1}: {2}\n'.format('通过' if c['通过'] else '失败', c['名称'], c['详情']))
        流.write('通过 {0} 项, 失败 {1} 项 -> {2}\n'.format(
            报告['通过数'], 报告['失败数'], 报告.get('报告路径', '')))
    return 0 if 报告['失败数'] == 0 else 2


if __name__ == '__main__':
    try:
        sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
    except SystemExit:
        raise
    except BaseException:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(3)
