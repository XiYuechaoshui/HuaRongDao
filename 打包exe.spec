# -*- mode: python ; coding: utf-8 -*-
"""华容道 发布版打包配置 (PyInstaller)

使用:
    在 D:\\XIYueの奇妙冒险\\华容道_打包 下运行  打包exe.ps1
    或手动:  D:\\XIYueの奇妙冒险\\华容道\\.venv\\Scripts\\python.exe -m PyInstaller 打包exe.spec ^
                 --distpath dist --workpath build

设计说明 (2026-09 修订):
  1. 训练数据**不打进包内**。原先 datas 里带 ('训练数据', '训练数据'), PyInstaller 6.x
     会把它放进 _internal\\训练数据, 而 布局哈希.py 在 frozen 模式下用的是
     dirname(sys.executable)\\训练数据 —— 即 exe 同级那份。结果包里有两份完全相同的数据,
     而 _internal 那份永远不会被读取。现在只由 打包exe.ps1 复制到 exe 同级。
  2. torch 子模块按"实际用到的 API"裁剪。工程对 torch 的使用只有:
     torch.tensor/zeros/cat/randint/log/sigmoid/sparse_coo_tensor/sparse.mm、
     torch.nn(+functional)、torch.optim.Adam、torch.no_grad、torch.load/save、
     torch.cuda.*、torch.backends.cudnn.*、torch.set_num_threads、torch.float32/long。
     torch_geometric / gudhi / ripser 等根本没安装也没被 import。
     注意: 真正的大头是 torch\\lib\\torch_cpu.dll (约 291 MB), 它**无法裁剪**,
     所以裁剪只减小体积与文件数, 不要期待 dist 体积大幅下降。
     若裁剪后运行报 ModuleNotFoundError, 把对应项从 excludes 里去掉即可。
  3. runtime_hooks 用工程目录自己的 rthook_torch.py (不再跨目录引用外部路径)。
  4. **excludes 里的每一项都必须以"打包后 exe 真能启动"来验证, 不能只看 import 分析。**
     2026-09-17 事故: excludes 里的 'unittest' 让 exe 启动即崩 (matplotlib -> pyparsing
     -> pyparsing.testing -> import unittest), 而旧的启动测试只看"进程还活着"就报了通过,
     因为 PyInstaller 窗口版的报错对话框会一直吊着进程。现在的守门有两道:
       打包exe.ps1 第 6 步断言主窗口标题 == 华容道状态空间分析器
       打包exe.ps1 第 7 步用 exe 跑包内自检 (自检.py, HRD_SELFTEST=1) 验 torch/模型/GUI
"""
import os

block_cipher = None

# 以本 spec 所在目录为工程目录, 便于整体搬移
proj_dir = os.path.dirname(os.path.abspath(SPEC))

datas = [
    # 只带窗口图标; 训练数据由打包脚本复制到 exe 同级 (见文件头说明 1)
    (os.path.join(proj_dir, 'app.ico'), '.'),
]

hiddenimports = [
    'sklearn.cluster',
    'sklearn.cluster._kmeans',
    'sklearn.metrics.pairwise',
    'sklearn.preprocessing',
    'sklearn.decomposition',
    'sklearn.ensemble',
    'sklearn.manifold',
    'PIL._tkinter_finder',
    'matplotlib.backends.backend_tkagg',
    # unittest 必须显式声明: 见下面 excludes 里的说明 (pyparsing -> unittest 是启动硬依赖)
    'unittest',
    'unittest.mock',
    # 自检.py 在 主程序.py 里是 if 块内 import, 静态分析可能漏掉, 显式声明保证进包
    '自检',
    # 这两个模块在 主窗口.py 里是懒加载 (按钮回调里 import), 静态分析同样收不到;
    # 少了它们的后果不是启动崩溃, 而是用户点「自动训练 / 拓扑训练器」时才 ModuleNotFoundError
    '自动训练',
    '拓扑训练器',
]

# 纯 Python 子树裁剪 —— 这份清单是**实测**出来的, 不是猜的 (2026-09-17 重定)
#
# 教训: 原来的清单按"工程对 torch 的用法"列了 11 项, 其中 7 项是 torch 的**传递性硬依赖**,
# 结果 exe 一 import torch 就 ModuleNotFoundError。
# 当年的隔离测试办法(保留备查): 逐项把某个 torch 子模块放进 excludes, 用子进程跑一遍
#   import torch + nn/optim/sparse/torch.load —— 立刻失败的就是传递依赖, 不能裁。
#   别用静态 import 分析代替它, 那样漏掉的正是这类间接依赖。
#
# 实测必须**保留**的 (屏蔽后 import torch 立刻失败):
#   torch.distributed  <- torch.utils.data.dataloader 无条件 import
#   torchgen           <- torch.utils._python_dispatch 无条件 import
#   torch._functorch   <- torch._ops -> torch._functorch.pyfunctorch
#   torch._dynamo / torch._inductor / torch.ao / torch.testing  <- 同样被 torch 内部牵连
# 实测**可以**安全排除的 (屏蔽后 nn/optim/sparse/nn.functional/torch.load 全部正常):
#   torch.onnx (2.0 MB) / torch.utils.benchmark / torch.utils.tensorboard / torch.utils.model_dump
#
# 收益预期要放低: 真正的大头是 torch_cpu.dll, 这几个纯 Python 模块省下的量很小,
# 之所以还留着只是因为它们已经被验证安全。升级 torch 后请重跑上面那个脚本再改这份清单。
torch_excludes = [
    'torch.onnx',
    'torch.utils.benchmark',
    'torch.utils.tensorboard',
    'torch.utils.model_dump',
]

excludes = list(torch_excludes) + [
    # 明确未使用且体积可观的科学计算/绘图侧模块
    'tkinter.test',
    'test',
    'pydoc_data',
    # ⚠ 不要把 'unittest' 加回来! (2026-09-17 踩坑记录)
    #   打包后 exe 一启动就弹 "Unhandled exception in script" 对话框, stderr 是:
    #       File "主窗口.py", line 23, in <module>          <- matplotlib.backends.backend_tkagg
    #       File "matplotlib\rcsetup.py", line 28           <- import pyparsing
    #       File "pyparsing\__init__.py", line 156          <- from .testing import pyparsing_test as testing
    #       File "pyparsing\testing.py", line 6             <- import unittest
    #       ModuleNotFoundError: No module named 'unittest'
    #   pyparsing/__init__.py 对 .testing 是**无条件**导入 (不是 try/except, 也不在 __all__ 条件下),
    #   而 matplotlib 是 主窗口.py 的顶层依赖, 因此 unittest 属于**启动硬依赖**。
    #   另外 torch 也有 `import unittest.mock` (_export / _dispatch / _guards / _utils._config_module),
    #   显式用了它。
    #   代价: unittest 整包约 550 KB / 23 个文件, 相对包内 torch_cpu.dll 的 291 MB 可以忽略,
    #   不值得为这点体积去赌 import 链。
    #   守门: 打包exe.ps1 第 6 步会断言主窗口标题, 再出错会立刻报 FAIL 而不是误报"启动成功"。
]

a = Analysis(
    [os.path.join(proj_dir, '主程序.py')],
    pathex=[proj_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[os.path.join(proj_dir, 'rthook_torch.py')],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='华容道',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX: 仅在 PATH 里有 upx 时才生效; 缺失时 PyInstaller 静默跳过。
    # 压缩 DLL 风险大于收益, 这里保持 False, 需要更小体积请自行安装 upx 后开启。
    upx=False,
    console=False,  # 无控制台窗口
    icon=os.path.join(proj_dir, 'app.ico'),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='华容道',
)
