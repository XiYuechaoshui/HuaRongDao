# -*- coding: utf-8 -*-
"""PyInstaller 运行时钩子: 预加载 VC++ 运行时和 torch DLL

加载顺序 (关键):
  1. 系统 VC++ DLL (vcruntime140, vcruntime140_1, msvcp140, dbghelp)
  2. torch 依赖 DLL (libiomp5md, c10, torch_cpu, torch_python)

不按此顺序会导致 c10.dll 初始化失败 (WinError 1114)。
"""
import os
import sys

if hasattr(sys, 'frozen'):
    _base = os.path.dirname(sys.executable)
    _internal = os.path.join(_base, '_internal')
    _torch_lib = os.path.join(_internal, 'torch', 'lib')

    # 1. 注册 DLL 搜索路径
    for _p in [_internal, _torch_lib]:
        if os.path.isdir(_p):
            try:
                os.add_dll_directory(_p)
            except (OSError, AttributeError):
                pass
    if os.path.isdir(_torch_lib):
        os.environ['PATH'] = _torch_lib + os.pathsep + os.environ.get('PATH', '')

    # 2. 预加载系统 VC++ 运行时 DLL (从 System32 加载, 避免打包版本冲突)
    import ctypes
    _sys_dir = os.environ.get('SystemRoot', r'C:\Windows\System32')
    _vc_dlls = [
        'vcruntime140.dll',
        'vcruntime140_1.dll',
        'msvcp140.dll',
        'dbghelp.dll',
    ]
    for _dll in _vc_dlls:
        _path = os.path.join(_sys_dir, _dll)
        if os.path.exists(_path):
            try:
                ctypes.WinDLL(_path)
            except OSError:
                pass

    # 3. 预加载 torch 依赖 DLL (按依赖顺序)
    _torch_dlls = [
        'libiomp5md.dll',
        'c10.dll',
        'torch_cpu.dll',
        'torch_python.dll',
    ]
    for _dll in _torch_dlls:
        _path = os.path.join(_torch_lib, _dll)
        if os.path.exists(_path):
            try:
                ctypes.WinDLL(_path)
            except OSError:
                pass
