"""中文字体探测与应用工具

按白名单优先级从系统已安装字体里找到第一个完整可用的中文字体，
并提供 apply_cn_font() 便捷函数统一设置 matplotlib Axes 的中文字体。
"""
import os
from typing import Tuple

import matplotlib as mpl
from matplotlib.font_manager import FontProperties, fontManager, findSystemFonts

# 按推荐顺序尝试，过滤含 ExtG 等残缺/不可用字体
_CN_FONT_CANDIDATES = [
    ('Microsoft YaHei', ['yahei', 'microsoftyahei', '微软雅黑'], 'Microsoft YaHei'),
    ('SimHei', ['simhei', '黑体'], 'SimHei'),
    ('Noto Sans CJK SC', ['notosanscjksc', 'notosanscjk', 'notosans', 'noto'], 'Noto Sans CJK SC'),
    ('Source Han Sans SC', ['sourcehansanssc', 'sourcehansans', '思源黑体'], 'Source Han Sans SC'),
    ('PingFang SC', ['pingfang', '苹方'], 'PingFang SC'),
    ('WenQuanYi Zen Hei', ['wenquanyi', 'wqy', '文泉驿'], 'WenQuanYi Zen Hei'),
    ('DengXian', ['dengxian', '等线'], 'DengXian'),
    ('KaiTi', ['kaiti', '楷体'], 'KaiTi'),
    ('FangSong', ['fangsong', '仿宋'], 'FangSong'),
    ('Arial Unicode MS', ['arialunicodems'], 'Arial Unicode MS'),
]


def _resolve_cn_font() -> Tuple[FontProperties, str]:
    """按白名单优先级从系统已安装字体里找到第一个完整可用的中文字体。"""
    system_fonts = findSystemFonts()
    name_to_path = {}
    for fp in system_fonts:
        try:
            bn = os.path.basename(fp).lower().replace(' ', '').replace('-', '').replace('_', '')
            name_to_path[bn] = fp
            fontManager.addfont(fp)
        except Exception:
            pass

    selected_path = None
    selected_family = None
    selected_label = None

    for label, keywords, family in _CN_FONT_CANDIDATES:
        try:
            test_fp = FontProperties(family=[family])
            test_file = fontManager.findfont(test_fp, fallback_to_default=False)
            if test_file and 'dejavu' not in test_file.lower():
                if 'extg' in os.path.basename(test_file).lower():
                    continue
                selected_path = test_file
                selected_family = family
                selected_label = label
                break
        except Exception:
            pass

        for key in keywords:
            for k, p in name_to_path.items():
                if key in k:
                    if 'extg' in k:
                        continue
                    selected_path = p
                    selected_label = label
                    selected_family = family
                    break
            if selected_path:
                break
        if selected_path:
            break

    if selected_path:
        try:
            fontManager.addfont(selected_path)
        except Exception:
            pass
        fp = FontProperties(fname=selected_path)
        try:
            mpl.rcParams['font.sans-serif'] = [selected_family or selected_label,
                                               'Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        except Exception:
            pass
        return fp, selected_label

    return FontProperties(), 'DejaVu Sans (缺中文字体)'


# 全局共享的中文 FontProperties
CN_FONT, CN_FONT_NAME = _resolve_cn_font()
mpl.rcParams['axes.unicode_minus'] = False


def apply_cn_font(ax, *, title=None, xlabel=None, ylabel=None,
                  title_fontsize=11, label_fontsize=9, tick_fontsize=9,
                  legend_fontsize=9, legend=None):
    """对 matplotlib Axes 统一应用中文字体，返回 ax 以便链式调用。"""
    if title is not None:
        ax.set_title(title, fontproperties=CN_FONT, fontsize=title_fontsize, pad=8)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontproperties=CN_FONT, fontsize=label_fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontproperties=CN_FONT, fontsize=label_fontsize)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontproperties(CN_FONT)
        lbl.set_fontsize(tick_fontsize)
    if legend is not None:
        for t in legend.get_texts():
            t.set_fontproperties(CN_FONT)
            t.set_fontsize(legend_fontsize)
    elif ax.get_legend():
        for t in ax.get_legend().get_texts():
            t.set_fontproperties(CN_FONT)
            t.set_fontsize(legend_fontsize)
    return ax
