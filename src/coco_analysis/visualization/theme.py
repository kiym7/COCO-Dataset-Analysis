"""统一视觉主题管理。

负责 matplotlib 样式设置、中文字体检测与配置、
以及图表的统一保存。
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns


# COCO 12 个超类别的颜色映射
SUPERCATEGORY_COLORS = {
    "person": "#1f77b4",
    "vehicle": "#ff7f0e",
    "outdoor": "#2ca02c",
    "animal": "#d62728",
    "accessory": "#9467bd",
    "sports": "#8c564b",
    "kitchen": "#e377c2",
    "food": "#7f7f7f",
    "furniture": "#bcbd22",
    "electronic": "#17becf",
    "appliance": "#aec7e8",
    "indoor": "#ffbb78",
}


# 备选调色板
PALETTES = {
    "default": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
                "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5"],
    "warm": ["#fbb4ae", "#b3cde3", "#ccebc5", "#decbe4", "#fed9a6",
             "#ffffcc", "#e5d8bd", "#fddaec", "#f2f2f2"],
}


def detect_chinese_font() -> Optional[str]:
    """自动检测系统可用的中文字体。

    按优先级尝试:
    1. SimHei (黑体) - Windows
    2. Microsoft YaHei (微软雅黑) - Windows
    3. WenQuanYi Micro Hei - Linux
    4. Noto Sans CJK SC - 跨平台
    5. Source Han Sans SC (思源黑体) - 跨平台

    Returns:
        找到的第一个可用字体名称，或 None
    """
    font_candidates = [
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Noto Sans CJK SC",
        "Noto Sans SC",
        "Source Han Sans SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
    ]

    available_fonts = {f.name for f in fm.fontManager.ttflist}

    for font_name in font_candidates:
        if font_name in available_fonts:
            return font_name

    return None


def set_chinese_font() -> str:
    """设置中文字体，如果不可用则打印安装提示。

    Returns:
        实际使用的字体名称
    """
    font_name = detect_chinese_font()

    if font_name:
        plt.rcParams["font.family"] = font_name
        plt.rcParams["axes.unicode_minus"] = False
        return font_name
    else:
        # 降级：使用 DejaVu Sans 并警告
        print("=" * 60)
        print("⚠ 未检测到中文字体，将使用英文标签。")
        print("  安装中文字体的方法:")
        print("  Windows: 确保系统已安装 'SimHei' 或 'Microsoft YaHei'")
        print("  Linux:   sudo apt install fonts-wqy-microhei")
        print("  Mac:     系统自带中文字体(PingFang SC)，应自动可用")
        print("=" * 60)
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["axes.unicode_minus"] = False
        return "sans-serif"


def set_style(style: str = "whitegrid", context: str = "notebook") -> None:
    """设置统一的 seaborn + matplotlib 样式。

    Args:
        style: seaborn 样式名 ('whitegrid', 'darkgrid', 'white', 'dark', 'ticks')
        context: seaborn 上下文 ('paper', 'notebook', 'talk', 'poster')
    """
    sns.set_style(style)
    sns.set_context(context)

    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })


def get_color_palette(n: int = 10, palette_name: str = "default") -> List[str]:
    """获取统一调色板。

    Args:
        n: 需要的颜色数量
        palette_name: 调色板名称 ('default' 或 'warm')

    Returns:
        颜色列表
    """
    colors = PALETTES.get(palette_name, PALETTES["default"])

    if n <= len(colors):
        return colors[:n]
    else:
        # 循环扩展
        repeats = (n + len(colors) - 1) // len(colors)
        return (colors * repeats)[:n]


def get_supercategory_color(supercategory: str) -> str:
    """获取超类别的对应颜色。

    Args:
        supercategory: 超类别名称

    Returns:
        对应的十六进制颜色
    """
    return SUPERCATEGORY_COLORS.get(
        supercategory, "#999999"  # 未知超类别用灰色
    )


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = "outputs/figures",
    formats: List[str] | None = None,
    dpi: int = 300,
    close: bool = True,
) -> List[str]:
    """统一保存图表到多种格式。

    Args:
        fig: matplotlib Figure 对象
        filename: 文件名（不含扩展名）
        output_dir: 输出目录
        formats: 输出格式列表 (如 ['png', 'pdf', 'svg'])
        dpi: 分辨率
        close: 保存后是否关闭 Figure

    Returns:
        保存的文件路径列表
    """
    if formats is None:
        formats = ["png"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(str(filepath), dpi=dpi, format=fmt)
        saved_paths.append(str(filepath))

    if close:
        plt.close(fig)

    return saved_paths


def initialize_visualization(config: dict | None = None) -> str:
    """初始化可视化环境（一站式调用）。

    调用 set_style + set_chinese_font，并返回使用的字体名称。

    Args:
        config: 可视化配置字典

    Returns:
        已设置的字体名称
    """
    cfg = config or {}
    style = cfg.get("style", "whitegrid")
    context = cfg.get("context", "notebook")

    set_style(style=style, context=context)
    return set_chinese_font()
