"""
ECG实时监测图像初始化模块

负责初始化matplotlib图形显示系统，包括：
- 子图配置和创建
- PQRST波形标记样式配置
- 搜索窗口可视化配置
- 所有绘图对象的创建和初始化
"""

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import numpy as np


# ============================================================================
# 子图配置：每个子图对应Pan-Tomkins算法的一个处理阶段
# ============================================================================
SUBPLOT_CONFIG = [
    {'ylabel': 'original signal', 'color': 'b', 'description': 'Original ECG Signal'},
    {'ylabel': 'filtered signal', 'color': 'g', 'description': 'Bandpass Filtered Signal'},
    {'ylabel': 'differentiated signal', 'color': 'm', 'description': 'Differentiated Signal'},
    {'ylabel': 'squared signal', 'color': 'y', 'description': 'Squared Signal'},
    {'ylabel': 'integrated signal', 'color': 'k', 'description': 'Moving Window Integrated Signal'}
]

# ============================================================================
# PQRST波形标记样式：定义每种波形的颜色、标记形状和说明
# ============================================================================
WAVE_MARKER_CONFIG = {
    'r': {'color': 'red', 'marker': 'o', 'label': 'R', 'size': 64, 'desc': 'R Peak - QRS Complex Main Peak'},
    'q': {'color': 'blue', 'marker': '^', 'label': 'Q', 'size': 64, 'desc': 'Q Wave - Negative Wave Before R Peak'},
    's': {'color': 'green', 'marker': 'v', 'label': 'S', 'size': 64, 'desc': 'S Wave - Negative Wave After R Peak'},
    'p': {'color': 'magenta', 'marker': 's', 'label': 'P', 'size': 64, 'desc': 'P Wave - Atrial Depolarization'},
    't': {'color': 'cyan', 'marker': 'D', 'label': 'T', 'size': 64, 'desc': 'T Wave - Ventricular Repolarization'}
}

# ============================================================================
# 搜索窗口样式：用于可视化每个波的搜索范围
# ============================================================================
SEARCH_WINDOW_CONFIG = {
    'q': {'color': 'blue', 'alpha': 0.40, 'label': 'Q Search'},
    's': {'color': 'green', 'alpha': 0.40, 'label': 'S Search'},
    'p': {'color': 'magenta', 'alpha': 0.40, 'label': 'P Search'},
    't': {'color': 'cyan', 'alpha': 0.40, 'label': 'T Search'}
}


def initialize_plot():
    """
    初始化ECG实时监测的图形显示系统
    
    返回:
        fig: matplotlib图形窗口对象
        axes: 子图数组
        lines: 信号曲线对象列表
        scatter_objects: PQRST波形标记对象字典
        search_window_collections: 搜索窗口可视化对象字典
    """
    
    # 开启交互模式，允许图形实时更新而不阻塞程序
    plt.ion()
    
    # ------------------------------------------------------------------------
    # 创建主图形窗口和5个垂直排列的子图
    # - 5个子图共享x轴（时间轴），便于对比不同处理阶段的信号
    # - figsize=(10, 8): 设置图形窗口大小为10x8英寸
    # ------------------------------------------------------------------------
    NUM_SUBPLOTS = len(SUBPLOT_CONFIG)
    fig, axes = plt.subplots(NUM_SUBPLOTS, 1, figsize=(10, 8), sharex=True)
    
    # 确保axes始终是列表（单子图时plt.subplots返回单个Axes对象）
    if NUM_SUBPLOTS == 1:
        axes = [axes]
    
    # ------------------------------------------------------------------------
    # 批量创建每个子图的信号曲线对象
    # ------------------------------------------------------------------------
    lines = []
    for ax, config in zip(axes, SUBPLOT_CONFIG):
        line, = ax.plot([], f"{config['color']}-", label=config['description'])
        ax.set_ylabel(config['ylabel'])
        lines.append(line)
    
    # 为第一个子图添加图例
    axes[0].legend(loc='upper right', fontsize=8)
    
    # ========================================================================
    # 批量创建scatter对象用于标记PQRST波的特征点
    # 优势：避免每次更新时重复创建/删除对象，大幅提升性能
    # ========================================================================
    scatter_objects = {}
    for wave_type, marker_cfg in WAVE_MARKER_CONFIG.items():
        scatter_list = []
        for i, ax in enumerate(axes):
            scatter = ax.scatter(
                [], [], 
                c=marker_cfg['color'],
                s=marker_cfg['size'],
                marker=marker_cfg['marker'],
                label=marker_cfg['label'] if i == 0 else None,  # 只在第一个子图显示标签
                zorder=5
            )
            scatter_list.append(scatter)
        scatter_objects[wave_type] = scatter_list
    
    # ========================================================================
    # 创建搜索窗口可视化对象（只在第一个子图显示）
    # 使用 PolyCollection 来高效绘制多个矩形区域
    # ========================================================================
    search_window_collections = {}
    for wave_type, window_cfg in SEARCH_WINDOW_CONFIG.items():
        collection = PolyCollection(
            [],  # 初始为空
            facecolors=window_cfg['color'],
            alpha=window_cfg['alpha'],
            edgecolors='none',
            zorder=1,  # 在背景层，不遮挡信号线和标记点
            label=window_cfg['label']
        )
        axes[0].add_collection(collection)
        search_window_collections[wave_type] = collection
    
    return fig, axes, lines, scatter_objects, search_window_collections

