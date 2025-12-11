import wfdb
import pywt
import numpy as np
import torch
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.pyplot import figure
import os

# MIT-BIH
# 导联方式 ['MLII', 'V5', 'V2', 'V1', 'V4']
root = '/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/'
numberSet = ['104']  # 可以添加更多记录，如 ['100', '101', '103']

# 标签映射和颜色
label_map = {
    'N': 'Normal',
    'A': 'Atrial Premature',
    'V': 'Ventricular Premature',
    'L': 'Left Bundle Branch Block',
    'R': 'Right Bundle Branch Block',
    '/': 'Paced Beat',
    'j': 'Nodal (junctional) Premature',
    'S': 'Supraventricular Premature',
    'V': 'Ventricular Premature',
    'R': 'Right Bundle Branch Block',
    '~': 'Signal Quality Change',
    '+': 'Rhythm Change',
    'J': 'Nodal (junction) Escape Beat',
    'Q': 'Unclassifiable Beat',
    'a': 'Aberrated Atrial Premature',
    'F': 'Fusion of Ventricular and Normal',
    'x': 'Non-conducted P-wave',
    'L': 'Left Bundle Branch Block',
    'A': 'Atrial Premature',
    'E': 'Ventricular Escape Beat',
    'N': 'Normal',
    'e': 'Atrial Escape Beat',
    '|': 'Isolated QRS-like artifact',
    '"': 'Comment annotation',
    'f': 'Fusion of Ventricular and Normal'
}

# 为不同标签分配颜色
color_map = {
    'N': 'green',
    'A': 'orange',
    'V': 'red',
    'L': 'purple',
    'R': 'blue',
    '/': 'gray',
    'j': 'cyan',
    'S': 'magenta',
    '~': 'brown',
    '+': 'pink',
    'J': 'olive',
    'Q': 'black',
    'a': 'yellow',
    'F': 'darkgreen',
    'x': 'darkblue',
    'E': 'darkred',
    'e': 'lightblue',
    '|': 'lightgray',
    '"': 'lightgreen',
    'f': 'darkviolet'
}

def bandpass_filter(data, fs=360, lowcut=0.5, highcut=40.0, order=5):
    """
    带通滤波器函数
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data


def wavelet_filter(data, wavelet="db5", level=9):
    """小波去噪函数"""
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata, coeffs


def create_complete_ecg_plot(number, channel='MLII', save_path='./ecg_plots/'):
    """
    创建完整的ECG波形图，标注所有标注点

    Args:
        number: 记录号
        channel: 导联名称
        save_path: 保存路径
    """
    print(f"正在处理记录 {number} 的 {channel} 导联...")

    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)

    # 读取原始ECG数据
    print("  读取ECG数据...")
    record = wfdb.rdrecord(root + number, channel_names=[channel])
    original_signal = record.p_signal.flatten()

    # 读取标注文件
    print("  读取标注文件...")
    annotation = wfdb.rdann(root + number, 'atr')
    r_locations = annotation.sample
    r_symbols = annotation.symbol

    # 数据预处理
    print("  数据预处理...")
    filtered_signal, coeffs = wavelet_filter(data=original_signal)
    clean_signal = bandpass_filter(data=filtered_signal)

    # 计算信号统计信息
    signal_min = np.min(clean_signal)
    signal_max = np.max(clean_signal)
    signal_range = signal_max - signal_min
    signal_mean = np.mean(clean_signal)

    print(f"  信号长度: {len(clean_signal)} 采样点")
    print(f"  信号范围: [{signal_min:.3f}, {signal_max:.3f}]")
    print(f"  标注点数量: {len(r_locations)}")

    # 创建长图 - 根据信号长度动态调整
    # 假设每英寸显示1000个采样点来保持清晰度
    pixels_per_inch = 100
    samples_per_inch = 1000
    fig_width = len(clean_signal) / samples_per_inch
    fig_height = 12  # 固定高度以显示足够的波形细节

    print(f"  创建图形，尺寸: {fig_width:.1f} x {fig_height} 英寸")

    # 创建图形
    fig = figure(figsize=(fig_width, fig_height), dpi=100)
    ax = fig.add_subplot(111)

    # 绘制ECG信号
    time_axis = np.arange(len(clean_signal)) / 360  # 转换为秒
    ax.plot(time_axis, clean_signal, 'b-', linewidth=0.5, alpha=0.8, label='ECG Signal')

    # 标注所有标注点
    print("  标注标注点...")
    legend_elements = []
    annotated_labels = set()

    for i, (loc, sym) in enumerate(zip(r_locations, r_symbols)):
        if loc < len(clean_signal):  # 确保标注点在信号范围内
            time_point = loc / 360  # 转换为秒
            color = color_map.get(sym, 'gray')

            # 绘制垂直线标注
            ax.axvline(x=time_point, color=color, alpha=0.7, linestyle='--', linewidth=0.8)

            # 添加标注点
            ax.plot(time_point, clean_signal[loc], 'o', color=color, markersize=3, alpha=0.8)

            # 为每种标签类型创建图例（只添加一次）
            if sym not in annotated_labels and sym in label_map:
                label_name = f"{sym}: {label_map[sym]}"
                legend_elements.append(mpatches.Patch(color=color, label=label_name))
                annotated_labels.add(sym)

            # 每1000个标注点打印一次进度
            if i % 1000 == 0:
                print(f"    已处理 {i}/{len(r_locations)} 个标注点")

    # 设置图形属性
    ax.set_xlabel('时间 (秒)', fontsize=12)
    ax.set_ylabel('幅度 (mV)', fontsize=12)
    ax.set_title(f'MIT-BIH 记录 {number} - {channel} 导联完整ECG波形\n'
                f'信号长度: {len(clean_signal)} 采样点 ({len(clean_signal)/360:.1f} 秒) | '
                f'标注点: {len(r_locations)} 个', fontsize=14)

    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_facecolor('#f0f0f0')

    # 添加图例
    if legend_elements:
        # 将图例分成多列以避免过长
        n_cols = min(4, len(legend_elements))
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
                 ncol=n_cols, fontsize=10, title="标注类型")

    # 调整x轴范围
    ax.set_xlim(0, time_axis[-1])

    # 添加水平参考线
    ax.axhline(y=signal_mean, color='red', linestyle=':', alpha=0.5, label='均值')

    # 优化布局
    plt.tight_layout()

    # 保存图形
    output_filename = os.path.join(save_path, f'{number}_{channel}_complete_ecg.png')
    print(f"  保存图形到: {output_filename}")

    # 使用高DPI保存以保持清晰度
    fig.savefig(output_filename, dpi=200, bbox_inches='tight', facecolor='white')

    # 关闭图形以释放内存
    plt.close(fig)

    # 生成统计报告
    generate_annotation_report(number, r_symbols, save_path)

    return output_filename


def generate_annotation_report(number, symbols, save_path):
    """生成标注统计报告"""
    print("  生成标注统计报告...")

    # 统计各种标注类型的数量
    symbol_counts = {}
    for sym in symbols:
        symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

    # 保存报告
    report_filename = os.path.join(save_path, f'{number}_annotation_report.txt')
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write(f"MIT-BIH 记录 {number} 标注统计报告\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"总标注点数量: {len(symbols)}\n\n")
        f.write("标注类型统计:\n")
        f.write("-" * 30 + "\n")

        for sym, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True):
            label_name = label_map.get(sym, 'Unknown')
            f.write(f"{sym:2s} ({label_name:30s}): {count:6d} ({count/len(symbols)*100:5.2f}%)\n")

    print(f"  报告已保存: {report_filename}")


def main():
    """主函数"""
    print("开始创建完整ECG波形图...")
    print(f"处理记录: {numberSet}")

    # 创建输出目录
    output_dir = './ecg_complete_plots/'
    os.makedirs(output_dir, exist_ok=True)

    # 处理每个记录
    for number in numberSet:
        try:
            # 支持多个导联，这里主要使用MLII
            channels = ['V5']  # 可以添加其他导联，如 ['MLII', 'V1', 'V5']

            for channel in channels:
                try:
                    output_file = create_complete_ecg_plot(number, channel, output_dir)
                    print(f"✓ 成功创建: {output_file}")
                except Exception as e:
                    print(f"✗ 处理记录 {number} 导联 {channel} 时出错: {e}")
                    continue

        except Exception as e:
            print(f"✗ 处理记录 {number} 时出错: {e}")
            continue

    print("处理完成!")
    print(f"所有图形已保存到: {output_dir}")


if __name__ == '__main__':
    main()