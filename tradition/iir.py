import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal


def apply_iir_filter(signal_data, fs=360, lowcut=0.5, highcut=40.0, order=4, filter_type='butterworth'):
    """
    应用IIR (Infinite Impulse Response) 滤波器到信号

    参数:
        signal_data: 输入信号数组
        fs: 采样频率 (Hz), 默认为360 Hz (ECG常用采样率)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        order: 滤波器阶数
        filter_type: 滤波器类型 ('butterworth', 'cheby1', 'cheby2', 'ellip', 'bessel')

    返回:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 设计IIR带通滤波器
    if filter_type == 'butterworth':
        b, a = scipy_signal.butter(order, [low, high], btype='band')
    elif filter_type == 'cheby1':
        b, a = scipy_signal.cheby1(order, 0.5, [low, high], btype='band')  # 0.5 dB ripple
    elif filter_type == 'cheby2':
        b, a = scipy_signal.cheby2(order, 20, [low, high], btype='band')   # 20 dB attenuation
    elif filter_type == 'ellip':
        b, a = scipy_signal.ellip(order, 0.5, 20, [low, high], btype='band')  # 0.5 dB ripple, 20 dB attenuation
    elif filter_type == 'bessel':
        b, a = scipy_signal.bessel(order, [low, high], btype='band')
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")

    # 应用零相位滤波
    filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

    return filtered_signal, b, a


def analyze_filter_stability(b, a):
    """
    分析IIR滤波器的稳定性

    参数:
        b: 滤波器分子系数
        a: 滤波器分母系数

    返回:
        is_stable: 布尔值，表示滤波器是否稳定
        poles: 滤波器极点
    """
    poles = np.roots(a)
    is_stable = np.all(np.abs(poles) < 1.0)
    return is_stable, poles


# 读取数据文件
data_path = 'mit-bih-dataset/ecg_102.txt'

# 读取数据，跳过行号前缀
data = []
with open(data_path, 'r') as file:
    for line in file:
        # 移除行号前缀，只保留数值部分
        if '→' in line:
            numeric_part = line.split('→')[1].strip()
        else:
            numeric_part = line.strip()

        if numeric_part:
            # 分割两列数据
            parts = numeric_part.split()
            if len(parts) >= 2:
                data.append([float(parts[0]), float(parts[1])])

# 转换为numpy数组
data = np.array(data)

# 分离第一列和第二列
column1 = data[:, 0]
column2 = data[:, 1]

# 应用IIR滤波器
print("Applying IIR filter to ECG signals...")

filter_types = ['butterworth', 'cheby1', 'cheby2', 'ellip', 'bessel']
for filter_type in filter_types:
    # 对两列信号分别应用IIR滤波器
    filtered_column1, b1, a1 = apply_iir_filter(column1, filter_type=filter_type)
    filtered_column2, b2, a2 = apply_iir_filter(column2, filter_type=filter_type)

    print("IIR filtering completed!")

    # 检查滤波器稳定性
    is_stable1, poles1 = analyze_filter_stability(b1, a1)
    is_stable2, poles2 = analyze_filter_stability(b2, a2)

    print(f"Filter 1 stability: {'Stable' if is_stable1 else 'Unstable'}")
    print(f"Filter 2 stability: {'Stable' if is_stable2 else 'Unstable'}")

    # 创建滤波前后的对比图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 第一列：原始信号 vs 滤波后信号 (前1000个样本)
    ax1.plot(column1[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
    ax1.plot(filtered_column1[:1000], 'r-', label='IIR Filtered', linewidth=2)
    ax1.set_title('Column 1 - Original vs IIR Filtered (First 1000 samples)')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 第二列：原始信号 vs 滤波后信号 (前1000个样本)
    ax2.plot(column2[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
    ax2.plot(filtered_column2[:1000], 'r-', label='IIR Filtered', linewidth=2)
    ax2.set_title('Column 2 - Original vs IIR Filtered (First 1000 samples)')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 第一列：详细对比 (前200个样本，放大显示)
    ax3.plot(column1[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
    ax3.plot(filtered_column1[:200], 'r-', label='IIR Filtered', linewidth=2)
    ax3.set_title('Column 1 - Detailed Comparison (First 200 samples)')
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Amplitude')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 第二列：详细对比 (前200个样本，放大显示)
    ax4.plot(column2[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
    ax4.plot(filtered_column2[:200], 'r-', label='IIR Filtered', linewidth=2)
    ax4.set_title('Column 2 - Detailed Comparison (First 200 samples)')
    ax4.set_xlabel('Sample Index')
    ax4.set_ylabel('Amplitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 打印基本信息
    print(f"Data shape: {data.shape}")
    print(f"Column 1 range: {np.min(column1):.6f} to {np.max(column1):.6f}")
    print(f"Column 2 range: {np.min(column2):.6f} to {np.max(column2):.6f}")
    print(f"Filtered Column 1 range: {np.min(filtered_column1):.6f} to {np.max(filtered_column1):.6f}")
    print(f"Filtered Column 2 range: {np.min(filtered_column2):.6f} to {np.max(filtered_column2):.6f}")

    # 打印滤波器信息
    print(f"\nIIR Filter 1 coefficients:")
    print(f"Numerator (b): {b1}")
    print(f"Denominator (a): {a1}")

    print(f"\nIIR Filter 2 coefficients:")
    print(f"Numerator (b): {b2}")
    print(f"Denominator (a): {a2}")

# # 额外的IIR滤波器类型对比
# print("\n" + "="*50)
# print("Comparison of different IIR filter types:")
# print("="*50)
#
# filter_types = ['butterworth', 'cheby1', 'cheby2', 'ellip', 'bessel']
# fig2, axes = plt.subplots(2, 3, figsize=(18, 12))
# axes = axes.flatten()
#
# for i, filter_type in enumerate(filter_types):
#     try:
#         filtered_test, _, _ = apply_iir_filter(column1[:1000], filter_type=filter_type)
#         axes[i].plot(column1[:1000], 'b-', alpha=0.5, label='Original', linewidth=1)
#         axes[i].plot(filtered_test, 'r-', label=f'{filter_type.title()}', linewidth=2)
#         axes[i].set_title(f'{filter_type.title()} IIR Filter')
#         axes[i].set_xlabel('Sample Index')
#         axes[i].set_ylabel('Amplitude')
#         axes[i].legend()
#         axes[i].grid(True, alpha=0.3)
#     except Exception as e:
#         axes[i].text(0.5, 0.5, f'Error with {filter_type}:\n{str(e)}',
#                     transform=axes[i].transAxes, ha='center', va='center')
#         axes[i].set_title(f'{filter_type.title()} IIR Filter - Error')
#
# # 隐藏最后一个子图
# axes[5].set_visible(False)
#
# plt.tight_layout()
# plt.show()