import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal


def apply_ffr_filter(signal_data, fs=360, lowcut=0.5, highcut=40.0, order=4):
    """
    应用FFR (Fast Fourier Response) 滤波器到信号

    参数:
        signal_data: 输入信号数组
        fs: 采样频率 (Hz), 默认为360 Hz (ECG常用采样率)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        order: 滤波器阶数

    返回:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 设计带通滤波器
    b, a = scipy_signal.butter(order, [low, high], btype='band')

    # 应用零相位滤波
    filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

    return filtered_signal


# 读取数据文件
data_path = 'mit-bih-dataset/ecg_108.txt'

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

# 应用FFR滤波器
print("Applying FFR filter to ECG signals...")

# 对两列信号分别应用FFR滤波器
filtered_column1 = apply_ffr_filter(column1)
filtered_column2 = apply_ffr_filter(column2)

print("FFR filtering completed!")

# 创建滤波前后的对比图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# 第一列：原始信号 vs 滤波后信号 (前1000个样本)
ax1.plot(column1[:2000], 'b-', alpha=0.7, label='Original', linewidth=1)
ax1.plot(filtered_column1[:2000], 'r-', label='FFR Filtered', linewidth=2)
ax1.set_title('Column 1 - Original vs FFR Filtered (First 1000 samples)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 第二列：原始信号 vs 滤波后信号 (前1000个样本)
ax2.plot(column2[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
ax2.plot(filtered_column2[:1000], 'r-', label='FFR Filtered', linewidth=2)
ax2.set_title('Column 2 - Original vs FFR Filtered (First 1000 samples)')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Amplitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 第一列：详细对比 (前200个样本，放大显示)
ax3.plot(column1[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
ax3.plot(filtered_column1[:200], 'r-', label='FFR Filtered', linewidth=2)
ax3.set_title('Column 1 - Detailed Comparison (First 200 samples)')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Amplitude')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 第二列：详细对比 (前200个样本，放大显示)
ax4.plot(column2[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
ax4.plot(filtered_column2[:200], 'r-', label='FFR Filtered', linewidth=2)
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