import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

# 读取数据文件
data_path = 'mit-bih-dataset/ecg_100.txt'

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
print(data)
# 分离第一列和第二列
column1 = data[:, 0]
column2 = data[:, 1]

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

def analyze_frequency_spectrum(signal_data, fs=360):
    """
    分析信号的频谱

    参数:
        signal_data: 输入信号数组
        fs: 采样频率 (Hz)

    返回:
        freqs: 频率数组
        magnitude: 幅度谱
    """
    # 计算FFT
    n = len(signal_data)
    fft_result = np.fft.fft(signal_data)
    freqs = np.fft.fftfreq(n, 1/fs)

    # 只取正频率部分
    positive_freqs = freqs[:n//2]
    magnitude = np.abs(fft_result[:n//2])

    return positive_freqs, magnitude

# 应用FFR滤波器
print("Applying FFR filter to ECG signals...")

# 对两列信号分别应用FFR滤波器
filtered_column1 = apply_ffr_filter(column1)
filtered_column2 = apply_ffr_filter(column2)

print("FFR filtering completed!")

# 频谱分析
print("Analyzing frequency spectrum...")
freqs1, magnitude1 = analyze_frequency_spectrum(column1)
freqs2, magnitude2 = analyze_frequency_spectrum(column2)

# 创建更全面的可视化
fig = plt.figure(figsize=(32, 24))

# 第一列：原始信号 vs 滤波后信号
ax1 = plt.subplot(3, 2, 1)
ax1.plot(column1[:1000], 'b-', alpha=0.6, label='Original')
ax1.plot(filtered_column1[:1000], 'r-', linewidth=2, label='FFR Filtered')
ax1.set_title('First Column - Original vs FFR Filtered')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Value')
ax1.legend()
ax1.grid(True)

# 第二列：原始信号 vs 滤波后信号
ax2 = plt.subplot(3, 2, 2)
ax2.plot(column2[:1000], 'b-', alpha=0.6, label='Original')
ax2.plot(filtered_column2[:1000], 'r-', linewidth=2, label='FFR Filtered')
ax2.set_title('Second Column - Original vs FFR Filtered')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Value')
ax2.legend()
ax2.grid(True)

# 第一列频谱分析
ax3 = plt.subplot(3, 2, 3)
ax3.plot(freqs1, magnitude1, 'b-', alpha=0.6, label='Original')
ax3.set_title('First Column - Frequency Spectrum')
ax3.set_xlabel('Frequency (Hz)')
ax3.set_ylabel('Magnitude')
ax3.set_xlim([0, 100])  # 显示0-100Hz范围
ax3.grid(True)

# 第二列频谱分析
ax4 = plt.subplot(3, 2, 4)
ax4.plot(freqs2, magnitude2, 'b-', alpha=0.6, label='Original')
ax4.set_title('Second Column - Frequency Spectrum')
ax4.set_xlabel('Frequency (Hz)')
ax4.set_ylabel('Magnitude')
ax4.set_xlim([0, 100])  # 显示0-100Hz范围
ax4.grid(True)

# 滤波后信号的频谱分析
ax5 = plt.subplot(3, 2, 5)
freqs1_filtered, magnitude1_filtered = analyze_frequency_spectrum(filtered_column1)
ax5.plot(freqs1_filtered, magnitude1_filtered, 'r-', linewidth=2, label='Filtered')
ax5.set_title('First Column - Filtered Frequency Spectrum')
ax5.set_xlabel('Frequency (Hz)')
ax5.set_ylabel('Magnitude')
ax5.set_xlim([0, 100])
ax5.grid(True)

ax6 = plt.subplot(3, 2, 6)
freqs2_filtered, magnitude2_filtered = analyze_frequency_spectrum(filtered_column2)
ax6.plot(freqs2_filtered, magnitude2_filtered, 'r-', linewidth=2, label='Filtered')
ax6.set_title('Second Column - Filtered Frequency Spectrum')
ax6.set_xlabel('Frequency (Hz)')
ax6.set_ylabel('Magnitude')
ax6.set_xlim([0, 100])
ax6.grid(True)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 打印数据信息和滤波效果
print(f"Data shape: {data.shape}")
print(f"Column 1 range: {np.min(column1):.6f} to {np.max(column1):.6f}")
print(f"Column 2 range: {np.min(column2):.6f} to {np.max(column2):.6f}")
print(f"Filtered Column 1 range: {np.min(filtered_column1):.6f} to {np.max(filtered_column1):.6f}")
print(f"Filtered Column 2 range: {np.min(filtered_column2):.6f} to {np.max(filtered_column2):.6f}")

# 计算信噪比改善
def calculate_snr(signal, noise_freq_range=(45, 55), signal_freq_range=(0.5, 40)):
    """计算信噪比"""
    freqs, magnitude = analyze_frequency_spectrum(signal)

    # 找到信号频带的能量
    signal_mask = (freqs >= signal_freq_range[0]) & (freqs <= signal_freq_range[1])
    signal_power = np.sum(magnitude[signal_mask]**2)

    # 找到噪声频带的能量 (例如50Hz工频干扰)
    noise_mask = (freqs >= noise_freq_range[0]) & (freqs <= noise_freq_range[1])
    noise_power = np.sum(magnitude[noise_mask]**2)

    snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    return snr_db

# 计算滤波前后的信噪比
snr1_original = calculate_snr(column1)
snr1_filtered = calculate_snr(filtered_column1)
snr2_original = calculate_snr(column2)
snr2_filtered = calculate_snr(filtered_column2)

print(f"\n=== FFR滤波器性能分析 ===")
print(f"滤波器参数:")
print(f"  - 低频截止: 0.5 Hz")
print(f"  - 高频截止: 40.0 Hz")
print(f"  - 滤波器阶数: 4阶")
print(f"  - 采样频率: 360 Hz")

print(f"\n信噪比改善:")
print(f"第一列:")
print(f"  原始信号 SNR: {snr1_original:.2f} dB")
print(f"  滤波后 SNR: {snr1_filtered:.2f} dB")
print(f"  改善: {snr1_filtered - snr1_original:.2f} dB")

print(f"第二列:")
print(f"  原始信号 SNR: {snr2_original:.2f} dB")
print(f"  滤波后 SNR: {snr2_filtered:.2f} dB")
print(f"  改善: {snr2_filtered - snr2_original:.2f} dB")