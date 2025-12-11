import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 数据路径
root = '../../mit-bih-arrhythmia-database-1.0.0/'
# 可以选择不同的记录进行可视化
# numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
#              '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
#              '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
#              '231', '232', '233', '234']
record_number = '105'
start_sample = 4000
end_sample = 8000


def bandpass_filter(data, fs=360, lowcut=0.5, highcut=40.0, order=5):
    """
    带通滤波器函数

    Args:
        data: 输入信号
        fs: 采样频率 (Hz)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    # 设计巴特沃斯带通滤波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 创建带通滤波器
    b, a = signal.butter(order, [low, high], btype='band')

    # 应用零相位滤波（forward-backward filtering）
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def wavelet_filter(data, wavelet="db5", level=9):
    """小波去噪函数"""
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换，获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata, coeffs


def plot_wavelet_comparison(record_num='100', start_sample=0, end_sample=2000):
    """
    绘制小波变换前后信号对比图

    Args:
        record_num: 记录编号
        sample_range: 显示的样本点数
        fs: 采样频率 (Hz)
    """
    # 读取ECG数据
    print(f"正在读取 {record_num} 号心电数据...")
    record = wfdb.rdrecord(root + record_num, channel_names=['MLII'])
    data = record.p_signal.flatten()

    # 截取一段数据进行可视化
    original_signal = data[start_sample:end_sample]

    # 小波去噪
    wavelet_signal, coeffs = wavelet_filter(data=original_signal)
    # 确保长度一致
    wavelet_signal = wavelet_signal[:len(original_signal)]

    # 应用带通滤波器
    print("应用带通滤波器 (0.5-40Hz)...")
    filtered_signal = bandpass_filter(wavelet_signal, fs=360, lowcut=0.5, highcut=40.0, order=5)

    # 创建图像
    fig, axes = plt.subplots(5, 1, figsize=(15, 16))
    fig.suptitle(f'ECG Signal Processing Pipeline (Record {record_num})', fontsize=16, fontweight='bold')

    # 子图1: 原始信号
    axes[0].plot(original_signal, 'b', linewidth=1, alpha=0.8, label='Original Signal')
    axes[0].set_title('Original ECG Signal', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude (mV)', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')
    axes[0].set_xlim(0, end_sample - start_sample)

    # 子图2: 小波去噪后的信号
    axes[1].plot(wavelet_signal, 'r', linewidth=1.2, label='Wavelet Denoised')
    axes[1].set_title('After Wavelet Denoising', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude (mV)', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')
    axes[1].set_xlim(0, end_sample - start_sample)

    # 子图3: 带通滤波后的信号
    axes[2].plot(filtered_signal, 'purple', linewidth=1.2, label='Bandpass Filtered')
    axes[2].set_title('After Bandpass Filtering (0.5-40Hz)', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Amplitude (mV)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')
    axes[2].set_xlim(0, end_sample - start_sample)

    # 子图4: 小波去噪和带通滤波对比
    axes[3].plot(wavelet_signal, 'r', linewidth=1, alpha=0.7, label='Wavelet Only')
    axes[3].plot(filtered_signal, 'purple', linewidth=1.2, label='Wavelet + Bandpass')
    axes[3].set_title('Wavelet Denoised vs Bandpass Filtered', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Amplitude (mV)', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper right')
    axes[3].set_xlim(0, end_sample - start_sample)

    # 子图5: 最终去除的噪声
    final_noise = original_signal - filtered_signal
    axes[4].plot(final_noise, 'g', linewidth=1, label='Total Removed Noise')
    axes[4].set_title('Total Noise Removed (Wavelet + Bandpass)', fontsize=14, fontweight='bold')
    axes[4].set_ylabel('Amplitude (mV)', fontsize=12)
    axes[4].set_xlabel('Sample Points', fontsize=12)
    axes[4].grid(True, alpha=0.3)
    axes[4].legend(loc='upper right')
    axes[4].set_xlim(0, end_sample - start_sample)

    # 调整布局
    plt.tight_layout()
    plt.savefig(f'wavelet_comparison_{record_num}.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 计算并显示信噪比改善
    signal_power = np.var(original_signal)
    noise_power = np.var(final_noise)
    snr_original = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')

    # 计算处理前后的信号质量指标
    wavelet_noise = original_signal - wavelet_signal
    wavelet_noise_power = np.var(wavelet_noise)
    wavelet_snr = 10 * np.log10(signal_power / wavelet_noise_power) if wavelet_noise_power > 0 else float('inf')

    print(f"\n信号处理结果:")
    print(f"原始信号长度: {len(original_signal)} 样本点")
    print(f"原始信号方差: {signal_power:.4f}")
    print(f"小波去噪后噪声方差: {wavelet_noise_power:.4f}")
    print(f"小波去噪信噪比: {wavelet_snr:.2f} dB")
    print(f"最终噪声方差: {noise_power:.4f}")
    print(f"最终信噪比: {snr_original:.2f} dB")
    print(f"信噪比总改善: {snr_original - wavelet_snr:.2f} dB (通过带通滤波)")

    return fig


def plot_wavelet_coefficients(record_num='100', start_sample=0, end_sample=1000):
    """
    绘制小波系数的可视化

    Args:
        record_num: 记录编号
        sample_range: 显示的样本点数
    """
    # 读取ECG数据
    record = wfdb.rdrecord(root + record_num, channel_names=['MLII'])
    data = record.p_signal.flatten()[start_sample:end_sample]

    # 进行小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=6)

    # 创建图像
    fig, axes = plt.subplots(len(coeffs), 1, figsize=(15, 2 * len(coeffs)))
    fig.suptitle(f'Wavelet Coefficients (db5 wavelet, 6 levels) - Record {record_num}',
                 fontsize=16, fontweight='bold')

    # 绘制每一级的小波系数
    for i, coeff in enumerate(coeffs):
        if i == 0:
            title = f'Approximation Coefficients (cA{len(coeffs) - 1})'
            color = 'blue'
        else:
            title = f'Detail Coefficients (cD{len(coeffs) - i})'
            color = 'red'

        axes[i].plot(coeff, color=color, linewidth=0.8)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylabel('Coefficient', fontsize=10)

        # 设置x轴标签，只在最后一个子图显示
        if i == len(coeffs) - 1:
            axes[i].set_xlabel('Coefficient Index', fontsize=12)

    plt.tight_layout()
    plt.savefig(f'wavelet_coefficients_{record_num}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


if __name__ == "__main__":
    # 示例使用
    print("ECG小波变换和带通滤波可视化工具")
    print("=" * 50)

    # 绘制小波变换前后对比
    print(f"\n1. 绘制小波变换和带通滤波信号处理流程图...")
    plot_wavelet_comparison(record_number, start_sample=start_sample, end_sample=end_sample)

    # 绘制小波系数
    print(f"\n2. 绘制小波系数分解图...")
    plot_wavelet_coefficients(record_number, start_sample=start_sample, end_sample=end_sample)

    print(f"\n可视化完成！图片已保存为PNG格式。")
    print(f"\n处理流程说明:")
    print(f"1. 原始ECG信号")
    print(f"2. 小波去噪 (db5小波, 9级分解)")
    print(f"3. 带通滤波 (0.5-40Hz, 5阶巴特沃斯滤波器)")
    print(f"4. 最终输出: 清洁的ECG信号")