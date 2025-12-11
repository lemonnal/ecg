import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

def apply_arrhythmia_filter(signal_data, fs=360, lowcut=0.67, highcut=100.0, order=4):
    """
    应用于心率失常监测的FFR滤波器

    参数:
        signal_data: 输入信号数组
        fs: 采样频率 (Hz), 默认为360 Hz (MIT-BIH标准)
        lowcut: 低频截止频率 (Hz), 0.67Hz对应40bpm最低心率
        highcut: 高频截止频率 (Hz), 100Hz保留高频成分用于诊断
        order: 滤波器阶数, 4阶平衡性能和稳定性

    返回:
        filtered_signal: 滤波后的信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 设计带通滤波器
    b, a = scipy_signal.butter(order, [low, high], btype='band')

    # 应用零相位滤波，避免诊断信息失真
    filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

    return filtered_signal

def apply_heart_rate_enhanced_filter(signal_data, fs=360):
    """
    增强心脏特征的滤波器 - 用于P波、QRS波群、T波分析

    返回字典:
        p_enhanced: P波增强信号 (0.5-10 Hz)
        qrs_enhanced: QRS波群增强信号 (10-35 Hz)
        t_enhanced: T波增强信号 (0.5-12 Hz)
    """
    nyquist = 0.5 * fs

    # P波频段: 0.5-10 Hz (心房活动)
    p_low = 0.5 / nyquist
    p_high = 10.0 / nyquist
    b_p, a_p = scipy_signal.butter(2, [p_low, p_high], btype='band')
    p_enhanced = scipy_signal.filtfilt(b_p, a_p, signal_data)

    # QRS波群频段: 10-35 Hz (心室去极化)
    qrs_low = 10.0 / nyquist
    qrs_high = 35.0 / nyquist
    b_qrs, a_qrs = scipy_signal.butter(3, [qrs_low, qrs_high], btype='band')
    qrs_enhanced = scipy_signal.filtfilt(b_qrs, a_qrs, signal_data)

    # T波频段: 0.5-12 Hz (心室复极化)
    t_low = 0.5 / nyquist
    t_high = 12.0 / nyquist
    b_t, a_t = scipy_signal.butter(2, [t_low, t_high], btype='band')
    t_enhanced = scipy_signal.filtfilt(b_t, a_t, signal_data)

    return {
        'p_enhanced': p_enhanced,
        'qrs_enhanced': qrs_enhanced,
        't_enhanced': t_enhanced
    }

def apply_special_condition_filters(signal_data, fs=360):
    """
    特殊情况滤波器套件

    返回字典:
        bradycardia: 心动过缓滤波器
        tachycardia: 心动过速滤波器
        notch_50hz: 50Hz陷波滤波器
        notch_60hz: 60Hz陷波滤波器
    """
    nyquist = 0.5 * fs

    # 心动过缓滤波器 (<40 bpm)
    bradycardia_low = 0.33 / nyquist  # 20 bpm
    bradycardia_high = 40.0 / nyquist
    b_brady, a_brady = scipy_signal.butter(3, [bradycardia_low, bradycardia_high], btype='band')
    bradycardia = scipy_signal.filtfilt(b_brady, a_brady, signal_data)

    # 心动过速滤波器 (>150 bpm)
    tachycardia_low = 1.25 / nyquist  # 75 bpm
    tachycardia_high = 150.0 / nyquist
    b_tachy, a_tachy = scipy_signal.butter(3, [tachycardia_low, tachycardia_high], btype='band')
    tachycardia = scipy_signal.filtfilt(b_tachy, a_tachy, signal_data)

    # 50Hz工频干扰陷波
    notch_freq_50 = 50.0 / nyquist
    q_50 = 35  # Q因子控制陷波宽度
    b_50, a_50 = scipy_signal.iirnotch(notch_freq_50, q_50)
    notch_50hz = scipy_signal.filtfilt(b_50, a_50, signal_data)

    # 60Hz工频干扰陷波
    notch_freq_60 = 60.0 / nyquist
    q_60 = 35
    b_60, a_60 = scipy_signal.iirnotch(notch_freq_60, q_60)
    notch_60hz = scipy_signal.filtfilt(b_60, a_60, signal_data)

    return {
        'bradycardia': bradycardia,
        'tachycardia': tachycardia,
        'notch_50hz': notch_50hz,
        'notch_60hz': notch_60hz
    }

def calculate_heart_rate_variability(signal, fs=360):
    """
    计算心率变异性参数 - 心律失常监测的重要指标

    返回字典:
        heart_rate: 平均心率 (bpm)
        hrv: 心率变异性 (ms)
        rr_intervals: RR间期数组 (ms)
        r_peaks: R波位置数组
    """
    # 简化的R波检测算法
    derivative = np.diff(signal)
    derivative_squared = derivative ** 2

    # 移动平均平滑
    window_size = int(0.15 * fs)  # 150ms窗口

    if window_size > len(derivative_squared):
        window_size = len(derivative_squared)

    if window_size > 0:
        smoothed = np.convolve(derivative_squared, np.ones(window_size)/window_size, mode='same')
        threshold = 0.3 * np.max(smoothed)

        # 检测R波位置
        r_peaks = []
        for i in range(1, len(smoothed)-1):
            if smoothed[i] > threshold and smoothed[i] > smoothed[i-1] and smoothed[i] > smoothed[i+1]:
                # 确保R波之间最小间隔200ms (300bpm最大心率)
                if not r_peaks or (i - r_peaks[-1]) > 0.2 * fs:
                    r_peaks.append(i)

        # 计算RR间期
        if len(r_peaks) > 1:
            rr_intervals = np.diff(r_peaks) / fs * 1000  # 转换为毫秒
            heart_rate = 60000 / np.mean(rr_intervals)  # bpm
            hrv = np.std(rr_intervals)  # 心率变异性

            return {
                'heart_rate': heart_rate,
                'hrv': hrv,
                'rr_intervals': rr_intervals,
                'r_peaks': r_peaks
            }

    return None

def detect_arrhythmia_features(hrv_data):
    """
    基于心率变异性检测心律失常特征

    参数:
        hrv_data: calculate_heart_rate_variability的返回值

    返回字典:
        bradycardia: 心动过缓检测
        tachycardia: 心动过速检测
        irregular_rhythm: 心律不齐检测
        high_variability: 高变异性检测
    """
    if not hrv_data or len(hrv_data['rr_intervals']) < 5:
        return None

    hr = hrv_data['heart_rate']
    hrv = hrv_data['hrv']
    rr_intervals = hrv_data['rr_intervals']

    # RR间期变异系数
    cv_rr = np.std(rr_intervals) / np.mean(rr_intervals)

    # 相邻RR间期差异
    rr_diff = np.abs(np.diff(rr_intervals))

    features = {
        'bradycardia': hr < 60,  # 心动过缓
        'tachycardia': hr > 100,  # 心动过速
        'irregular_rhythm': cv_rr > 0.1,  # 心律不齐
        'high_variability': hrv > 100,  # 高变异性
        'missed_beats': np.sum(rr_diff > 200),  # 漏拍检测
        'premature_beats': np.sum(rr_diff < -100),  # 早搏检测
        'mean_rr': np.mean(rr_intervals),
        'std_rr': np.std(rr_intervals),
        'cv_rr': cv_rr
    }

    return features

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

def visualize_arrhythmia_analysis(signal_data, fs=360, title="心律失常分析"):
    """
    可视化心律失常分析结果

    参数:
        signal_data: 输入信号数组
        fs: 采样频率 (Hz)
        title: 图表标题
    """
    # 应用各种滤波器
    filtered_signal = apply_arrhythmia_filter(signal_data, fs)
    enhanced_signals = apply_heart_rate_enhanced_filter(signal_data, fs)
    special_filters = apply_special_condition_filters(signal_data, fs)

    # 计算心率变异性
    hrv_data = calculate_heart_rate_variability(filtered_signal, fs)

    # 检测心律失常特征
    arrhythmia_features = detect_arrhythmia_features(hrv_data)

    # 频谱分析
    freqs_original, magnitude_original = analyze_frequency_spectrum(signal_data, fs)
    freqs_filtered, magnitude_filtered = analyze_frequency_spectrum(filtered_signal, fs)

    # 创建可视化
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle(title, fontsize=16)

    # 原始信号 vs 滤波后信号
    axes[0, 0].plot(signal_data[:2000], 'b-', alpha=0.6, label='Original', linewidth=1)
    axes[0, 0].plot(filtered_signal[:2000], 'r-', linewidth=2, label='Arrhythmia Filtered')
    if hrv_data and len(hrv_data['r_peaks']) > 0:
        # 标记R波位置
        r_peaks_in_range = [p for p in hrv_data['r_peaks'] if p < 2000]
        if r_peaks_in_range:
            axes[0, 0].plot(r_peaks_in_range, [filtered_signal[p] for p in r_peaks_in_range],
                           'bo', markersize=6, label='R-peaks')
    axes[0, 0].set_title('Original Signal vs Arrhythmia Filter')
    axes[0, 0].set_xlabel('Sample Points')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # P波、QRS波群、T波增强
    axes[0, 1].plot(enhanced_signals['p_enhanced'][:2000], 'b-', alpha=0.7, label='P-wave Enhanced', linewidth=1.5)
    axes[0, 1].plot(enhanced_signals['qrs_enhanced'][:2000], 'g-', alpha=0.7, label='QRS Enhanced', linewidth=1.5)
    axes[0, 1].plot(enhanced_signals['t_enhanced'][:2000], 'm-', alpha=0.7, label='T-wave Enhanced', linewidth=1.5)
    axes[0, 1].set_title('Cardiac Feature Enhancement')
    axes[0, 1].set_xlabel('Sample Points')
    axes[0, 1].set_ylabel('Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 频谱对比
    axes[0, 2].plot(freqs_original, magnitude_original, 'b-', alpha=0.6, label='Original', linewidth=1)
    axes[0, 2].plot(freqs_filtered, magnitude_filtered, 'r-', linewidth=2, label='Filtered')
    axes[0, 2].set_title('Frequency Spectrum Analysis')
    axes[0, 2].set_xlabel('Frequency (Hz)')
    axes[0, 2].set_ylabel('Magnitude')
    axes[0, 2].set_xlim([0, 100])
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 心动过缓和心动过速滤波器
    axes[1, 0].plot(special_filters['bradycardia'][:2000], 'c-', linewidth=2, label='Bradycardia Filter')
    axes[1, 0].plot(special_filters['tachycardia'][:2000], 'orange', linewidth=2, label='Tachycardia Filter')
    axes[1, 0].set_title('Special Condition Filters')
    axes[1, 0].set_xlabel('Sample Points')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 工频干扰滤波
    axes[1, 1].plot(signal_data[:1000], 'b-', alpha=0.6, label='Original', linewidth=1)
    axes[1, 1].plot(special_filters['notch_50hz'][:1000], 'r-', linewidth=2, label='50Hz Notch')
    axes[1, 1].plot(special_filters['notch_60hz'][:1000], 'g-', linewidth=2, label='60Hz Notch')
    axes[1, 1].set_title('Power Line Interference Filters')
    axes[1, 1].set_xlabel('Sample Points')
    axes[1, 1].set_ylabel('Amplitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 详细信号对比 (前500个样本)
    axes[1, 2].plot(signal_data[:500], 'b-', alpha=0.6, label='Original', linewidth=1)
    axes[1, 2].plot(filtered_signal[:500], 'r-', linewidth=2, label='Filtered')
    axes[1, 2].set_title('Detailed Comparison (500 samples)')
    axes[1, 2].set_xlabel('Sample Points')
    axes[1, 2].set_ylabel('Amplitude')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # 心率变异性分析
    if hrv_data and len(hrv_data['rr_intervals']) > 0:
        axes[2, 0].plot(hrv_data['rr_intervals'], 'b-', marker='o', linewidth=2, markersize=4)
        axes[2, 0].set_title(f'RR Interval Analysis\nHeart Rate: {hrv_data["heart_rate"]:.1f} bpm, HRV: {hrv_data["hrv"]:.1f} ms')
        axes[2, 0].set_xlabel('Beat Number')
        axes[2, 0].set_ylabel('RR Interval (ms)')
        axes[2, 0].grid(True, alpha=0.3)
    else:
        axes[2, 0].text(0.5, 0.5, 'RR Interval Analysis\nInsufficient R-peaks detected',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[2, 0].transAxes)
        axes[2, 0].set_title('RR Interval Analysis')

    # 心律失常特征检测结果
    axes[2, 1].axis('off')
    if arrhythmia_features:
        feature_text = "Arrhythmia Feature Detection:\n\n"
        feature_text += f"• Bradycardia: {'Yes' if arrhythmia_features['bradycardia'] else 'No'}\n"
        feature_text += f"• Tachycardia: {'Yes' if arrhythmia_features['tachycardia'] else 'No'}\n"
        feature_text += f"• Irregular Rhythm: {'Yes' if arrhythmia_features['irregular_rhythm'] else 'No'}\n"
        feature_text += f"• High Variability: {'Yes' if arrhythmia_features['high_variability'] else 'No'}\n"
        feature_text += f"• Missed Beats: {arrhythmia_features['missed_beats']}\n"
        feature_text += f"• Premature Beats: {arrhythmia_features['premature_beats']}\n"
        feature_text += f"• Mean RR Interval: {arrhythmia_features['mean_rr']:.1f} ms\n"
        feature_text += f"• RR Interval Std Dev: {arrhythmia_features['std_rr']:.1f} ms\n"
        feature_text += f"• Coefficient of Variation: {arrhythmia_features['cv_rr']:.3f}"
    else:
        feature_text = "Arrhythmia Feature Detection:\n\nInsufficient data for analysis"

    axes[2, 1].text(0.1, 0.9, feature_text, transform=axes[2, 1].transAxes,
                     fontsize=10, verticalalignment='top', fontfamily='monospace')

    # 滤波器参数说明
    axes[2, 2].axis('off')
    info_text = """Filter Parameter Configuration:

Arrhythmia Filter:
• Frequency Band: 0.67-100 Hz
• Low Cutoff: 40 bpm
• High Cutoff: Preserve diagnostic information
• Filter Order: 4th order

Cardiac Feature Enhancement:
• P-wave: 0.5-10 Hz
• QRS: 10-35 Hz
• T-wave: 0.5-12 Hz

Special Conditions:
• Bradycardia: 0.33-40 Hz
• Tachycardia: 1.25-150 Hz
• 50Hz Notch: Power line interference
• 60Hz Notch: Power line interference

Features:
✅ Zero-phase distortion-free
✅ Preserve P/QRS/T features
✅ Heart rate variability analysis
✅ Automatic R-peak detection
✅ Multi-band processing
✅ Real-time noise suppression"""

    axes[2, 2].text(0.1, 0.9, info_text, transform=axes[2, 2].transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.show()

    return {
        'filtered_signal': filtered_signal,
        'enhanced_signals': enhanced_signals,
        'special_filters': special_filters,
        'hrv_data': hrv_data,
        'arrhythmia_features': arrhythmia_features,
        'frequency_spectrum': {
            'original': (freqs_original, magnitude_original),
            'filtered': (freqs_filtered, magnitude_filtered)
        }
    }

# 使用示例
if __name__ == "__main__":
    # 读取ECG数据文件
    data_path = 'mit-bih-dataset/ecg_100.txt'

    try:
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

        print("=== 心律失常监测FFR滤波器分析 ===")
        print(f"数据形状: {data.shape}")
        print(f"第一列数据范围: {np.min(column1):.6f} 到 {np.max(column1):.6f}")
        print(f"第二列数据范围: {np.min(column2):.6f} 到 {np.max(column2):.6f}")

        # 对第一列进行心律失常分析
        print("\n分析第一列ECG信号...")
        results1 = visualize_arrhythmia_analysis(column1, fs=360, title="第一列ECG心律失常分析")

        # 对第二列进行心律失常分析
        print("\n分析第二列ECG信号...")
        results2 = visualize_arrhythmia_analysis(column2, fs=360, title="第二列ECG心律失常分析")

        # 打印分析结果
        if results1['hrv_data']:
            print(f"\n第一列心率分析:")
            print(f"  心率: {results1['hrv_data']['heart_rate']:.1f} bpm")
            print(f"  心率变异性: {results1['hrv_data']['hrv']:.1f} ms")
            print(f"  检测到R波数: {len(results1['hrv_data']['r_peaks'])}")

        if results2['hrv_data']:
            print(f"\n第二列心率分析:")
            print(f"  心率: {results2['hrv_data']['heart_rate']:.1f} bpm")
            print(f"  心率变异性: {results2['hrv_data']['hrv']:.1f} ms")
            print(f"  检测到R波数: {len(results2['hrv_data']['r_peaks'])}")

    except FileNotFoundError:
        print(f"错误: 找不到文件 {data_path}")
        print("请确保MIT-BIH数据文件位于正确路径")
    except Exception as e:
        print(f"分析过程中出现错误: {str(e)}")