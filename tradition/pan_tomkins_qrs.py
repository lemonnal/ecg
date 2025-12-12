import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import wfdb
from tqdm import tqdm

# Set default font for better English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display
root = "/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/"
numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                     '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                     '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                     '231', '232', '233', '234']



class PanTomkinsQRSDetector:
    """
    基于Pan-Tomkins算法的QRS波检测器

    Pan-Tomkins算法是ECG信号处理中经典的QRS波检测算法，
    通过带通滤波、微分、平方和移动积分等步骤检测R波峰值
    """

    def __init__(self, fs=360, signal_name="MLII"):
        """
        初始化QRS检测器

        参数:
            fs: 采样频率 (Hz)
            adaptive_params: 是否使用自适应参数优化
        """
        self.fs = fs
        self.signal = None
        self.filtered_signal = None
        self.differentiated_signal = None
        self.squared_signal = None
        self.integrated_signal = None
        self.qrs_peaks = []
        
    def get_filter_parameters(self, signal_name="MLII"):
        """根据导联获取最优滤波参数"""
        
        # 基于导联特性的频率参数
        filter_params = {
            # 肢体导联-加压单极导联
            'aVR': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            'aVL': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            'aVF': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            # 肢体导联-标准双极导联
            'I': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            'MLII': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            'MLIII': {'low': 0.5, 'high': 40.0, 'threshold_factor': 1.4},
            
            # 胸前导联 - 
            # V1特殊处理
            'V1': {'low': 0.3, 'high': 50.0, 'threshold_factor': 1.2},
            # V1导联特点：R波小，S波深，需要更低频率捕获，更高频率保留细节
            
            # 胸前导联 - 过渡区
            'V2': {'low': 0.4, 'high': 45.0, 'threshold_factor': 1.3},
            # V2导联特点：介于V1和V3之间，中等参数
            
            # 胸前导联 - 左心前区
            'V3': {'low': 0.5, 'high': 45.0, 'threshold_factor': 1.4},
            'V4': {'low': 0.5, 'high': 45.0, 'threshold_factor': 1.4},
            'V5': {'low': 0.5, 'high': 45.0, 'threshold_factor': 1.4},
            'V6': {'low': 0.5, 'high': 45.0, 'threshold_factor': 1.4},
            # V3-V6导联特点：R波明显，标准参数即可
        }

        return filter_params.get(signal_name, filter_params['MLII'])

    def bandpass_filter(self, signal_data, signal_name="MLII"):
        """
        自适应带通滤波器
        根据不同导联使用不同的频率参数
        参数:
            signal_data: 输入ECG信号

        返回:
            filtered_signal: 滤波后的信号
        """
        # 获取该导联的滤波参数
        params = self.get_filter_parameters(signal_name)
        
        # 设计带通滤波器
        nyquist = 0.5 * self.fs
        low = params['low'] / nyquist
        high = params['high'] / nyquist

        # 使用3阶Butterworth滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(5, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # # # 为了减少漏检，添加原始信号的加权
        # original_weight = 0.3  # 原始信号权重
        # filtered_weight = 0.7  # 滤波信号权重
        # combined_signal = original_weight * signal_data + filtered_weight * filtered_signal
        combined_signal = filtered_signal
        return combined_signal

    def derivative(self, signal_data):
        """
        优化的微分器 - 使用5点中心差分
        更好地突出QRS波的高斜率特性，减少噪声影响

        参数:
            signal_data: 输入信号

        返回:
            differentiated_signal: 微分后的信号
        """
        differentiated_signal = np.zeros_like(signal_data)

        # 使用5点中心差分公式提高精度
        # f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
        for i in range(2, len(signal_data) - 2):
            differentiated_signal[i] = (
                                               -signal_data[i + 2] + 8 * signal_data[i + 1] - 8 * signal_data[i - 1] +
                                               signal_data[i - 2]
                                       ) / 12

        return differentiated_signal

    def squaring(self, signal_data):
        """
        平方函数
        使所有点为正值，并放大高斜率点

        参数:
            signal_data: 输入信号

        返回:
            squared_signal: 平方后的信号
        """
        return signal_data ** 2

    def moving_window_integration(self, signal_data, window_size=None):
        """
        优化的移动窗口积分器
        动态调整窗口大小以适应不同心率

        参数:
            signal_data: 输入信号
            window_size: 窗口大小 (样本数)，默认自适应

        返回:
            integrated_signal: 积分后的信号
        """
        if window_size is None:
            # 自适应窗口大小 - 基于QRS波群的典型宽度
            # 对于360Hz采样率，QRS波群约80ms，使用略大的窗口以确保完整覆盖
            window_size = int(0.080 * self.fs)  # 80ms窗口，更适合QRS波群

        # 使用卷积实现高效的移动平均积分
        window = np.ones(window_size) / window_size
        integrated_signal = np.convolve(signal_data, window, mode='same')

        return integrated_signal

    def detect_qrs_peaks(self, signal_data, signal_name="MLII"):
        """
        检测QRS波峰值
        使用双阈值检测算法

        参数:
            signal_data: 输入ECG信号
            signal_name: 信号名称（用于自适应参数）

        返回:
            qrs_peaks: QRS波峰值位置索引
        """
        # 保存信号名称用于自适应参数
        self.current_signal_name = signal_name

        # 步骤1: 带通滤波 - 传递signal_name以使用自适应参数
        self.filtered_signal = self.bandpass_filter(signal_data, signal_name)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: QRS检测 - 传递信号名称用于自适应阈值
        self.qrs_peaks = self._threshold_detection(signal_name)

        return self.qrs_peaks

    def _threshold_detection(self, signal_name="MLII"):
        """
        自适应阈值检测算法
        根据不同导联使用不同的阈值参数和不应期
        """
        if self.integrated_signal is None:
            return []

        # 获取该导联的自适应参数
        params = self.get_filter_parameters(signal_name)

        # 设置自适应阈值 - 基于信号的统计特性和导联特性
        signal_mean = np.mean(self.integrated_signal)
        signal_std = np.std(self.integrated_signal)
        threshold_factor = params.get('threshold_factor', 1.5)
        threshold = signal_mean + threshold_factor * signal_std

        # 设置自适应不应期 - V1导联使用更短的不应期以支持更高心率
        if signal_name == 'V1':
            refractory_period = int(0.15 * self.fs)  # 150ms，针对V1导联的小R波
        elif signal_name in ['V2', 'V3']:
            refractory_period = int(0.17 * self.fs)  # 170ms，过渡区导联
        else:
            refractory_period = int(0.2 * self.fs)   # 200ms，标准不应期

        peaks = []

        # 遍历信号寻找超过阈值的峰值
        for i in range(len(self.integrated_signal)):
            current_value = self.integrated_signal[i]

            # 检查是否超过阈值
            if current_value > threshold:
                # 检查是否在不应期内
                if len(peaks) == 0 or (i - peaks[-1]) > refractory_period:
                    # 在小窗口内寻找真正的峰值
                    search_window = min(10, len(self.integrated_signal) - i - 1)
                    local_peak = i

                    for j in range(max(0, i - 5), min(len(self.integrated_signal), i + search_window + 1)):
                        if self.integrated_signal[j] > self.integrated_signal[local_peak]:
                            local_peak = j

                    # 添加找到的峰值
                    if local_peak not in peaks:
                        peaks.append(local_peak)

        # 简单的峰值精确定位 - 在原始滤波信号上找到最大值
        refined_peaks = []
        search_window = int(0.03 * self.fs)  # 30ms搜索窗口

        for peak in peaks:
            search_start = max(0, peak - search_window)
            search_end = min(len(self.filtered_signal), peak + search_window)

            if search_start < search_end:
                search_segment = self.filtered_signal[search_start:search_end]
                if len(search_segment) > 0:
                    local_max_idx = np.argmax(search_segment) + search_start
                    refined_peaks.append(local_max_idx)
                else:
                    refined_peaks.append(peak)
            else:
                refined_peaks.append(peak)

        return refined_peaks

    # def _threshold_detection(self):
    #     """
    #     优化的阈值检测算法（已注释 - 现在使用下面的简单版本）
    #     使用自适应双阈值检测QRS波，包含初始化阶段和精确定位

    #     返回:
    #         peaks: 检测到的峰值位置
    #     """
    #     # ============================================
    #     # 原来的复杂阈值检测算法（已完全注释）
    #     # ============================================
    #     # # 初始化阶段 - 使用前2秒信号建立初始阈值
    #     # init_samples = int(2 * self.fs)
    #     # if len(self.integrated_signal) < init_samples:
    #     #     init_samples = len(self.integrated_signal)

    #     # init_signal = self.integrated_signal[:init_samples]
    #     # # 降低初始阈值，对小R波更敏感
    #     # init_threshold = np.mean(init_signal) + 2.0 * np.std(init_signal)

    #     # # 噪声和信号阈值初始化
    #     # signal_peak = init_threshold
    #     # noise_peak = np.mean(init_signal)
    #     # threshold = init_threshold

    #     # # 优化不应期参数 - 合理设置以平衡检测效果
    #     # rr_interval_min = int(0.2 * self.fs)   # 200ms (支持300bpm)
    #     # rr_interval_max = int(2.0 * self.fs)   # 2000ms (30bpm下限)

    #     # peaks = []
    #     # searchback_threshold = 0.25  # 适度降低回溯阈值，提高回溯敏感性

    #     # # 初始化标志 - 前几个心跳用于学习
    #     # learning_beats = 5  # 合理的学习时间，充分适应信号
    #     # learning_count = 0

    #     # for i in range(len(self.integrated_signal)):
    #     #     current_value = self.integrated_signal[i]

    #     #     # 检查是否超过阈值
    #     #     if current_value > threshold:
    #     #         # 检查是否在不应期内
    #     #         if len(peaks) == 0 or (i - peaks[-1]) > rr_interval_min:
    #     #             # 检查是否过长的间隔 (可能漏检)
    #     #             if len(peaks) > 0 and (i - peaks[-1]) > rr_interval_max:
    #     #                 # 触发回溯搜索
    #     #                 missed_peaks = self._searchback_detection(peaks[-1], i, searchback_threshold * threshold)
    #     #                 peaks.extend(missed_peaks)

    #     #             # 添加当前峰值
    #     #             peaks.append(i)

    #     #             # 学习阶段使用更高的学习率
    #     #             if learning_count < learning_beats:
    #     #                 learning_factor = 0.5
    #     #                 learning_count += 1
    #     #             else:
    #     #                 learning_factor = 0.125  # 稳定后使用较小学习率

    #     #             signal_peak = learning_factor * current_value + (1 - learning_factor) * signal_peak
    #     #         else:
    #     #             # 在不应期内，视为噪声
    #     #             noise_peak = 0.25 * current_value + 0.75 * noise_peak

    #     #         # 动态调整阈值更新策略 - 合理的阈值调整以平衡检测效果
    #     #         if learning_count < learning_beats:
    #     #             # 学习阶段：更积极的阈值调整
    #     #             threshold_factor = 0.35  # 提高学习因子，更敏感
    #     #         else:
    #     #             # 稳定阶段：保守但仍保持敏感性
    #     #             threshold_factor = 0.25  # 适度保守的阈值因子

    #     #         threshold = noise_peak + threshold_factor * (signal_peak - noise_peak)
    #     #     else:
    #     #         # 更新噪声峰值 - 适度适应噪声变化
    #     #         if current_value > noise_peak:
    #     #             noise_peak = 0.2 * current_value + 0.8 * noise_peak  # 提高噪声学习率

    #     #         # 在长时间没有检测到峰值时，逐渐降低阈值
    #     #         if len(peaks) > 0 and (i - peaks[-1]) > int(1.0 * self.fs):  # 超过1秒无峰值
    #     #             threshold *= 0.99  # 每个样本降低阈值1%

    #     # # 最终回溯搜索 - 检查最后一个长间隔
    #     # if len(peaks) > 0 and (len(self.integrated_signal) - peaks[-1]) > rr_interval_max * 0.8:
    #     #     missed_peaks = self._searchback_detection(peaks[-1], len(self.integrated_signal),
    #     #                                            searchback_threshold * threshold)
    #     #     peaks.extend(missed_peaks)

    #     # # 全局回溯搜索 - 检查所有间隔是否合理
    #     # if len(peaks) > 2:
    #     #     additional_peaks = []
    #     #     for i in range(len(peaks) - 1):
    #     #         interval = peaks[i+1] - peaks[i]
    #     #         if interval > rr_interval_max:  # 间隔过长，可能存在漏检
    #     #             missed_peaks = self._searchback_detection(peaks[i], peaks[i+1],
    #     #                                                    searchback_threshold * threshold)
    #     #             additional_peaks.extend(missed_peaks)

    #     #     # 合并并排序所有峰值
    #     #     all_peaks = sorted(peaks + additional_peaks)
    #     # else:
    #     #     all_peaks = peaks

    #     # # R波峰值精确定位
    #     # refined_peaks = self._refine_peak_locations(all_peaks)

    #     # return refined_peaks

    # # def _searchback_detection(self, start_idx, end_idx, threshold):
    # #     """
    # #     改进的回溯搜索检测遗漏的QRS波
    # #     """
    # #     search_start = start_idx + int(0.15 * self.fs)  # 缩短搜索起始延迟
    # #     search_end = min(end_idx, start_idx + int(1.5 * self.fs))  # 适当扩大搜索范围

    # #     if search_start >= search_end:
    # #         return []

    # #     search_segment = self.integrated_signal[search_start:search_end]
    # #     if len(search_segment) == 0:
    # #         return []

    # #     # 寻找局部最大值 - 使用合理的条件
    # #     peaks = []
    # #     min_peak_distance = int(0.25 * self.fs)  # 缩短最小峰值间距
    # #     local_threshold = threshold * 0.7  # 降低回溯搜索阈值

    # #     # 使用改进的峰值检测 - 考虑相对高度
    # #     for i in range(2, len(search_segment) - 2):
    # #         # 检查是否为局部最大值
    # #         if (search_segment[i] > local_threshold and
    # #             search_segment[i] > search_segment[i-1] and
    # #             search_segment[i] > search_segment[i+1] and
    # #             search_segment[i] > search_segment[i-2] and
    # #             search_segment[i] > search_segment[i+2]):

    # #             peak_idx = search_start + i

    # #             # 检查与已有峰值的距离
    # #             if not peaks or (peak_idx - peaks[-1]) > min_peak_distance:
    # #                 # 确保峰值足够显著
    # #                 window_size = min(20, i, len(search_segment) - i - 1)
    # #                 window_start = max(0, i - window_size)
    # #                 window_end = min(len(search_segment), i + window_size + 1)
    # #                 local_window = search_segment[window_start:window_end]

    # #                 if len(local_window) > 0 and search_segment[i] > np.mean(local_window) * 1.2:
    # #                     peaks.append(peak_idx)

    # #     return peaks

    # # def _refine_peak_locations(self, peak_indices):
    # #     """
    # #     将积分信号上的峰值位置精确定位到原始ECG信号的R波峰值
    # #     """
    # #     refined_peaks = []

    # #     for peak_idx in peak_indices:
    # #         # 在原始信号上搜索R波峰值
    # #         search_window = int(0.05 * self.fs)  # ±50ms搜索窗口
    # #         search_start = max(0, peak_idx - search_window)
    # #         search_end = min(len(self.filtered_signal),
    # #                        peak_idx + search_window)

    # #         if search_start < search_end:
    # #             search_segment = self.filtered_signal[search_start:search_end]
    # #             if len(search_segment) > 0:
    # #                 # 寻找局部最大值
    # #                 local_max_idx = np.argmax(search_segment) + search_start
    # #                 refined_peaks.append(local_max_idx)
    # #             else:
    # #                 refined_peaks.append(peak_idx)
    # #         else:
    # #             refined_peaks.append(peak_idx)

    # #     return refined_peaks

    def calculate_heart_rate(self):
        """
        计算心率

        返回:
            heart_rate_bpm: 平均心率 (bpm)
            rr_intervals: R-R间期数组 (ms)
        """
        if len(self.qrs_peaks) < 2:
            return 0, []

        # 计算R-R间期 (转换为ms)
        rr_intervals = np.diff(self.qrs_peaks) * 1000 / self.fs

        # 计算平均心率
        avg_rr_interval = np.mean(rr_intervals)
        heart_rate_bpm = 60000 / avg_rr_interval

        return heart_rate_bpm, rr_intervals

    def plot_results(self, signal_data, start_idx=0, num_samples=2000):
        """
        绘制增强的QRS检测结果，包含详细的处理步骤可视化

        参数:
            signal_data: 原始ECG信号
            start_idx: 起始索引
            num_samples: 显示的样本数
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(3, 2, figsize=(16, 10))

        # 时间轴
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # 1. 原始信号和R波检测
        ax1 = axes[0, 0]
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='原始ECG')

        # 标记检测到的R波
        for i, peak in enumerate(self.qrs_peaks):
            if start_idx <= peak < end_idx:
                ax1.plot(peak / self.fs, signal_data[peak], 'ro', markersize=8,
                         label='Detected R-wave' if i == 0 else "")
                # 添加R波编号
                ax1.annotate(f'R{i + 1}', (peak / self.fs, signal_data[peak]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_title('Optimized R-wave Detection Results')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 优化后的带通滤波信号
        ax2 = axes[0, 1]
        if self.filtered_signal is not None:
            ax2.plot(time_axis, self.filtered_signal[start_idx:end_idx], 'g-', linewidth=1,
                     label='Filtered Signal (5-45 Hz)')
            ax2.set_title('Bandpass Filtered Signal - Optimized Frequency Range')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)

        # 3. 优化后的微分信号
        ax3 = axes[1, 0]
        if self.differentiated_signal is not None:
            ax3.plot(time_axis, self.differentiated_signal[start_idx:end_idx], 'r-', linewidth=1,
                     label='5-point Derivative')
            ax3.set_title('Differentiated Signal - 5-point Central Difference')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Amplitude')
            ax3.grid(True, alpha=0.3)

        # 4. 积分信号和检测点
        ax4 = axes[1, 1]
        if self.integrated_signal is not None:
            ax4.plot(time_axis, self.integrated_signal[start_idx:end_idx], 'c-', linewidth=1.5,
                     label='Integrated Signal')

            # 标记积分信号上的检测点
            for peak in self.qrs_peaks:
                if start_idx <= peak < end_idx:
                    ax4.plot(peak / self.fs, self.integrated_signal[peak], 'ro', markersize=6)

            ax4.set_title('Moving Window Integration (80ms window)')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Amplitude')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        # 5. RR间期序列
        ax5 = axes[2, 0]
        if len(self.qrs_peaks) > 1:
            rr_intervals = np.diff(self.qrs_peaks) * 1000 / self.fs
            rr_times = np.array(self.qrs_peaks[1:]) / self.fs

            # 只显示在可视范围内的RR间期
            peaks_array = np.array(self.qrs_peaks[1:])
            mask = (peaks_array >= start_idx) & (peaks_array < end_idx)
            if np.any(mask):
                ax5.bar(rr_times[mask], rr_intervals[mask], width=0.01, alpha=0.7, color='blue')
                ax5.axhline(y=np.mean(rr_intervals), color='red', linestyle='--',
                            label=f'Mean: {np.mean(rr_intervals):.1f} ms')

            ax5.set_title('RR Interval Variability')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('RR Interval (ms)')
            ax5.legend()
            ax5.grid(True, alpha=0.3)

        # 6. 统计信息文本
        ax6 = axes[2, 1]
        ax6.axis('off')

        # 计算统计信息
        if len(self.qrs_peaks) > 1:
            rr_intervals = np.diff(self.qrs_peaks) * 1000 / self.fs
            heart_rate, _ = self.calculate_heart_rate()

            stats_text = f"""Detection Statistics

R-wave Detected: {len(self.qrs_peaks)}
Average Heart Rate: {heart_rate:.1f} bpm

RR Interval Statistics:
  Mean: {np.mean(rr_intervals):.1f} ms
  Std Dev: {np.std(rr_intervals):.1f} ms
  Range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms

Algorithm Parameters:
  Filter Band: 4-18 Hz (mixed signal)
  Derivative: 5-point central difference
  Integration Window: 80 ms
  Refractory Period: 200 ms (supports 300bpm)
  Learning Phase: 5 beats
  Searchback Threshold: 25%
  Dynamic Decay: Gradual threshold reduction
"""
        else:
            stats_text = "Detection Failed\nInsufficient R-waves"

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.show()


def main():
    # """
    # 主函数：读取ECG数据并应用Pan-Tomkins算法
    # """
    # print("初始化Pan-Tomkins QRS检测器...")
    #
    # # 创建QRS检测器实例
    # qrs_detector = PanTomkinsQRSDetector(fs=360)
    #
    # # 读取数据文件
    # data_path = '/home/yogsothoth/DataSet/old_dataset/mit-bih-dataset/ecg_'+ num_seq + '.txt'
    #
    # print(f"读取ECG数据: {data_path}")
    #
    # # 读取数据，跳过行号前缀
    # data = []
    # with open(data_path, 'r') as file:
    #     for line in file:
    #         # 移除行号前缀，只保留数值部分
    #         if '→' in line:
    #             numeric_part = line.split('→')[1].strip()
    #         else:
    #             numeric_part = line.strip()
    #
    #         if numeric_part:
    #             # 分割两列数据
    #             parts = numeric_part.split()
    #             if len(parts) >= 2:
    #                 data.append([float(parts[0]), float(parts[1])])
    #
    # # 转换为numpy数组
    # data = np.array(data)
    #
    # # 分离第一列和第二列信号
    # signal1 = data[:, 0]
    # signal2 = data[:, 1]

    # MIT-BIH
    # 导联方式 ['MLII', 'MLIII', 'V1', 'V2', 'V4', 'V5']
    total_annotation_num = {
        "I": 0,
        "MLII": 0,
        "MLIII": 0,
        "V1": 0,
        "V2": 0,
        "V3": 0,
        "V4": 0,
        "V5": 0}
    total_detection_num = {
        "I": 0,
        "MLII": 0,
        "MLIII": 0,
        "V1": 0,
        "V2": 0,
        "V3": 0,
        "V4": 0,
        "V5": 0}

    for num in tqdm(numberSet,
                 desc="ECG",
                 ncols=100,
                 bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n',
                 position=0,
                 leave=True):
        # 加载数据文件
        print(f"file:{num}")
        input_data = wfdb.rdrecord(root + num)
        sig_name = input_data.sig_name
        print(f"{sig_name}")
        signal1 = wfdb.rdrecord(root + num, channel_names=[sig_name[0]]).p_signal.flatten()
        signal2 = wfdb.rdrecord(root + num, channel_names=[sig_name[1]]).p_signal.flatten()
        print(f"{signal1.shape}, {signal2.shape}")

        # 加载标注文件
        annotation = wfdb.rdann(root + num, 'atr')
        fs = annotation.fs
        ann_len = annotation.ann_len
        sig_sample = annotation.sample
        sig_symbol = annotation.symbol
        total_annotation_num[sig_name[0]] += ann_len
        total_annotation_num[sig_name[1]] += ann_len

        # 创建QRS检测器实例
        qrs_detector1 = PanTomkinsQRSDetector(fs=fs)
        qrs_detector2 = PanTomkinsQRSDetector(fs=fs)

        # 对第一列信号进行QRS检测 - 传递导联名称以使用自适应参数
        qrs_peaks1 = qrs_detector1.detect_qrs_peaks(signal1, sig_name[0])
        heart_rate1, rr_intervals1 = qrs_detector1.calculate_heart_rate()
        total_detection_num[sig_name[0]] += len(qrs_peaks1)

        # 对第二列信号进行QRS检测 - 传递导联名称以使用自适应参数
        qrs_peaks2 = qrs_detector2.detect_qrs_peaks(signal2, sig_name[1])
        heart_rate2, rr_intervals2 = qrs_detector2.calculate_heart_rate()
        total_detection_num[sig_name[1]] += len(qrs_peaks2)

        # print(f"第一列信号检测到 {len(qrs_peaks1)} 个QRS波")
        # print(f"平均心率: {heart_rate1:.1f} bpm")
        # print(f"第二列信号检测到 {len(qrs_peaks2)} 个QRS波")
        # print(f"平均心率: {heart_rate2:.1f} bpm")

    print("\n" + "=" * 60)
    print("总体检测统计信息:")

    for key in total_detection_num.keys():
        detected = total_detection_num[key]
        annotated = total_annotation_num[key]

        if annotated > 0:
            detection_rate = (detected / annotated) * 100
            print(f"导联 {key}:")
            print(f"  检测到QRS波数量: {detected}")
            print(f"  标注QRS波数量: {annotated}")
            print(f"  检测率: {detection_rate:.2f}%")
            print(f"  漏检数: {max(0, annotated - detected)}")
            print("-" * 40)
        else:
            print(f"导联 {key}: 无标注数据")
            print(f"  检测到QRS波数量: {detected}")
            print("-" * 40)

    # 计算总体统计
    total_detected = sum(total_detection_num.values())
    total_annotated = sum(total_annotation_num.values())

    if total_annotated > 0:
        overall_rate = (total_detected / total_annotated) * 100
        print(f"总体统计:")
        print(f"  总检测数量: {total_detected}")
        print(f"  总标注数量: {total_annotated}")
        print(f"  总体检测率: {overall_rate:.2f}%")
        print(f"  总漏检数: {max(0, total_annotated - total_detected)}")
    else:
        print(f"总体统计:")
        print(f"  总检测数量: {total_detected}")
        print(f"  无标注数据进行比较")

    print("=" * 60)

        # # 绘制结果
        # print("\n绘制第一列信号的QRS检测结果...")
        # qrs_detector.plot_results(signal1, start_idx=0, num_samples=3000)
        #
        # print("\n绘制第二列信号的QRS检测结果...")
        # qrs_detector2.plot_results(signal2, start_idx=0, num_samples=3000)

        # # 打印统计信息
        # print("\n=== QRS检测统计信息 ===")
        # print(f"信号1 - QRS波原始数量: {len(ann_len)}")
        # print(f"信号1 - QRS波检测数量: {len(qrs_peaks1)}")
        # print(f"信号1 - 平均心率: {heart_rate1:.2f} bpm")
        # if len(rr_intervals1) > 0:
        #     print(f"信号1 - R-R间期均值: {np.mean(rr_intervals1):.2f} ms")
        #     print(f"信号1 - R-R间期标准差: {np.std(rr_intervals1):.2f} ms")
        #
        # print(f"\n信号2 - QRS波原始数量: {len(ann_len)}")
        # print(f"信号2 - QRS波检测数量: {len(qrs_peaks2)}")
        # print(f"信号2 - 平均心率: {heart_rate2:.2f} bpm")
        # if len(rr_intervals2) > 0:
        #     print(f"信号2 - R-R间期均值: {np.mean(rr_intervals2):.2f} ms")
        #     print(f"信号2 - R-R间期标准差: {np.std(rr_intervals2):.2f} ms")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
