import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import wfdb


# Set default font for better English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display
num_seq = '100'


class ComprehensiveECGDetector:
    """
    基于Pan-Tomkins算法的完整ECG特征点检测器

    检测ECG信号中的P、Q、R、S、T各个特征点
    """

    def __init__(self, fs=360):
        """
        初始化ECG检测器

        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        self.signal = None
        self.filtered_signal = None
        self.differentiated_signal = None
        self.squared_signal = None
        self.integrated_signal = None

        # 特征点检测结果
        self.r_peaks = []
        self.q_points = []
        self.s_points = []
        self.p_peaks = []
        self.p_onsets = []
        self.p_ends = []
        self.t_peaks = []
        self.t_ends = []

        # 算法参数
        self.qrs_window = int(0.1 * self.fs)  # QRS窗口 100ms
        self.p_window = int(0.3 * self.fs)    # P波窗口 300ms
        self.t_window = int(0.4 * self.fs)    # T波窗口 400ms

    def bandpass_filter(self, signal_data):
        """
        带通滤波器 (0.5-45 Hz)
        专门针对QRS波群的频率特性设计

        参数:
            signal_data: 输入ECG信号

        返回:
            filtered_signal: 滤波后的信号
        """
        # 设计带通滤波器 - 针对QRS波群优化频率范围，略微扩展频带
        nyquist = 0.5 * self.fs
        low = 5.0 / nyquist      # 略微降低低频截止，保留更多QRS信息
        high = 40.0 / nyquist    # 略微提高高频截止，保留高频成分

        # 使用3阶Butterworth滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(3, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # # 为了减少漏检，添加原始信号的加权
        # original_weight = 0.3  # 原始信号权重
        # filtered_weight = 0.7  # 滤波信号权重
        # combined_signal = original_weight * signal_data + filtered_weight * filtered_signal
        combined_signal = filtered_signal
        return combined_signal

    def lowpass_filter(self, signal_data, cutoff=10):
        """
        低通滤波器 - 用于P波和T波检测
        """
        nyquist = 0.5 * self.fs
        cutoff_norm = cutoff / nyquist

        b, a = scipy_signal.butter(3, cutoff_norm, btype='low')
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        return filtered_signal

    def derivative(self, signal_data):
        """
        5点中心差分微分器
        """
        differentiated_signal = np.zeros_like(signal_data)

        for i in range(2, len(signal_data) - 2):
            differentiated_signal[i] = (
                -signal_data[i+2] + 8*signal_data[i+1] - 8*signal_data[i-1] + signal_data[i-2]
            ) / 12

        return differentiated_signal

    def squaring(self, signal_data):
        """
        平方函数
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

    def detect_r_peaks(self, signal_data):
        """
        检测R波峰值 - 基于Pan-Tomkins算法
        """
        # 步骤1: 带通滤波
        self.filtered_signal = self.bandpass_filter(signal_data)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: 阈值检测
        self.r_peaks = self._threshold_detection_r()

        return self.r_peaks

    def _threshold_detection_r(self):
        """
        优化的阈值检测算法
        使用自适应双阈值检测QRS波，包含初始化阶段和精确定位

        返回:
            peaks: 检测到的峰值位置
        """
        # 初始化阶段 - 使用前2秒信号建立初始阈值
        init_samples = int(2 * self.fs)
        if len(self.integrated_signal) < init_samples:
            init_samples = len(self.integrated_signal)

        init_signal = self.integrated_signal[:init_samples]
        # 降低初始阈值，对小R波更敏感
        init_threshold = np.mean(init_signal) + 2.0 * np.std(init_signal)

        # 噪声和信号阈值初始化
        signal_peak = init_threshold
        noise_peak = np.mean(init_signal)
        threshold = init_threshold

        # 优化不应期参数 - 合理设置以平衡检测效果
        rr_interval_min = int(0.2 * self.fs)   # 200ms (支持300bpm)
        rr_interval_max = int(2.0 * self.fs)   # 2000ms (30bpm下限)

        peaks = []
        searchback_threshold = 0.25  # 适度降低回溯阈值，提高回溯敏感性

        # 初始化标志 - 前几个心跳用于学习
        learning_beats = 5  # 合理的学习时间，充分适应信号
        learning_count = 0

        for i in range(len(self.integrated_signal)):
            current_value = self.integrated_signal[i]

            # 检查是否超过阈值
            if current_value > threshold:
                # 检查是否在不应期内
                if len(peaks) == 0 or (i - peaks[-1]) > rr_interval_min:
                    # 检查是否过长的间隔 (可能漏检)
                    if len(peaks) > 0 and (i - peaks[-1]) > rr_interval_max:
                        # 触发回溯搜索
                        missed_peaks = self._searchback_detection(peaks[-1], i, searchback_threshold * threshold)
                        peaks.extend(missed_peaks)

                    # 添加当前峰值
                    peaks.append(i)

                    # 学习阶段使用更高的学习率
                    if learning_count < learning_beats:
                        learning_factor = 0.5
                        learning_count += 1
                    else:
                        learning_factor = 0.125  # 稳定后使用较小学习率

                    signal_peak = learning_factor * current_value + (1 - learning_factor) * signal_peak
                else:
                    # 在不应期内，视为噪声
                    noise_peak = 0.25 * current_value + 0.75 * noise_peak

                # 动态调整阈值更新策略 - 合理的阈值调整以平衡检测效果
                if learning_count < learning_beats:
                    # 学习阶段：更积极的阈值调整
                    threshold_factor = 0.35  # 提高学习因子，更敏感
                else:
                    # 稳定阶段：保守但仍保持敏感性
                    threshold_factor = 0.25  # 适度保守的阈值因子

                threshold = noise_peak + threshold_factor * (signal_peak - noise_peak)
            else:
                # 更新噪声峰值 - 适度适应噪声变化
                if current_value > noise_peak:
                    noise_peak = 0.2 * current_value + 0.8 * noise_peak  # 提高噪声学习率

                # 在长时间没有检测到峰值时，逐渐降低阈值
                if len(peaks) > 0 and (i - peaks[-1]) > int(1.0 * self.fs):  # 超过1秒无峰值
                    threshold *= 0.99  # 每个样本降低阈值1%

        # 最终回溯搜索 - 检查最后一个长间隔
        if len(peaks) > 0 and (len(self.integrated_signal) - peaks[-1]) > rr_interval_max * 0.8:
            missed_peaks = self._searchback_detection(peaks[-1], len(self.integrated_signal),
                                                   searchback_threshold * threshold)
            peaks.extend(missed_peaks)

        # 全局回溯搜索 - 检查所有间隔是否合理
        if len(peaks) > 2:
            additional_peaks = []
            for i in range(len(peaks) - 1):
                interval = peaks[i+1] - peaks[i]
                if interval > rr_interval_max:  # 间隔过长，可能存在漏检
                    missed_peaks = self._searchback_detection(peaks[i], peaks[i+1],
                                                           searchback_threshold * threshold)
                    additional_peaks.extend(missed_peaks)

            # 合并并排序所有峰值
            all_peaks = sorted(peaks + additional_peaks)
        else:
            all_peaks = peaks

        # R波峰值精确定位
        refined_peaks = self._refine_r_locations(all_peaks)
        return refined_peaks

    def _searchback_detection(self, start_idx, end_idx, threshold):
        """
        改进的回溯搜索检测遗漏的QRS波
        """
        search_start = start_idx + int(0.15 * self.fs)  # 缩短搜索起始延迟
        search_end = min(end_idx, start_idx + int(1.5 * self.fs))  # 适当扩大搜索范围

        if search_start >= search_end:
            return []

        search_segment = self.integrated_signal[search_start:search_end]
        if len(search_segment) == 0:
            return []

        # 寻找局部最大值
        peaks = []
        min_peak_distance = int(0.25 * self.fs)  # 缩短最小峰值间距
        local_threshold = threshold * 0.7  # 降低回溯搜索阈值

        for i in range(2, len(search_segment) - 2):
            # 检查是否为局部最大值
            if (search_segment[i] > local_threshold and
                search_segment[i] > search_segment[i-1] and
                search_segment[i] > search_segment[i+1] and
                search_segment[i] > search_segment[i-2] and
                search_segment[i] > search_segment[i+2]):

                peak_idx = search_start + i

                # 检查与已有峰值的距离
                if not peaks or (peak_idx - peaks[-1]) > min_peak_distance:
                    # 确保峰值足够显著
                    window_size = min(20, i, len(search_segment) - i - 1)
                    window_start = max(0, i - window_size)
                    window_end = min(len(search_segment), i + window_size + 1)
                    local_window = search_segment[window_start:window_end]

                    if len(local_window) > 0 and search_segment[i] > np.mean(local_window) * 1.2:
                        peaks.append(peak_idx)

        return peaks

    def _refine_r_locations(self, peak_indices):
        """
        将积分信号上的峰值位置精确定位到原始ECG信号的R波峰值
        """
        refined_peaks = []

        for peak_idx in peak_indices:
            # 在原始信号上搜索R波峰值
            search_window = int(0.05 * self.fs)  # ±50ms搜索窗口
            search_start = max(0, peak_idx - search_window)
            search_end = min(len(self.filtered_signal),
                           peak_idx + search_window)

            if search_start < search_end:
                search_segment = self.filtered_signal[search_start:search_end]
                if len(search_segment) > 0:
                    # 寻找局部最大值
                    local_max_idx = np.argmax(search_segment) + search_start
                    refined_peaks.append(local_max_idx)

        return refined_peaks

    def detect_qrs_points(self):
        """
        检测Q点和S点
        """
        self.q_points = []
        self.s_points = []

        for r_peak in self.r_peaks:
            # 检测Q点 (R波前)
            q_search_start = max(0, r_peak - self.qrs_window // 2)
            q_search_end = r_peak

            if q_search_start < q_search_end:
                q_segment = self.filtered_signal[q_search_start:q_search_end]
                if len(q_segment) > 0:
                    # Q点通常是R波前的最小值
                    q_local_idx = np.argmin(q_segment) + q_search_start
                    self.q_points.append(q_local_idx)

            # 检测S点 (R波后)
            s_search_start = r_peak
            s_search_end = min(len(self.filtered_signal), r_peak + self.qrs_window // 2)

            if s_search_start < s_search_end:
                s_segment = self.filtered_signal[s_search_start:s_search_end]
                if len(s_segment) > 0:
                    # S点通常是R波后的最小值
                    s_local_idx = np.argmin(s_segment) + s_search_start
                    self.s_points.append(s_local_idx)

    def detect_p_waves(self):
        """
        检测P波特征点 (P_onset, P_peak, P_end)
        """
        self.p_peaks = []
        self.p_onsets = []
        self.p_ends = []

        # 使用低通滤波突出P波
        p_filtered = self.lowpass_filter(self.signal, cutoff=10)

        for i, r_peak in enumerate(self.r_peaks):
            # P波在R波前的搜索窗口
            if i == 0:
                # 第一个心跳，从信号开始搜索
                p_search_start = 0
            else:
                # 正常情况，从前一个R波后开始搜索
                p_search_start = self.r_peaks[i-1] + int(0.2 * self.fs)

            p_search_end = self.q_points[i] if i < len(self.q_points) else r_peak - int(0.05 * self.fs)

            if p_search_start < p_search_end:
                p_segment = p_filtered[p_search_start:p_search_end]
                if len(p_segment) > 0:
                    # P波峰值 - 通常是最大值
                    p_peak_local_idx = np.argmax(p_segment) + p_search_start
                    self.p_peaks.append(p_peak_local_idx)

                    # P波起始点 - 查找上升沿起点
                    p_onset_start = p_search_start
                    p_onset_end = p_peak_local_idx

                    if p_onset_start < p_onset_end:
                        p_onset_segment = p_filtered[p_onset_start:p_onset_end]
                        # 寻找斜率变化最大点
                        p_onset_local = self._find_onset_point(p_onset_segment)
                        if p_onset_local is not None:
                            self.p_onsets.append(p_onset_start + p_onset_local)
                        else:
                            self.p_onsets.append(p_search_start)

                    # P波结束点 - 查找到基线
                    p_end_start = p_peak_local_idx
                    p_end_end = p_search_end

                    if p_end_start < p_end_end:
                        p_end_segment = p_filtered[p_end_start:p_end_end]
                        # 寻找返回基线的点
                        p_end_local = self._find_offset_point(p_end_segment)
                        if p_end_local is not None:
                            self.p_ends.append(p_end_start + p_end_local)
                        else:
                            self.p_ends.append(p_search_end)

    def detect_t_waves(self):
        """
        检测T波特征点 (T_peak, T_end)
        """
        self.t_peaks = []
        self.t_ends = []

        # 使用更低的截止频率突出T波
        t_filtered = self.lowpass_filter(self.signal, cutoff=8)

        for i, r_peak in enumerate(self.r_peaks):
            # T波在S波后的搜索窗口
            t_search_start = self.s_points[i] + int(0.05 * self.fs) if i < len(self.s_points) else r_peak + int(0.1 * self.fs)

            if i < len(self.r_peaks) - 1:
                # 不是最后一个心跳，到下一个R波前
                t_search_end = self.r_peaks[i+1] - int(0.1 * self.fs)
            else:
                # 最后一个心跳，到信号结束
                t_search_end = len(t_filtered)

            t_search_end = min(t_search_end, t_search_start + self.t_window)

            if t_search_start < t_search_end:
                t_segment = t_filtered[t_search_start:t_search_end]
                if len(t_segment) > 0:
                    # T波峰值 - 可能是正波或负波，取绝对值最大
                    t_peak_local_idx = np.argmax(np.abs(t_segment)) + t_search_start
                    self.t_peaks.append(t_peak_local_idx)

                    # T波结束点
                    t_end_start = t_peak_local_idx
                    t_end_end = t_search_end

                    if t_end_start < t_end_end:
                        t_end_segment = t_filtered[t_end_start:t_end_end]
                        # 寻找返回基线的点
                        t_end_local = self._find_offset_point(t_end_segment)
                        if t_end_local is not None:
                            self.t_ends.append(t_end_start + t_end_local)
                        else:
                            self.t_ends.append(t_search_end)

    def _find_onset_point(self, segment):
        """
        查找波形的起始点
        """
        if len(segment) < 10:
            return None

        # 计算梯度
        gradient = np.gradient(segment)

        # 寻找梯度开始显著增加的点
        threshold = np.std(gradient) * 0.5

        for i in range(1, len(gradient) - 1):
            if gradient[i] > threshold and gradient[i] > gradient[i-1]:
                return i

        return 0

    def _find_offset_point(self, segment):
        """
        查找波形的结束点
        """
        if len(segment) < 10:
            return None

        # 计算梯度
        gradient = np.gradient(segment)

        # 寻找梯度接近零的点
        threshold = np.std(gradient) * 0.3

        for i in range(len(gradient) - 1, 0, -1):
            if abs(gradient[i]) < threshold:
                return i

        return len(segment) - 1

    def detect_all_features(self, signal_data):
        """
        检测所有ECG特征点
        """
        self.signal = signal_data

        # 检测R波
        print("检测R波峰值...")
        self.r_peaks = self.detect_r_peaks(signal_data)
        print(f"检测到 {len(self.r_peaks)} 个R波")

        if len(self.r_peaks) > 0:
            # 检测Q点和S点
            print("检测Q点和S点...")
            self.detect_qrs_points()

            # 检测P波
            print("检测P波特征...")
            self.detect_p_waves()

            # 检测T波
            print("检测T波特征...")
            self.detect_t_waves()

        return {
            'r_peaks': self.r_peaks,
            'q_points': self.q_points,
            's_points': self.s_points,
            'p_peaks': self.p_peaks,
            'p_onsets': self.p_onsets,
            'p_ends': self.p_ends,
            't_peaks': self.t_peaks,
            't_ends': self.t_ends
        }

    def plot_detailed_ecg(self, signal_data, start_idx=0, num_samples=3000):
        """
        绘制详细的ECG特征检测结果
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(3, 1, figsize=(16, 12))

        # 时间轴
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # 1. 原始信号和所有特征点
        ax1 = axes[0]
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.8, label='ECG Signal')

        # 标记R波
        for r_peak in self.r_peaks:
            if start_idx <= r_peak < end_idx:
                ax1.plot(r_peak/self.fs, signal_data[r_peak], 'ro', markersize=10, label='R Peak' if r_peak == self.r_peaks[0] else "")

        # 标记Q点
        for i, q_point in enumerate(self.q_points):
            if start_idx <= q_point < end_idx:
                ax1.plot(q_point/self.fs, signal_data[q_point], 'g^', markersize=8, label='Q Point' if i == 0 else "")

        # 标记S点
        for i, s_point in enumerate(self.s_points):
            if start_idx <= s_point < end_idx:
                ax1.plot(s_point/self.fs, signal_data[s_point], 'g^', markersize=8, label='S Point' if i == 0 else "")

        # 标记P波特征
        for i, p_peak in enumerate(self.p_peaks):
            if start_idx <= p_peak < end_idx:
                ax1.plot(p_peak/self.fs, signal_data[p_peak], 'ms', markersize=8, label='P Peak' if i == 0 else "")

        for i, p_onset in enumerate(self.p_onsets):
            if start_idx <= p_onset < end_idx:
                ax1.plot(p_onset/self.fs, signal_data[p_onset], 'c|', markersize=10, label='P Onset' if i == 0 else "")

        for i, p_end in enumerate(self.p_ends):
            if start_idx <= p_end < end_idx:
                ax1.plot(p_end/self.fs, signal_data[p_end], 'c|', markersize=10, label='P End' if i == 0 else "")

        # 标记T波特征
        for i, t_peak in enumerate(self.t_peaks):
            if start_idx <= t_peak < end_idx:
                ax1.plot(t_peak/self.fs, signal_data[t_peak], 'md', markersize=8, label='T Peak' if i == 0 else "")

        for i, t_end in enumerate(self.t_ends):
            if start_idx <= t_end < end_idx:
                ax1.plot(t_end/self.fs, signal_data[t_end], 'y|', markersize=10, label='T End' if i == 0 else "")

        ax1.set_title('Comprehensive ECG Feature Detection', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. 滤波后信号和积分信号
        ax2 = axes[1]
        if self.filtered_signal is not None:
            ax2.plot(time_axis, self.filtered_signal[start_idx:end_idx], 'g-', linewidth=1, alpha=0.7, label='Filtered Signal')

        if self.integrated_signal is not None:
            # 归一化积分信号以便在同一图中显示
            integrated_norm = self.integrated_signal[start_idx:end_idx]
            integrated_norm = (integrated_norm - np.min(integrated_norm)) / (np.max(integrated_norm) - np.min(integrated_norm))
            integrated_norm = integrated_norm * (np.max(self.filtered_signal[start_idx:end_idx]) - np.min(self.filtered_signal[start_idx:end_idx])) + np.min(self.filtered_signal[start_idx:end_idx])
            ax2.plot(time_axis, integrated_norm, 'r-', linewidth=1, alpha=0.7, label='Integrated Signal (Normalized)')

        ax2.set_title('Filtered and Integrated Signals', fontsize=12)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 统计信息
        ax3 = axes[2]
        ax3.axis('off')

        # 计算统计信息
        stats_text = self._generate_statistics_text()

        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        plt.tight_layout()
        plt.show()

    def _generate_statistics_text(self):
        """
        生成统计信息文本
        """
        stats_text = f"""📊 ECG Feature Detection Statistics

R Waves Detected: {len(self.r_peaks)}
Q Points Detected: {len(self.q_points)}
S Points Detected: {len(self.s_points)}
P Peaks Detected: {len(self.p_peaks)}
P Onsets Detected: {len(self.p_onsets)}
P Ends Detected: {len(self.p_ends)}
T Peaks Detected: {len(self.t_peaks)}
T Ends Detected: {len(self.t_ends)}

Heart Rate Analysis:"""

        if len(self.r_peaks) > 1:
            rr_intervals = np.diff(self.r_peaks) * 1000 / self.fs
            heart_rate = 60000 / np.mean(rr_intervals)

            stats_text += f"""
  Average Heart Rate: {heart_rate:.1f} bpm
  RR Interval Mean: {np.mean(rr_intervals):.1f} ms
  RR Interval Std: {np.std(rr_intervals):.1f} ms
  RR Interval Range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms"""

        stats_text += f"""

Algorithm Parameters:
  Sampling Rate: {self.fs} Hz
  QRS Detection: Pan-Tomkins (5-40 Hz)
  Filter Order: 3rd order Butterworth
  Integration Window: 80 ms ({int(0.080 * self.fs)} samples)
  Refractory Period: 200 ms (supports 300bpm)
  Learning Phase: 5 beats
  Searchback Threshold: 25%
  Dynamic Decay: 1% per sample (when >1s without peaks)
  P/T Wave Detection: Low-pass filter (8-10 Hz)
  Search Windows: QRS={self.qrs_window} samples, P={self.p_window} samples, T={self.t_window} samples

Detection Features:
✓ R-wave detection using Pan-Tomkins algorithm
✓ Q/S point detection around R-peaks
✓ P-wave detection in pre-R interval
✓ T-wave detection in post-QRS interval
✓ Automatic onset and offset detection
✓ Real-time capable implementation"""

        return stats_text


def main():
    """
    主函数：读取ECG数据并应用综合特征检测算法
    """
    print("=" * 60)
    print("Comprehensive ECG Feature Detector")
    print("Based on Pan-Tomkins Algorithm for QRS Detection")
    print("=" * 60)

    # 创建检测器实例
    detector = ComprehensiveECGDetector(fs=360)

    # 读取数据文件
    data_path = 'mit-bih-dataset/ecg_' + num_seq + '.txt'
    print(f"\n读取ECG数据: {data_path}")

    # 读取数据
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            if '→' in line:
                numeric_part = line.split('→')[1].strip()
            else:
                numeric_part = line.strip()

            if numeric_part:
                parts = numeric_part.split()
                if len(parts) >= 2:
                    data.append([float(parts[0]), float(parts[1])])

    data = np.array(data)
    signal1 = data[:, 0]
    signal2 = data[:, 1]

    print(f"数据加载完成: {data.shape}")

    # 检测第一列信号
    print("\n" + "=" * 40)
    print("分析第一列ECG信号...")
    print("=" * 40)

    features1 = detector.detect_all_features(signal1)

    print(f"\n第一列信号检测结果:")
    print(f"  R波: {len(features1['r_peaks'])} 个")
    print(f"  Q点: {len(features1['q_points'])} 个")
    print(f"  S点: {len(features1['s_points'])} 个")
    print(f"  P波峰值: {len(features1['p_peaks'])} 个")
    print(f"  P波起始: {len(features1['p_onsets'])} 个")
    print(f"  P波结束: {len(features1['p_ends'])} 个")
    print(f"  T波峰值: {len(features1['t_peaks'])} 个")
    print(f"  T波结束: {len(features1['t_ends'])} 个")

    # 绘制第一列信号结果
    print("\n绘制第一列信号的详细ECG特征检测结果...")
    detector.plot_detailed_ecg(signal1, start_idx=0, num_samples=4000)

    # 检测第二列信号
    print("\n" + "=" * 40)
    print("分析第二列ECG信号...")
    print("=" * 40)

    detector2 = ComprehensiveECGDetector(fs=360)
    features2 = detector2.detect_all_features(signal2)

    print(f"\n第二列信号检测结果:")
    print(f"  R波: {len(features2['r_peaks'])} 个")
    print(f"  Q点: {len(features2['q_points'])} 个")
    print(f"  S点: {len(features2['s_points'])} 个")
    print(f"  P波峰值: {len(features2['p_peaks'])} 个")
    print(f"  P波起始: {len(features2['p_onsets'])} 个")
    print(f"  P波结束: {len(features2['p_ends'])} 个")
    print(f"  T波峰值: {len(features2['t_peaks'])} 个")
    print(f"  T波结束: {len(features2['t_ends'])} 个")

    # 绘制第二列信号结果
    print("\n绘制第二列信号的详细ECG特征检测结果...")
    detector2.plot_detailed_ecg(signal2, start_idx=0, num_samples=4000)

    print("\n" + "=" * 60)
    print("ECG特征检测完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()