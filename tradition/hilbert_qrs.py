import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import wfdb


# Set default font for better English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display
num_seq= '103'


class HilbertQRSDetector:
    """
    基于希尔伯特变换的QRS波检测器

    利用希尔伯特变换提取信号的包络，通过包络的峰值检测QRS波，
    对基线漂移不敏感，计算简单且实时性好
    """

    def __init__(self, fs=360):
        """
        初始化希尔伯特QRS检测器

        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        self.signal = None
        self.filtered_signal = None
        self.hilbert_envelope = None
        self.analytic_signal = None
        self.qrs_peaks = []

    def bandpass_filter(self, signal_data):
        """
        优化的带通滤波器 (5-40 Hz)
        专门针对QRS波群的频率特性设计

        参数:
            signal_data: 输入ECG信号

        返回:
            filtered_signal: 滤波后的信号
        """
        # 设计带通滤波器 - 针对QRS波群优化频率范围
        nyquist = 0.5 * self.fs
        low = 5.0 / nyquist      # 低频截止，保留更多QRS信息
        high = 40.0 / nyquist    # 高频截止，抑制高频噪声

        # 使用3阶Butterworth滤波器
        b, a = scipy_signal.butter(3, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # 为了减少漏检，添加原始信号的加权
        original_weight = 0.2  # 原始信号权重
        filtered_weight = 0.8  # 滤波信号权重
        combined_signal = original_weight * signal_data + filtered_weight * filtered_signal

        return combined_signal

    def compute_hilbert_envelope(self, signal_data):
        """
        计算希尔伯特包络

        参数:
            signal_data: 输入信号

        返回:
            hilbert_envelope: 希尔伯特包络
        """
        self.signal = signal_data
        self.filtered_signal = self.bandpass_filter(signal_data)

        # 计算解析信号
        self.analytic_signal = scipy_signal.hilbert(self.filtered_signal)

        # 计算包络
        self.hilbert_envelope = np.abs(self.analytic_signal)

        # 平滑包络以减少噪声影响
        window_size = int(0.01 * self.fs)  # 10ms平滑窗口
        if window_size > 1:
            window = np.ones(window_size) / window_size
            self.hilbert_envelope = np.convolve(self.hilbert_envelope, window, mode='same')

        return self.hilbert_envelope

    def detect_qrs_peaks(self, signal_data):
        """
        使用希尔伯特变换检测QRS波
        使用自适应双阈值检测算法

        参数:
            signal_data: 输入ECG信号

        返回:
            qrs_peaks: QRS波峰值位置索引
        """
        # 计算希尔伯特包络
        self.compute_hilbert_envelope(signal_data)

        # 阈值检测QRS波
        self.qrs_peaks = self._threshold_detection()

        return self.qrs_peaks

    def _threshold_detection(self):
        """
        自适应双阈值检测算法
        使用希尔伯特包络检测QRS波，包含初始化阶段和精确定位

        返回:
            peaks: 检测到的峰值位置
        """
        # 初始化阶段 - 使用前2秒信号建立初始阈值
        init_samples = int(2 * self.fs)
        if len(self.hilbert_envelope) < init_samples:
            init_samples = len(self.hilbert_envelope)

        init_envelope = self.hilbert_envelope[:init_samples]
        # 使用保守的初始阈值，对小R波更敏感
        init_threshold = np.mean(init_envelope) + 2.0 * np.std(init_envelope)

        # 噪声和信号阈值初始化
        signal_peak = init_threshold
        noise_peak = np.mean(init_envelope)
        threshold = init_threshold

        # 不应期参数
        rr_interval_min = int(0.2 * self.fs)   # 200ms (支持300bpm)
        rr_interval_max = int(2.0 * self.fs)   # 2000ms (30bpm下限)

        peaks = []
        searchback_threshold = 0.3  # 回溯搜索阈值

        # 初始化标志 - 前几个心跳用于学习
        learning_beats = 5
        learning_count = 0

        for i in range(len(self.hilbert_envelope)):
            current_value = self.hilbert_envelope[i]

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
                        learning_factor = 0.4
                        learning_count += 1
                    else:
                        learning_factor = 0.1  # 稳定后使用较小学习率

                    signal_peak = learning_factor * current_value + (1 - learning_factor) * signal_peak
                else:
                    # 在不应期内，视为噪声
                    noise_peak = 0.25 * current_value + 0.75 * noise_peak

                # 动态调整阈值
                if learning_count < learning_beats:
                    # 学习阶段：更积极的阈值调整
                    threshold_factor = 0.3
                else:
                    # 稳定阶段：保守但仍保持敏感性
                    threshold_factor = 0.25

                threshold = noise_peak + threshold_factor * (signal_peak - noise_peak)
            else:
                # 更新噪声峰值
                if current_value > noise_peak:
                    noise_peak = 0.1 * current_value + 0.9 * noise_peak

                # 在长时间没有检测到峰值时，逐渐降低阈值
                if len(peaks) > 0 and (i - peaks[-1]) > int(1.0 * self.fs):  # 超过1秒无峰值
                    threshold *= 0.995  # 每个样本降低阈值0.5%

        # 最终回溯搜索 - 检查最后一个长间隔
        if len(peaks) > 0 and (len(self.hilbert_envelope) - peaks[-1]) > rr_interval_max * 0.8:
            missed_peaks = self._searchback_detection(peaks[-1], len(self.hilbert_envelope),
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
        refined_peaks = self._refine_peak_locations(all_peaks)

        return refined_peaks

    def _searchback_detection(self, start_idx, end_idx, threshold):
        """
        回溯搜索检测遗漏的QRS波
        """
        search_start = start_idx + int(0.15 * self.fs)
        search_end = min(end_idx, start_idx + int(1.5 * self.fs))

        if search_start >= search_end:
            return []

        search_segment = self.hilbert_envelope[search_start:search_end]
        if len(search_segment) == 0:
            return []

        # 寻找局部最大值
        peaks = []
        min_peak_distance = int(0.25 * self.fs)
        local_threshold = threshold * 0.7

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

                    if len(local_window) > 0 and search_segment[i] > np.mean(local_window) * 1.3:
                        peaks.append(peak_idx)

        return peaks

    def _refine_peak_locations(self, peak_indices):
        """
        将包络上的峰值位置精确定位到原始ECG信号的R波峰值
        """
        refined_peaks = []

        for peak_idx in peak_indices:
            # 在滤波信号上搜索R波峰值
            search_window = int(0.04 * self.fs)  # ±40ms搜索窗口
            search_start = max(0, peak_idx - search_window)
            search_end = min(len(self.filtered_signal),
                           peak_idx + search_window)

            if search_start < search_end:
                search_segment = self.filtered_signal[search_start:search_end]
                if len(search_segment) > 0:
                    # 寻找绝对值最大值（R波可能是正或负）
                    local_max_idx = np.argmax(np.abs(search_segment)) + search_start
                    refined_peaks.append(local_max_idx)
                else:
                    refined_peaks.append(peak_idx)
            else:
                refined_peaks.append(peak_idx)

        return refined_peaks

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

    def plot_results(self, signal_data, start_idx=0, num_samples=1000):
        """
        绘制QRS检测结果

        参数:
            signal_data: 原始ECG信号
            start_idx: 起始索引
            num_samples: 显示的样本数
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(4, 1, figsize=(15, 10))

        # 原始信号
        axes[0].plot(signal_data[start_idx:end_idx], 'b-', linewidth=1)
        axes[0].set_title('Original ECG Signal')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)

        # 滤波后信号
        if self.filtered_signal is not None:
            axes[1].plot(self.filtered_signal[start_idx:end_idx], 'g-', linewidth=1)
            axes[1].set_title('Bandpass Filtered Signal (5-40 Hz)')
            axes[1].set_ylabel('Amplitude')
            axes[1].grid(True, alpha=0.3)

        # 希尔伯特包络
        if self.hilbert_envelope is not None:
            axes[2].plot(self.hilbert_envelope[start_idx:end_idx], 'm-', linewidth=1.5, label='Hilbert Envelope')
            axes[2].set_title('Hilbert Envelope Signal')
            axes[2].set_ylabel('Envelope Amplitude')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        # 希尔伯特包络和QRS检测
        if self.hilbert_envelope is not None:
            axes[3].plot(self.hilbert_envelope[start_idx:end_idx], 'c-', linewidth=1.5, label='Envelope Signal')

            # 标记检测到的QRS波
            for peak in self.qrs_peaks:
                if start_idx <= peak < end_idx:
                    axes[3].plot(peak - start_idx, self.hilbert_envelope[peak], 'ro',
                               markersize=8, label='QRS Detection')

            axes[3].set_title('Hilbert Envelope Signal and QRS Detection Results')
            axes[3].set_xlabel('Sample Index')
            axes[3].set_ylabel('Envelope Amplitude')
            axes[3].legend()
            axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def plot_enhanced_results(self, signal_data, start_idx=0, num_samples=2000):
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
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='Original ECG')

        # 标记检测到的R波
        for i, peak in enumerate(self.qrs_peaks):
            if start_idx <= peak < end_idx:
                ax1.plot(peak/self.fs, signal_data[peak], 'ro', markersize=8, label='Detected R-wave' if i == 0 else "")
                # 添加R波编号
                ax1.annotate(f'R{i+1}', (peak/self.fs, signal_data[peak]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_title('Hilbert Transform - R-wave Detection Results')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 滤波后信号
        ax2 = axes[0, 1]
        if self.filtered_signal is not None:
            ax2.plot(time_axis, self.filtered_signal[start_idx:end_idx], 'g-', linewidth=1, label='Filtered Signal (5-40 Hz)')
            ax2.set_title('Bandpass Filtered Signal')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        # 3. 希尔伯特包络
        ax3 = axes[1, 0]
        if self.hilbert_envelope is not None:
            ax3.plot(time_axis, self.hilbert_envelope[start_idx:end_idx], 'c-', linewidth=1.5, label='Hilbert Envelope')

            # 标记包络上的检测点
            for peak in self.qrs_peaks:
                if start_idx <= peak < end_idx:
                    ax3.plot(peak/self.fs, self.hilbert_envelope[peak], 'ro', markersize=6)

            ax3.set_title('Hilbert Envelope Signal')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Envelope Amplitude')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. 解析信号（实部和虚部）
        ax4 = axes[1, 1]
        if self.analytic_signal is not None:
            analytic_segment = self.analytic_signal[start_idx:end_idx]
            ax4.plot(time_axis, np.real(analytic_segment), 'b-', linewidth=1, alpha=0.7, label='Real Part')
            ax4.plot(time_axis, np.imag(analytic_segment), 'r-', linewidth=1, alpha=0.7, label='Imaginary Part')
            ax4.set_title('Analytic Signal (Hilbert Transform)')
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

            stats_text = f"""📊 Detection Statistics

R-waves Detected: {len(self.qrs_peaks)}
Average Heart Rate: {heart_rate:.1f} bpm

RR Interval Statistics:
  Mean: {np.mean(rr_intervals):.1f} ms
  Std Dev: {np.std(rr_intervals):.1f} ms
  Range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms

Algorithm Parameters:
  Filter Band: 5-40 Hz
  Envelope Smoothing: 10 ms window
  Refractory Period: 200 ms
  Learning Phase: 5 beats
  Searchback Threshold: 30%

Algorithm Features:
✓ Insensitive to baseline drift
✓ Envelope analysis highlights QRS features
✓ Simple computation, good real-time performance
"""
        else:
            stats_text = "❌ Detection Failed\nInsufficient R-waves"

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.show()


def main():
    """
    主函数：读取ECG数据并应用希尔伯特变换算法
    """
    print("初始化希尔伯特变换QRS检测器...")

    # 读取数据文件
    data_path = 'mit-bih-dataset/ecg_'+ num_seq + '.txt'

    print(f"读取ECG数据: {data_path}")

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

    # 分离第一列和第二列信号
    signal1 = data[:, 0]
    signal2 = data[:, 1]

    print(f"数据加载完成: {data.shape}")

    # 对第一列信号进行QRS检测
    print("\n对第一列信号进行QRS检测...")
    qrs_detector1 = HilbertQRSDetector(fs=360)
    qrs_peaks1 = qrs_detector1.detect_qrs_peaks(signal1)
    heart_rate1, rr_intervals1 = qrs_detector1.calculate_heart_rate()

    print(f"第一列信号检测到 {len(qrs_peaks1)} 个QRS波")
    print(f"平均心率: {heart_rate1:.1f} bpm")

    # 对第二列信号进行QRS检测
    print("\n对第二列信号进行QRS检测...")
    qrs_detector2 = HilbertQRSDetector(fs=360)
    qrs_peaks2 = qrs_detector2.detect_qrs_peaks(signal2)
    heart_rate2, rr_intervals2 = qrs_detector2.calculate_heart_rate()

    print(f"第二列信号检测到 {len(qrs_peaks2)} 个QRS波")
    print(f"平均心率: {heart_rate2:.1f} bpm")

    # 绘制结果
    print("\n绘制第一列信号的QRS检测结果...")
    qrs_detector1.plot_enhanced_results(signal1, start_idx=0, num_samples=3000)

    print("\n绘制第二列信号的QRS检测结果...")
    qrs_detector2.plot_enhanced_results(signal2, start_idx=0, num_samples=3000)

    # 打印统计信息
    print("\n=== QRS检测统计信息 ===")
    print(f"信号1 - QRS波数量: {len(qrs_peaks1)}")
    print(f"信号1 - 平均心率: {heart_rate1:.2f} bpm")
    if len(rr_intervals1) > 0:
        print(f"信号1 - R-R间期均值: {np.mean(rr_intervals1):.2f} ms")
        print(f"信号1 - R-R间期标准差: {np.std(rr_intervals1):.2f} ms")

    print(f"\n信号2 - QRS波数量: {len(qrs_peaks2)}")
    print(f"信号2 - 平均心率: {heart_rate2:.2f} bpm")
    if len(rr_intervals2) > 0:
        print(f"信号2 - R-R间期均值: {np.mean(rr_intervals2):.2f} ms")
        print(f"信号2 - R-R间期标准差: {np.std(rr_intervals2):.2f} ms")


def printnum():
    folder = 'mit-bih-arrhythmia-dataset/'
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(folder + num_seq, 'atr')
    for key in annotation.__dict__:
        print(key, ":", annotation.__dict__[key])
        if type(annotation.__dict__[key]) == np.ndarray:
            print(annotation.__dict__[key].shape)
    Rlocation = annotation.sample  # 对应位置
    print(Rlocation)
    Rclass = annotation.symbol  # 对应标签
    print(Rclass)
    return


if __name__ == "__main__":
    main()

    # printnum()