import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from pan_tomkins_qrs import PanTomkinsQRSDetector

# 设置字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class ECGFullAnalyzer:
    """
    完整的ECG信号分析器
    在QRS检测基础上，实现P波、T波检测和各项指标测量
    """

    def __init__(self, fs=360):
        """
        初始化ECG分析器

        参数:
            fs: 采样频率 (Hz)
        """
        self.fs = fs
        self.qrs_detector = PanTomkinsQRSDetector(fs)

        # 存储检测结果
        self.r_peaks = []
        self.q_waves = []
        self.s_waves = []
        self.p_waves = []
        self.t_waves = []

        # 存储测量结果
        self.pr_intervals = []
        self.qrs_durations = []
        self.qt_intervals = []
        self.rr_intervals = []
        self.hrv_metrics = {}

    def analyze_ecg(self, signal_data):
        """
        完整分析ECG信号

        参数:
            signal_data: 输入ECG信号

        返回:
            analysis_results: 包含所有分析结果的字典
        """
        print("开始完整ECG分析...")

        # 步骤1: R波检测 (使用Pan-Tomkins算法)
        print("1. 检测R波...")
        self.r_peaks = self.qrs_detector.detect_qrs_peaks(signal_data)
        print(f"   检测到 {len(self.r_peaks)} 个R波")

        if len(self.r_peaks) < 2:
            print("错误：检测到的R波数量不足，无法进行完整分析")
            return None

        # 步骤2: QRS波群边界检测 (Q波和S波)
        print("2. 检测QRS波群边界...")
        self._detect_qrs_boundaries(signal_data)

        # 步骤3: P波检测
        print("3. 检测P波...")
        self._detect_p_waves(signal_data)

        # 步骤4: T波检测
        print("4. 检测T波...")
        self._detect_t_waves(signal_data)

        # 步骤5: 测量各项间期
        print("5. 测量关键间期...")
        self._measure_intervals(signal_data)

        # 步骤6: 心率变异性分析
        print("6. 进行心率变异性分析...")
        self._analyze_hrv()

        # 生成分析结果
        analysis_results = {
            'r_peaks': self.r_peaks,
            'q_waves': self.q_waves,
            's_waves': self.s_waves,
            'p_waves': self.p_waves,
            't_waves': self.t_waves,
            'pr_intervals': self.pr_intervals,
            'qrs_durations': self.qrs_durations,
            'qt_intervals': self.qt_intervals,
            'rr_intervals': self.rr_intervals,
            'hrv_metrics': self.hrv_metrics
        }

        print("ECG分析完成!")
        return analysis_results

    def _detect_qrs_boundaries(self, signal_data):
        """
        检测QRS波群的Q波和S波边界
        """
        self.q_waves = []
        self.s_waves = []

        # QRS窗口参数 (基于典型ECG波形)
        qrs_window_left = int(0.05 * self.fs)  # R波前50ms
        qrs_window_right = int(0.05 * self.fs)  # R波后50ms

        for r_peak in self.r_peaks:
            # 检测Q波 (R波前的负向偏转)
            q_start = max(0, r_peak - qrs_window_left)
            q_end = r_peak
            q_segment = signal_data[q_start:q_end]

            if len(q_segment) > 0:
                q_idx = np.argmin(q_segment) + q_start
                self.q_waves.append(q_idx)
            else:
                self.q_waves.append(r_peak)

            # 检测S波 (R波后的负向偏转)
            s_start = r_peak
            s_end = min(len(signal_data), r_peak + qrs_window_right)
            s_segment = signal_data[s_start:s_end]

            if len(s_segment) > 0:
                s_idx = np.argmin(s_segment) + s_start
                self.s_waves.append(s_idx)
            else:
                self.s_waves.append(r_peak)

    def _detect_p_waves(self, signal_data):
        """
        检测P波 (QRS前的正向偏转)
        """
        self.p_waves = []

        # P波搜索窗口
        p_search_start = int(0.2 * self.fs)  # R波前200ms
        p_search_end = int(0.05 * self.fs)   # R波前50ms

        for i, r_peak in enumerate(self.r_peaks):
            if i == 0:
                continue

            prev_r_peak = self.r_peaks[i-1]
            midpoint = (prev_r_peak + r_peak) // 2

            # P波在前一个R波之后，当前R波之前
            p_start = max(prev_r_peak + int(0.1 * self.fs),
                         r_peak - p_search_start)
            p_end = r_peak - p_search_end

            if p_end > p_start:
                # 低通滤波后寻找P波
                p_segment = signal_data[p_start:p_end]
                if len(p_segment) > 0:
                    # P波通常是正向波，寻找最大值
                    p_idx = np.argmax(p_segment) + p_start
                    self.p_waves.append(p_idx)
                else:
                    self.p_waves.append(midpoint)
            else:
                self.p_waves.append(midpoint)

    def _detect_t_waves(self, signal_data):
        """
        检测T波 (QRS后的正向偏转)
        """
        self.t_waves = []

        # T波搜索窗口
        t_search_start = int(0.05 * self.fs)  # R波后50ms
        t_search_end = int(0.3 * self.fs)    # R波后300ms

        for i, r_peak in enumerate(self.r_peaks):
            if i >= len(self.r_peaks) - 1:
                continue

            next_r_peak = self.r_peaks[i+1]

            # T波在当前R波之后，下一个R波之前
            t_start = r_peak + t_search_start
            t_end = min(r_peak + t_search_end, next_r_peak - int(0.1 * self.fs))

            if t_end > t_start:
                # T波通常是双向波，寻找绝对值最大值
                t_segment = signal_data[t_start:t_end]
                if len(t_segment) > 0:
                    t_idx = np.argmax(np.abs(t_segment - np.mean(t_segment))) + t_start
                    self.t_waves.append(t_idx)
                else:
                    self.t_waves.append(r_peak + int(0.2 * self.fs))
            else:
                self.t_waves.append(r_peak + int(0.2 * self.fs))

    def _measure_intervals(self, signal_data):
        """
        测量关键间期：PR间期、QRS间期、QT间期、RR间期
        """
        # RR间期 (相邻R波之间的时间)
        self.rr_intervals = np.diff(self.r_peaks) * 1000 / self.fs

        # PR间期 (P波开始到QRS开始)
        self.pr_intervals = []
        for i in range(len(self.r_peaks)):
            if i < len(self.p_waves):
                pr_interval = (self.r_peaks[i] - self.p_waves[i]) * 1000 / self.fs
                # PR间期通常在120-200ms范围内
                if 50 < pr_interval < 300:  # 宽松的约束
                    self.pr_intervals.append(pr_interval)

        # QRS间期 (Q波开始到S波结束)
        self.qrs_durations = []
        for i in range(len(self.r_peaks)):
            if i < len(self.q_waves) and i < len(self.s_waves):
                qrs_duration = (self.s_waves[i] - self.q_waves[i]) * 1000 / self.fs
                # QRS间期通常在60-120ms范围内
                if 40 < qrs_duration < 200:  # 宽松的约束
                    self.qrs_durations.append(qrs_duration)

        # QT间期 (Q波开始到T波结束)
        self.qt_intervals = []
        for i in range(min(len(self.q_waves), len(self.t_waves))):
            qt_interval = (self.t_waves[i] - self.q_waves[i]) * 1000 / self.fs
            # QT间期通常在300-460ms范围内
            if 200 < qt_interval < 600:  # 宽松的约束
                self.qt_intervals.append(qt_interval)

    def _analyze_hrv(self):
        """
        心率变异性分析 (HRV)
        """
        if len(self.rr_intervals) < 2:
            return

        # 时域指标
        self.hrv_metrics = {
            'mean_rr': np.mean(self.rr_intervals),
            'std_rr': np.std(self.rr_intervals),
            'rmssd': self._calculate_rmssd(),
            'nn50': self._calculate_nn50(),
            'pnn50': self._calculate_pnn50(),
            'mean_heart_rate': 60000 / np.mean(self.rr_intervals),
            'std_heart_rate': np.std(60000 / self.rr_intervals)
        }

    def _calculate_rmssd(self):
        """计算连续RR间期差值的均方根"""
        if len(self.rr_intervals) < 2:
            return 0
        diff_rr = np.diff(self.rr_intervals)
        return np.sqrt(np.mean(diff_rr ** 2))

    def _calculate_nn50(self):
        """计算相差超过50ms的RR间期对数"""
        if len(self.rr_intervals) < 2:
            return 0
        diff_rr = np.abs(np.diff(self.rr_intervals))
        return np.sum(diff_rr > 50)

    def _calculate_pnn50(self):
        """计算NN50占总RR间期差值的百分比"""
        nn50 = self._calculate_nn50()
        total_diff = len(self.rr_intervals) - 1
        return (nn50 / total_diff * 100) if total_diff > 0 else 0

    def generate_report(self):
        """
        生成ECG分析报告
        """
        print("\n" + "="*60)
        print("          ECG信号完整分析报告")
        print("="*60)

        print(f"\n📊 基本检测统计:")
        print(f"   R波数量: {len(self.r_peaks)}")
        print(f"   P波数量: {len(self.p_waves)}")
        print(f"   T波数量: {len(self.t_waves)}")
        print(f"   Q波数量: {len(self.q_waves)}")
        print(f"   S波数量: {len(self.s_waves)}")

        print(f"\n❤️ 心率分析:")
        if self.hrv_metrics:
            print(f"   平均心率: {self.hrv_metrics['mean_heart_rate']:.1f} bpm")
            print(f"   心率标准差: {self.hrv_metrics['std_heart_rate']:.1f} bpm")

        print(f"\n📏 间期测量:")
        if len(self.pr_intervals) > 0:
            print(f"   PR间期: {np.mean(self.pr_intervals):.1f} ± {np.std(self.pr_intervals):.1f} ms")
            print(f"   PR间期范围: {np.min(self.pr_intervals):.1f} - {np.max(self.pr_intervals):.1f} ms")

        if len(self.qrs_durations) > 0:
            print(f"   QRS间期: {np.mean(self.qrs_durations):.1f} ± {np.std(self.qrs_durations):.1f} ms")
            print(f"   QRS间期范围: {np.min(self.qrs_durations):.1f} - {np.max(self.qrs_durations):.1f} ms")

        if len(self.qt_intervals) > 0:
            print(f"   QT间期: {np.mean(self.qt_intervals):.1f} ± {np.std(self.qt_intervals):.1f} ms")
            print(f"   QT间期范围: {np.min(self.qt_intervals):.1f} - {np.max(self.qt_intervals):.1f} ms")

        if len(self.rr_intervals) > 0:
            print(f"   RR间期: {np.mean(self.rr_intervals):.1f} ± {np.std(self.rr_intervals):.1f} ms")

        print(f"\n💓 心率变异性(HRV)分析:")
        if self.hrv_metrics:
            print(f"   平均RR间期: {self.hrv_metrics['mean_rr']:.1f} ms")
            print(f"   RR间期标准差(SDNN): {self.hrv_metrics['std_rr']:.1f} ms")
            print(f"   RMSSD: {self.hrv_metrics['rmssd']:.1f} ms")
            print(f"   NN50: {self.hrv_metrics['nn50']}")
            print(f"   pNN50: {self.hrv_metrics['pnn50']:.1f}%")

        print(f"\n⚠️ 参考正常范围:")
        print(f"   PR间期: 120-200 ms")
        print(f"   QRS间期: 60-120 ms")
        print(f"   QT间期: 300-460 ms (与心率相关)")
        print(f"   正常心率: 60-100 bpm")

        print("="*60)

    def plot_full_analysis(self, signal_data, start_idx=0, num_samples=2000):
        """
        绘制完整的ECG分析结果
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(2, 1, figsize=(16, 10))

        # 第一子图：原始信号和波形标记
        ax1 = axes[0]
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # 绘制原始信号
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='ECG信号')

        # 标记各个波形点
        for i, r_peak in enumerate(self.r_peaks):
            if start_idx <= r_peak < end_idx:
                ax1.plot(r_peak/self.fs, signal_data[r_peak], 'ro', markersize=8, label='R波' if i == 0 else "")

        for i, p_wave in enumerate(self.p_waves):
            if start_idx <= p_wave < end_idx:
                ax1.plot(p_wave/self.fs, signal_data[p_wave], 'go', markersize=6, label='P波' if i == 0 else "")

        for i, t_wave in enumerate(self.t_waves):
            if start_idx <= t_wave < end_idx:
                ax1.plot(t_wave/self.fs, signal_data[t_wave], 'mo', markersize=6, label='T波' if i == 0 else "")

        for i, q_wave in enumerate(self.q_waves):
            if start_idx <= q_wave < end_idx:
                ax1.plot(q_wave/self.fs, signal_data[q_wave], 'y^', markersize=5, label='Q波' if i == 0 else "")

        for i, s_wave in enumerate(self.s_waves):
            if start_idx <= s_wave < end_idx:
                ax1.plot(s_wave/self.fs, signal_data[s_wave], 'c^', markersize=5, label='S波' if i == 0 else "")

        ax1.set_title('ECG信号完整波形检测')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('幅度')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 第二子图：RR间期序列
        ax2 = axes[1]
        if len(self.rr_intervals) > 0:
            rr_time = np.cumsum(self.rr_intervals) / 1000  # 转换为秒
            ax2.plot(rr_time, self.rr_intervals, 'g-', linewidth=2, marker='o', markersize=4)
            ax2.set_title('RR间期序列 (心率变异性)')
            ax2.set_xlabel('时间 (s)')
            ax2.set_ylabel('RR间期 (ms)')
            ax2.grid(True, alpha=0.3)

            # 添加平均线
            if len(self.rr_intervals) > 0:
                mean_rr = np.mean(self.rr_intervals)
                ax2.axhline(y=mean_rr, color='r', linestyle='--', alpha=0.7, label=f'平均值: {mean_rr:.1f} ms')
                ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot_optimization_comparison(self, signal_data, start_idx=0, num_samples=3000):
        """
        绘制优化前后的对比结果
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # 时间轴
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # 1. 完整波形检测结果
        ax1 = axes[0, 0]
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='原始ECG')

        # 标记各个波形点
        for i, r_peak in enumerate(self.r_peaks):
            if start_idx <= r_peak < end_idx:
                ax1.plot(r_peak/self.fs, signal_data[r_peak], 'ro', markersize=8, label='R波' if i == 0 else "")

        for i, p_wave in enumerate(self.p_waves):
            if start_idx <= p_wave < end_idx:
                ax1.plot(p_wave/self.fs, signal_data[p_wave], 'go', markersize=6, label='P波' if i == 0 else "")

        for i, t_wave in enumerate(self.t_waves):
            if start_idx <= t_wave < end_idx:
                ax1.plot(t_wave/self.fs, signal_data[t_wave], 'mo', markersize=6, label='T波' if i == 0 else "")

        ax1.set_title('完整ECG波形检测结果')
        ax1.set_xlabel('时间 (s)')
        ax1.set_ylabel('幅度')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 2. 优化后的R波检测细节
        ax2 = axes[0, 1]
        ax2.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7)

        for i, r_peak in enumerate(self.r_peaks):
            if start_idx <= r_peak < end_idx:
                ax2.plot(r_peak/self.fs, signal_data[r_peak], 'ro', markersize=10)
                ax2.annotate(f'R{i+1}', (r_peak/self.fs, signal_data[r_peak]),
                            xytext=(5, 5), textcoords='offset points', fontsize=8, color='red')

        # 添加间期标记
        for i in range(len(self.r_peaks) - 1):
            if start_idx <= self.r_peaks[i] < end_idx and start_idx <= self.r_peaks[i+1] < end_idx:
                rr_interval = (self.r_peaks[i+1] - self.r_peaks[i]) / self.fs
                mid_point = (self.r_peaks[i] + self.r_peaks[i+1]) / 2 / self.fs
                ax2.annotate(f'{rr_interval:.2f}s', (mid_point, max(signal_data[start_idx:end_idx])*0.8),
                            ha='center', fontsize=8, color='green',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

        ax2.set_title('R波检测精度 (优化后)')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('幅度')
        ax2.grid(True, alpha=0.3)

        # 3. 间期测量结果
        ax3 = axes[1, 0]
        intervals_data = []

        if len(self.pr_intervals) > 0:
            intervals_data.append(f'PR间期: {np.mean(self.pr_intervals):.1f}±{np.std(self.pr_intervals):.1f} ms')

        if len(self.qrs_durations) > 0:
            intervals_data.append(f'QRS间期: {np.mean(self.qrs_durations):.1f}±{np.std(self.qrs_durations):.1f} ms')

        if len(self.qt_intervals) > 0:
            intervals_data.append(f'QT间期: {np.mean(self.qt_intervals):.1f}±{np.std(self.qt_intervals):.1f} ms')

        if len(self.rr_intervals) > 0:
            intervals_data.append(f'RR间期: {np.mean(self.rr_intervals):.1f}±{np.std(self.rr_intervals):.1f} ms')

        intervals_text = '\n'.join(intervals_data) if intervals_data else "间期测量失败"

        ax3.text(0.1, 0.9, f"📏 关键间期测量结果:\n\n{intervals_text}",
                transform=ax3.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 添加正常参考值
        reference_text = """正常参考范围:
PR间期: 120-200 ms
QRS间期: 60-120 ms
QT间期: 300-460 ms
RR间期: 600-1000 ms (60-100 bpm)"""

        ax3.text(0.6, 0.9, reference_text,
                transform=ax3.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

        ax3.set_title('间期测量结果与参考值')
        ax3.axis('off')

        # 4. HRV分析结果
        ax4 = axes[1, 1]
        if self.hrv_metrics:
            hrv_text = f"""💓 心率变异性分析:

平均心率: {self.hrv_metrics['mean_heart_rate']:.1f} bpm
心率标准差: {self.hrv_metrics['std_heart_rate']:.1f} bpm

时域HRV指标:
  SDNN: {self.hrv_metrics['std_rr']:.1f} ms
  RMSSD: {self.hrv_metrics['rmssd']:.1f} ms
  NN50: {self.hrv_metrics['nn50']}
  pNN50: {self.hrv_metrics['pnn50']:.1f}%"""
        else:
            hrv_text = "HRV分析失败"

        ax4.text(0.1, 0.9, hrv_text,
                transform=ax4.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        ax4.set_title('心率变异性(HRV)分析')
        ax4.axis('off')

        plt.suptitle('ECG信号完整分析 - 优化后的Pan-Tomkins算法', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


def main():
    """
    主函数：完整ECG分析示例
    """
    print("初始化完整ECG分析器...")

    # 创建分析器实例
    ecg_analyzer = ECGFullAnalyzer(fs=360)

    # 读取数据文件
    data_path = 'mit-bih-dataset/ecg_100.txt'

    print(f"读取ECG数据: {data_path}")

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

    # 分析第一列信号
    print(f"\n{'='*60}")
    print("分析第一列ECG信号")
    print(f"{'='*60}")

    results1 = ecg_analyzer.analyze_ecg(signal1)

    if results1:
        ecg_analyzer.generate_report()
        print(f"\n绘制第一列信号的完整分析结果...")
        ecg_analyzer.plot_full_analysis(signal1, start_idx=0, num_samples=3000)

        # 同时显示优化后的R波检测结果
        print(f"\n显示第一列信号的优化R波检测...")
        ecg_analyzer.qrs_detector.plot_enhanced_results(signal1, start_idx=0, num_samples=2000)

        # 显示完整的优化对比分析
        print(f"\n显示第一列信号的完整优化分析对比...")
        ecg_analyzer.plot_optimization_comparison(signal1, start_idx=0, num_samples=3000)

    # 分析第二列信号
    print(f"\n{'='*60}")
    print("分析第二列ECG信号")
    print(f"{'='*60}")

    ecg_analyzer2 = ECGFullAnalyzer(fs=360)
    results2 = ecg_analyzer2.analyze_ecg(signal2)

    if results2:
        ecg_analyzer2.generate_report()
        print(f"\n绘制第二列信号的完整分析结果...")
        ecg_analyzer2.plot_full_analysis(signal2, start_idx=0, num_samples=3000)

        # 同时显示优化后的R波检测结果
        print(f"\n显示第二列信号的优化R波检测...")
        ecg_analyzer2.qrs_detector.plot_enhanced_results(signal2, start_idx=0, num_samples=2000)

        # 显示完整的优化对比分析
        print(f"\n显示第二列信号的完整优化分析对比...")
        ecg_analyzer2.plot_optimization_comparison(signal2, start_idx=0, num_samples=3000)


if __name__ == "__main__":
    main()