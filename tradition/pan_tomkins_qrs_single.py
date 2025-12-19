import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import wfdb
from tqdm import tqdm
from QRSDetector.Method import PanTomkinsQRSDetectorOffline

# Set default font for better English display
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # Correct minus sign display
root = "/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/"
numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
             '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
             '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
             '231', '232', '233', '234']
# root = "/home/yogsothoth/DataSet/european-st-t-database-1.0.0/"
# numberSet = ['e0103', 'e0104', 'e0105', 'e0106', 'e0107', 'e0108', 'e0110', 'e0111', 'e0112', 'e0113', 'e0114', 'e0115',
#              'e0116', 'e0118', 'e0119', 'e0121', 'e0122', 'e0123', 'e0124', 'e0125', 'e0126', 'e0127', 'e0129', 'e0133',
#              'e0136', 'e0139', 'e0147', 'e0148', 'e0151', 'e0154', 'e0155', 'e0159', 'e0161', 'e0162', 'e0163', 'e0166',
#              'e0170', 'e0202', 'e0203', 'e0204', 'e0205', 'e0206', 'e0207', 'e0208', 'e0210', 'e0211', 'e0212', 'e0213',
#              'e0302', 'e0303', 'e0304', 'e0305', 'e0306', 'e0403', 'e0404', 'e0405', 'e0406', 'e0408', 'e0409', 'e0410',
#              'e0411', 'e0413', 'e0415', 'e0417', 'e0418', 'e0501', 'e0509', 'e0515', 'e0601', 'e0602', 'e0603', 'e0604',
#              'e0605', 'e0606', 'e0607', 'e0609', 'e0610', 'e0611', 'e0612', 'e0613', 'e0614', 'e0615', 'e0704', 'e0801',
#              'e0808', 'e0817', 'e0818', 'e1301', 'e1302', 'e1304']

class PanTomkinsQRSDetector(PanTomkinsQRSDetectorOffline):
    """
    基于Pan-Tomkins算法的QRS波检测器
    """

    def __init__(self, signal_name="MLII"):
        super().__init__(signal_name=signal_name)

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

    def compare_with_annotations(self, annotation_samples, tolerance_ms=50):
        """
        比较检测结果与标注数据，输出详细统计结果

        参数:
            annotation_samples: 标注数据位置索引数组
            tolerance_ms: 匹配容差（毫秒），默认50ms

        返回:
            stats_dict: 包含各种统计指标的字典
        """
        if len(self.qrs_peaks) == 0 or len(annotation_samples) == 0:
            return {
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0,
                'sensitivity': 0,
                'positive_predictive_value': 0,
                'detection_rate': 0,
                'total_annotations': len(annotation_samples),
                'total_detections': len(self.qrs_peaks),
                'matched_pairs': []
            }

        # 转换容差为样本数
        tolerance_samples = int(tolerance_ms * self.fs / 1000)

        # 寻找匹配的检测点
        matched_annotations = set()
        matched_detections = set()
        matched_pairs = []

        for i, ann_sample in enumerate(annotation_samples):
            # 寻找距离当前标注最近的检测点
            best_match = None
            min_distance = float('inf')

            for j, det_sample in enumerate(self.qrs_peaks):
                if j in matched_detections:  # 已匹配的检测点跳过
                    continue

                distance = abs(ann_sample - det_sample)
                if distance <= tolerance_samples and distance < min_distance:
                    best_match = j
                    min_distance = distance

            # 如果找到匹配，记录下来
            if best_match is not None:
                matched_annotations.add(i)
                matched_detections.add(best_match)
                matched_pairs.append({
                    'annotation_idx': i,
                    'annotation_sample': ann_sample,
                    'detection_idx': best_match,
                    'detection_sample': self.qrs_peaks[best_match],
                    'time_diff_ms': min_distance * 1000 / self.fs
                })

        # 计算各种统计指标
        true_positives = len(matched_annotations)
        false_positives = len(self.qrs_peaks) - len(matched_detections)
        false_negatives = len(annotation_samples) - len(matched_annotations)

        # 计算敏感度（召回率）
        sensitivity = true_positives / len(annotation_samples) if len(annotation_samples) > 0 else 0

        # 计算阳性预测值（精确率）
        positive_predictive_value = true_positives / len(self.qrs_peaks) if len(self.qrs_peaks) > 0 else 0

        # 计算检测率
        detection_rate = (true_positives / len(annotation_samples)) * 100 if len(annotation_samples) > 0 else 0

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'sensitivity': sensitivity,
            'positive_predictive_value': positive_predictive_value,
            'detection_rate': detection_rate,
            'total_annotations': len(annotation_samples),
            'total_detections': len(self.qrs_peaks),
            'matched_pairs': matched_pairs,
            'tolerance_ms': tolerance_ms
        }

    def print_comparison_stats(self, stats_dict, signal_name="ECG"):
        """
        打印比较统计结果

        参数:
            stats_dict: compare_with_annotations返回的统计字典
            signal_name: 信号名称
        """
        print(f"\n=== {signal_name} QRS检测结果对比统计 ===")
        print(f"标注总数: {stats_dict['total_annotations']}")
        print(f"检测总数: {stats_dict['total_detections']}")
        print(f"正确检测 (True Positives): {stats_dict['true_positives']}")
        print(f"误检 (False Positives): {stats_dict['false_positives']}")
        print(f"漏检 (False Negatives): {stats_dict['false_negatives']}")
        print(f"检测率: {stats_dict['detection_rate']:.2f}%")
        print(f"敏感度 (Sensitivity/Recall): {stats_dict['sensitivity']:.4f}")
        print(f"阳性预测值 (Precision): {stats_dict['positive_predictive_value']:.4f}")
        print(f"匹配容差: {stats_dict['tolerance_ms']} ms")

        # 如果有匹配对，显示时间误差统计
        if stats_dict['matched_pairs']:
            time_diffs = [pair['time_diff_ms'] for pair in stats_dict['matched_pairs']]
            print(f"\n时间误差统计:")
            print(f"  平均误差: {np.mean(time_diffs):.2f} ms")
            print(f"  标准差: {np.std(time_diffs):.2f} ms")
            print(f"  最大误差: {np.max(time_diffs):.2f} ms")
            print(f"  最小误差: {np.min(time_diffs):.2f} ms")

        print("-" * 50)

    def plot_error_analysis(self, signal_data, annotation_samples, start_idx=0, num_samples=3000, tolerance_ms=150,
                            record_number=""):
        """
        绘制错误分析图，显示漏检和误检的具体例子

        参数:
            signal_data: 原始ECG信号
            annotation_samples: 标注数据位置索引数组
            start_idx: 起始索引（如果为0，会自动调整到第一个错误附近）
            num_samples: 显示的样本数
            tolerance_ms: 匹配容差（毫秒）
            record_number: 数据集记录编号
        """
        # 分析所有错误，找到第一个错误位置
        tolerance_samples = int(tolerance_ms * self.fs / 1000)
        all_false_negatives = []
        all_false_positives = []

        # 寻找所有的错误
        for ann_sample in annotation_samples:
            found_match = False
            for det_sample in self.qrs_peaks:
                if abs(ann_sample - det_sample) <= tolerance_samples:
                    found_match = True
                    break
            if not found_match:
                all_false_negatives.append(ann_sample)

        for det_sample in self.qrs_peaks:
            found_match = False
            for ann_sample in annotation_samples:
                if abs(ann_sample - det_sample) <= tolerance_samples:
                    found_match = True
                    break
            if not found_match:
                all_false_positives.append(det_sample)

        # 如果有错误且start_idx为0，调整到第一个错误附近
        if start_idx == 0 and (all_false_negatives or all_false_positives):
            first_error = min(all_false_negatives + all_false_positives)
            # 调整到第一个错误前2秒开始
            start_idx = max(0, first_error - int(2 * self.fs))
            num_samples = min(num_samples, len(signal_data) - start_idx)

        end_idx = min(start_idx + num_samples, len(signal_data))
        time_axis = np.arange(start_idx, end_idx) / self.fs

        fig, axes = plt.subplots(2, 1, figsize=(16, 8))

        # 在指定窗口内分析错误类型
        matched_annotations = set()
        matched_detections = set()
        false_negatives = []
        false_positives = []
        true_positives = []

        # 寻找窗口内的匹配
        for i, ann_sample in enumerate(annotation_samples):
            if not (start_idx <= ann_sample < end_idx):
                continue

            found_match = False
            best_match_idx = -1
            min_distance = float('inf')

            for j, det_sample in enumerate(self.qrs_peaks):
                if j in matched_detections:
                    continue

                distance = abs(ann_sample - det_sample)
                if distance <= tolerance_samples and distance < min_distance:
                    best_match_idx = j
                    min_distance = distance
                    found_match = True

            if found_match and best_match_idx != -1:
                matched_annotations.add(i)
                matched_detections.add(best_match_idx)
                true_positives.append((ann_sample, self.qrs_peaks[best_match_idx], min_distance))
            else:
                false_negatives.append(ann_sample)

        # 查找窗口内的误检
        for j, det_sample in enumerate(self.qrs_peaks):
            if j in matched_detections or not (start_idx <= det_sample < end_idx):
                continue

            found_match = False
            for ann_sample in annotation_samples:
                if abs(ann_sample - det_sample) <= tolerance_samples:
                    found_match = True
                    break

            if not found_match:
                false_positives.append(det_sample)

        # 第一个子图：显示所有检测和标注
        ax1 = axes[0]
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='Original ECG Signal')

        # 绘制标注
        for ann in annotation_samples:
            if start_idx <= ann < end_idx:
                ax1.axvline(x=ann / self.fs, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # 绘制检测结果
        for det in self.qrs_peaks:
            if start_idx <= det < end_idx:
                ax1.axvline(x=det / self.fs, color='red', linestyle=':', alpha=0.7, linewidth=2)

        # 添加图例
        ax1.plot([], [], 'g--', linewidth=2, label='Annotations')
        ax1.plot([], [], 'r:', linewidth=2, label='Detections')

        ax1.set_title(f'Detection vs Annotation Comparison (Tolerance: {tolerance_ms}ms, Record {record_number})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # 第二个子图：标记错误类型
        ax2 = axes[1]
        ax2.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='Original ECG Signal')

        # 标记正确检测
        for ann, det, diff in true_positives:
            ax2.plot(det / self.fs, signal_data[det], 'go', markersize=10,
                     label='True Positive' if len(true_positives) > 0 and (ann, det) == true_positives[0] else "")
            ax2.annotate(f'+{diff * 1000 / self.fs:.1f}ms', (det / self.fs, signal_data[det]),
                         xytext=(5, -15), textcoords='offset points', fontsize=7, color='green')

        # 标记漏检
        for fn in false_negatives:
            ax2.plot(fn / self.fs, signal_data[fn], 'rx', markersize=12, markeredgewidth=3,
                     label='False Negative' if len(false_negatives) > 0 and fn == false_negatives[0] else "")
            ax2.annotate('FN', (fn / self.fs, signal_data[fn]),
                         xytext=(5, -20), textcoords='offset points', fontsize=7,
                         color='red', weight='bold')

        # 标记误检
        for fp in false_positives:
            ax2.plot(fp / self.fs, signal_data[fp], 'r^', markersize=10,
                     label='False Positive' if len(false_positives) > 0 and fp == false_positives[0] else "")
            ax2.annotate('FP', (fp / self.fs, signal_data[fp]),
                         xytext=(5, 15), textcoords='offset points', fontsize=7,
                         color='red', weight='bold')

        ax2.set_title(
            f'Error Analysis: TP={len(true_positives)}, FN={len(false_negatives)}, FP={len(false_positives)} (Record {record_number})')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        return {
            'true_positives': len(true_positives),
            'false_negatives': len(false_negatives),
            'false_positives': len(false_positives),
            'fn_positions': false_negatives,
            'fp_positions': false_positives
        }

    def plot_results(self, signal_data, start_idx=0, num_samples=2000, annotation_samples=None, record_number=""):
        """
        绘制增强的QRS检测结果，包含详细的处理步骤可视化

        参数:
            signal_data: 原始ECG信号
            start_idx: 起始索引
            num_samples: 显示的样本数
            annotation_samples: 标注数据位置索引数组
            record_number: 数据集记录编号
        """
        end_idx = min(start_idx + num_samples, len(signal_data))

        fig, axes = plt.subplots(3, 2, figsize=(16, 10))

        # 时间轴
        time_axis = np.arange(start_idx, end_idx) / self.fs

        # 1. 原始信号和R波检测
        ax1 = axes[0, 0]
        ax1.plot(time_axis, signal_data[start_idx:end_idx], 'b-', linewidth=1, alpha=0.7, label='Original ECG')

        # 标记标注数据
        if annotation_samples is not None:
            for i, ann_sample in enumerate(annotation_samples):
                if start_idx <= ann_sample < end_idx:
                    ax1.plot(ann_sample / self.fs, signal_data[ann_sample], 'g*', markersize=10,
                             label='Annotation' if i == 0 else "")

        # 标记检测到的R波
        for i, peak in enumerate(self.qrs_peaks):
            if start_idx <= peak < end_idx:
                ax1.plot(peak / self.fs, signal_data[peak], 'ro', markersize=8,
                         label='Detection' if i == 0 else "")
                # 添加R波编号
                ax1.annotate(f'R{i + 1}', (peak / self.fs, signal_data[peak]),
                             xytext=(5, 5), textcoords='offset points', fontsize=8)

        ax1.set_title(f'R-wave Detection Results with Annotations (Record {record_number})')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. 带通滤波信号
        ax2 = axes[0, 1]
        if self.filtered_signal is not None:
            ax2.plot(time_axis, self.filtered_signal[start_idx:end_idx], 'g-', linewidth=1,
                     label='Filtered Signal')
            ax2.set_title('Bandpass Filtered Signal')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Amplitude')
            ax2.grid(True, alpha=0.3)

        # 3. 微分信号
        ax3 = axes[1, 0]
        if self.differentiated_signal is not None:
            ax3.plot(time_axis, self.differentiated_signal[start_idx:end_idx], 'r-', linewidth=1,
                     label='Differentiated Signal')
            ax3.set_title('Differentiated Signal')
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

            ax4.set_title('Moving Window Integration Signal')
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

            # 计算检测准确性（如果有标注数据）
            accuracy_text = ""
            if annotation_samples is not None:
                # 统计在显示范围内的标注和检测结果
                ann_in_range = [a for a in annotation_samples if start_idx <= a < end_idx]
                detected_in_range = [d for d in self.qrs_peaks if start_idx <= d < end_idx]

                if len(ann_in_range) > 0:
                    # 简单的匹配检测（容差±50ms）
                    tolerance = int(0.05 * self.fs)  # 50ms容差
                    matches = 0
                    for ann in ann_in_range:
                        for det in detected_in_range:
                            if abs(ann - det) <= tolerance:
                                matches += 1
                                break

                    detection_rate = (matches / len(ann_in_range)) * 100
                    accuracy_text = f"""

Annotation Comparison:
  Annotations in view: {len(ann_in_range)}
  Detections in view: {len(detected_in_range)}
  Matches: {matches}
  Detection Rate: {detection_rate:.1f}%"""

            stats_text = f"""Detection Statistics

R-waves Detected: {len(self.qrs_peaks)}
Average Heart Rate: {heart_rate:.1f} bpm

RR Interval Statistics:
  Mean: {np.mean(rr_intervals):.1f} ms
  Std Dev: {np.std(rr_intervals):.1f} ms
  Range: {np.min(rr_intervals):.1f} - {np.max(rr_intervals):.1f} ms{accuracy_text}

Algorithm Parameters:
  Filter Band: Adaptive (per lead)
  Derivative: 5-point central difference
  Integration Window: 80 ms
  Refractory Period: Adaptive
  Signal-dependent threshold
"""
        else:
            stats_text = "Detection Failed\nInsufficient R-waves"

        ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

        plt.tight_layout()
        plt.show()


def debug_lead_parameters(target_lead="MLII", record_subset=None, tolerance_ms=50):
    """
    导联参数调试函数 - 循环测试不同参数配置

    参数:
        target_lead: 目标导联名称（如 "MLII", "V1", "V5" 等）
        record_subset: 要测试的记录子集，如果为None则使用所有可用记录
        tolerance_ms: 匹配容差（毫秒）
    """
    # 根据目标导联定义不同的参数配置组合
    if target_lead == "MLII":
        configs = [
            # 基准配置
            {'name': 'Baseline', 'low': 5, 'high': 25.0, 'threshold_factor': 1.4},
            # 低频变化测试
            {'name': 'LowFreq_3Hz', 'low': 3, 'high': 25.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_8Hz', 'low': 8, 'high': 25.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_10Hz', 'low': 10, 'high': 25.0, 'threshold_factor': 1.4},
            # 高频变化测试
            {'name': 'HighFreq_10Hz', 'low': 5, 'high': 25.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_20Hz', 'low': 5, 'high': 30.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_25Hz', 'low': 5, 'high': 35.0, 'threshold_factor': 1.4},
            # 阈值因子变化测试
            {'name': 'Threshold_1.2', 'low': 5, 'high': 25.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.6', 'low': 5, 'high': 25.0, 'threshold_factor': 1.6},
            {'name': 'Threshold_1.8', 'low': 5, 'high': 25.0, 'threshold_factor': 1.8},
            # 组合优化配置
            {'name': 'Config_A', 'low': 3, 'high': 35.0, 'threshold_factor': 1.3},
            {'name': 'Config_B', 'low': 8, 'high': 30.0, 'threshold_factor': 1.5},
            {'name': 'Config_C', 'low': 6, 'high': 25.0, 'threshold_factor': 1.2},
        ]
    elif target_lead == "MLIII":
        configs = [
            # 基准配置
            {'name': 'Baseline', 'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 低频变化测试
            {'name': 'LowFreq_3Hz', 'low': 3, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_8Hz', 'low': 8, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_10Hz', 'low': 10, 'high': 15.0, 'threshold_factor': 1.4},
            # 高频变化测试
            {'name': 'HighFreq_10Hz', 'low': 5, 'high': 10.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_20Hz', 'low': 5, 'high': 20.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_25Hz', 'low': 5, 'high': 25.0, 'threshold_factor': 1.4},
            # 阈值因子变化测试
            {'name': 'Threshold_1.2', 'low': 5, 'high': 15.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.6', 'low': 5, 'high': 15.0, 'threshold_factor': 1.6},
            {'name': 'Threshold_1.8', 'low': 5, 'high': 15.0, 'threshold_factor': 1.8},
            # 组合优化配置
            {'name': 'Config_A', 'low': 3, 'high': 20.0, 'threshold_factor': 1.3},
            {'name': 'Config_B', 'low': 8, 'high': 18.0, 'threshold_factor': 1.5},
            {'name': 'Config_C', 'low': 6, 'high': 12.0, 'threshold_factor': 1.2},
        ]
    elif target_lead == "V1":
        configs = [
            # V1导联特殊配置：需要更低频率和更高频率
            {'name': 'Baseline', 'low': 1, 'high': 50.0, 'threshold_factor': 1.2},
            {'name': 'LowFreq_0.5Hz', 'low': 0.5, 'high': 50.0, 'threshold_factor': 1.2},
            {'name': 'LowFreq_2Hz', 'low': 2, 'high': 50.0, 'threshold_factor': 1.2},
            {'name': 'HighFreq_40Hz', 'low': 1, 'high': 40.0, 'threshold_factor': 1.2},
            {'name': 'HighFreq_60Hz', 'low': 1, 'high': 60.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.1', 'low': 1, 'high': 50.0, 'threshold_factor': 1.1},
            {'name': 'Threshold_1.3', 'low': 1, 'high': 50.0, 'threshold_factor': 1.3},
            {'name': 'Config_V1_A', 'low': 0.8, 'high': 45.0, 'threshold_factor': 1.15},
            {'name': 'Config_V1_B', 'low': 1.5, 'high': 55.0, 'threshold_factor': 1.25},
        ]
    elif target_lead == "V2":
        configs = [
            # V2导联配置：过渡区参数
            {'name': 'Baseline', 'low': 3, 'high': 30.0, 'threshold_factor': 1.3},
            {'name': 'LowFreq_2Hz', 'low': 2, 'high': 30.0, 'threshold_factor': 1.3},
            {'name': 'LowFreq_5Hz', 'low': 5, 'high': 30.0, 'threshold_factor': 1.3},
            {'name': 'HighFreq_25Hz', 'low': 3, 'high': 25.0, 'threshold_factor': 1.3},
            {'name': 'HighFreq_35Hz', 'low': 3, 'high': 35.0, 'threshold_factor': 1.3},
            {'name': 'Threshold_1.2', 'low': 3, 'high': 30.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.4', 'low': 3, 'high': 30.0, 'threshold_factor': 1.4},
            {'name': 'Config_V2_A', 'low': 2.5, 'high': 32.0, 'threshold_factor': 1.25},
            {'name': 'Config_V2_B', 'low': 4, 'high': 28.0, 'threshold_factor': 1.35},
        ]
    elif target_lead == "V5":
        configs = []

        # 多层参数循环配置
        low_freq_range = [4, 5, 6]
        high_freq_range = [30, 35]
        threshold_range = [1.0, 1.2]
        config_count = 0
        for lf in low_freq_range:
            for hf in high_freq_range:
                for tf in threshold_range:
                    # 确保低频小于高频
                    if lf < hf:
                        config_name = f'L{lf}H{hf}T{tf}'
                        configs.append({
                            'name': config_name,
                            'low': lf,
                            'high': hf,
                            'threshold_factor': tf
                        })
                        config_count += 1

        print(f"V5导联参数组合: {config_count} 种配置")
        print(f"  低频范围: {low_freq_range} Hz")
        print(f"  高频范围: {high_freq_range} Hz")
        print(f"  阈值范围: {threshold_range}")
    elif target_lead in ["V3", "V4", "V6"]:
        configs = [
            # V3-V6导联配置：标准参数
            {'name': 'Baseline', 'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_3Hz', 'low': 3, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_8Hz', 'low': 8, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_12Hz', 'low': 5, 'high': 12.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_20Hz', 'low': 5, 'high': 20.0, 'threshold_factor': 1.4},
            {'name': 'Threshold_1.2', 'low': 5, 'high': 15.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.6', 'low': 5, 'high': 15.0, 'threshold_factor': 1.6},
            {'name': 'Config_V3_A', 'low': 4, 'high': 18.0, 'threshold_factor': 1.3},
            {'name': 'Config_V3_B', 'low': 7, 'high': 13.0, 'threshold_factor': 1.5},
        ]
    else:
        # 其他导联使用通用配置
        configs = [
            # 通用配置
            {'name': 'Baseline', 'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_3Hz', 'low': 3, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'LowFreq_8Hz', 'low': 8, 'high': 15.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_10Hz', 'low': 5, 'high': 10.0, 'threshold_factor': 1.4},
            {'name': 'HighFreq_20Hz', 'low': 5, 'high': 20.0, 'threshold_factor': 1.4},
            {'name': 'Threshold_1.2', 'low': 5, 'high': 15.0, 'threshold_factor': 1.2},
            {'name': 'Threshold_1.6', 'low': 5, 'high': 15.0, 'threshold_factor': 1.6},
            {'name': 'Config_A', 'low': 3, 'high': 20.0, 'threshold_factor': 1.3},
            {'name': 'Config_B', 'low': 8, 'high': 18.0, 'threshold_factor': 1.5},
        ]

    # 如果没有指定记录子集，使用所有包含目标导联的记录
    if record_subset is None:
        target_records = []
        for num in numberSet:
            try:
                input_data = wfdb.rdrecord(root + num)
                sig_name = input_data.sig_name
                if target_lead in sig_name:
                    target_records.append(num)
            except Exception as e:
                print(e)
                continue
        record_subset = target_records
    else:
        # 确保指定的记录存在且包含目标导联
        target_records = []
        for num in record_subset:
            try:
                input_data = wfdb.rdrecord(root + num)
                sig_name = input_data.sig_name
                if target_lead in sig_name:
                    target_records.append(num)
            except Exception as e:
                print(e)
                continue
        record_subset = target_records

    if not record_subset:
        print(f"错误：没有找到包含{target_lead}导联的记录")
        return

    print(f"{target_lead}导联参数调试开始...")
    print(f"测试记录: {record_subset}")
    print(f"参数配置数量: {len(configs)}")
    print("=" * 80)

    # 存储每个配置的总体结果
    config_results = []

    # 对每个配置进行测试
    for config in configs:
        print(f"\n测试配置: {config['name']}")
        print(f"参数: low={config['low']}Hz, high={config['high']}Hz, threshold_factor={config['threshold_factor']}")
        print("-" * 60)

        # 统计数据
        total_stats = {
            "tp": 0, "fp": 0, "fn": 0, "time_errors": [],
            "records_processed": 0, "total_annotations": 0, "total_detections": 0
        }

        # 处理每个记录
        for num in record_subset:
            try:
                # 加载数据
                signal = wfdb.rdrecord(root + num, channel_names=[target_lead]).p_signal.flatten()
                annotation = wfdb.rdann(root + num, 'atr')
                fs = annotation.fs
                # MIT-BIH标注从1开始，需要转换为0-based索引
                sig_sample = annotation.sample[1:]

                # 创建检测器并临时修改MLII参数
                qrs_detector = PanTomkinsQRSDetector(signal_name=target_lead)
                qrs_detector.fs = fs
                qrs_detector.params['low'] = config['low']
                qrs_detector.params['high'] = config['high']
                qrs_detector.params['threshold_factor'] = config['threshold_factor']

                # 进行检测
                _ = qrs_detector.detect_qrs_peaks(signal)

                # 统计结果
                stats = qrs_detector.compare_with_annotations(sig_sample, tolerance_ms=tolerance_ms)
                total_stats["tp"] += stats['true_positives']
                total_stats["fp"] += stats['false_positives']
                total_stats["fn"] += stats['false_negatives']
                total_stats["time_errors"].extend([p['time_diff_ms'] for p in stats['matched_pairs']])
                total_stats["records_processed"] += 1
                total_stats["total_annotations"] += stats['total_annotations']
                total_stats["total_detections"] += stats['total_detections']

                print(f"  记录 {num}: 标注={stats['total_annotations']}, 检测={stats['total_detections']}, "
                      f"TP={stats['true_positives']}, FP={stats['false_positives']}, FN={stats['false_negatives']}")

            except Exception as e:
                print(f"  记录 {num} 处理失败: {str(e)}")
                continue

        # 计算该配置的总体性能指标
        tp = total_stats["tp"]
        fp = total_stats["fp"]
        fn = total_stats["fn"]
        total_annotations = tp + fn
        total_detections = tp + fp

        if total_annotations > 0:
            sensitivity = tp / total_annotations
            detection_rate = sensitivity * 100
        else:
            sensitivity = 0
            detection_rate = 0

        if total_detections > 0:
            precision = tp / total_detections
        else:
            precision = 0

        if sensitivity + precision > 0:
            f1_score = 2 * (sensitivity * precision) / (sensitivity + precision)
        else:
            f1_score = 0

        # 计算时间误差统计
        if total_stats["time_errors"]:
            avg_time_error = np.mean(total_stats["time_errors"])
            std_time_error = np.std(total_stats["time_errors"])
        else:
            avg_time_error = 0
            std_time_error = 0

        # 保存配置结果
        config_result = {
            'name': config['name'],
            'params': config,
            'records_processed': total_stats["records_processed"],
            'total_annotations': total_annotations,
            'total_detections': total_detections,
            'tp': tp, 'fp': fp, 'fn': fn,
            'sensitivity': sensitivity,
            'precision': precision,
            'f1_score': f1_score,
            'detection_rate': detection_rate,
            'avg_time_error': avg_time_error,
            'std_time_error': std_time_error
        }
        config_results.append(config_result)

        print(f"\n配置 {config['name']} 总体结果:")
        print(f"  处理记录数: {total_stats['records_processed']}")
        print(f"  总标注数: {total_annotations}")
        print(f"  总检测数: {total_detections}")
        print(f"  检测率: {detection_rate:.2f}%")
        print(f"  敏感度: {sensitivity:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  F1分数: {f1_score:.4f}")
        print(f"  平均时间误差: {avg_time_error:.2f}±{std_time_error:.2f} ms")

    # 生成对比报告
    print(f"\n" + "=" * 80)
    print(f"{target_lead}导联参数调试对比报告")
    print("=" * 80)

    # 按F1分数排序
    config_results.sort(key=lambda x: x['f1_score'], reverse=True)

    print(f"\n配置性能排名 (按F1分数排序):")
    print("-" * 80)
    print(f"{'排名':<4} {'配置名称':<15} {'检测率(%)':<10} {'敏感度':<10} {'精确率':<10} {'F1分数':<10} {'参数':<25}")
    print("-" * 80)

    for i, result in enumerate(config_results, 1):
        params_str = f"L{result['params']['low']}-H{result['params']['high']}-T{result['params']['threshold_factor']}"
        print(f"{i:<4} {result['name']:<15} {result['detection_rate']:<10.2f} {result['sensitivity']:<10.4f} "
              f"{result['precision']:<10.4f} {result['f1_score']:<10.4f} {params_str:<25}")

    # 详细的最佳配置分析
    best_config = config_results[0]
    print(f"\n最佳配置详细分析: {best_config['name']}")
    print("=" * 50)
    print(f"{target_lead}导联参数设置:")
    print(f"  低频截止: {best_config['params']['low']} Hz")
    print(f"  高频截止: {best_config['params']['high']} Hz")
    print(f"  阈值因子: {best_config['params']['threshold_factor']}")
    print(f"\n性能指标:")
    print(f"  处理记录数: {best_config['records_processed']}")
    print(f"  总标注数: {best_config['total_annotations']}")
    print(f"  总检测数: {best_config['total_detections']}")
    print(f"  正确检测: {best_config['tp']}")
    print(f"  误检数: {best_config['fp']}")
    print(f"  漏检数: {best_config['fn']}")
    print(f"  检测率: {best_config['detection_rate']:.2f}%")
    print(f"  敏感度: {best_config['sensitivity']:.4f}")
    print(f"  精确率: {best_config['precision']:.4f}")
    print(f"  F1分数: {best_config['f1_score']:.4f}")
    print(f"  时间误差: {best_config['avg_time_error']:.2f}±{best_config['std_time_error']:.2f} ms")

    # 参数影响分析
    print(f"\n参数影响分析:")
    print("-" * 40)

    # 按低频分组分析
    low_freq_groups = {}
    for result in config_results:
        low = result['params']['low']
        if low not in low_freq_groups:
            low_freq_groups[low] = []
        low_freq_groups[low].append(result['f1_score'])

    print(f"低频截止频率影响:")
    for low, scores in sorted(low_freq_groups.items()):
        avg_score = np.mean(scores)
        print(f"  {low}Hz: 平均F1分数 {avg_score:.4f} (共{len(scores)}个配置)")

    # 按高频分组分析
    high_freq_groups = {}
    for result in config_results:
        high = result['params']['high']
        if high not in high_freq_groups:
            high_freq_groups[high] = []
        high_freq_groups[high].append(result['f1_score'])

    print(f"\n高频截止频率影响:")
    for high, scores in sorted(high_freq_groups.items()):
        avg_score = np.mean(scores)
        print(f"  {high}Hz: 平均F1分数 {avg_score:.4f} (共{len(scores)}个配置)")

    # 按阈值因子分组分析
    threshold_groups = {}
    for result in config_results:
        thresh = result['params']['threshold_factor']
        if thresh not in threshold_groups:
            threshold_groups[thresh] = []
        threshold_groups[thresh].append(result['f1_score'])

    print(f"\n阈值因子影响:")
    for thresh, scores in sorted(threshold_groups.items()):
        avg_score = np.mean(scores)
        print(f"  {thresh}: 平均F1分数 {avg_score:.4f} (共{len(scores)}个配置)")

    return config_results


def main():
    # MIT-BIH - 单通道检测
    # 定义要检测的目标通道
    target_lead = "MLII"  # 可以修改为其他导联如 "V1", "V5" 等

    total_annotation_num = 0
    total_detection_num = 0

    # 详细统计数据结构
    detailed_stats = {
        "tp": 0, "fp": 0, "fn": 0, "time_errors": []
    }

    print(f"开始检测 {target_lead} 导联...")
    print("=" * 60)

    processed_count = 0
    for num in numberSet:
        try:
            # 加载数据文件
            input_data = wfdb.rdrecord(root + num)
            sig_name = input_data.sig_name

            # 检查是否包含目标导联
            if target_lead not in sig_name:
                continue

            # 获取目标导联的信号
            target_idx = sig_name.index(target_lead)
            signal = wfdb.rdrecord(root + num, channel_names=[target_lead]).p_signal.flatten()

            print(f"\n处理记录 {num} - {target_lead} 导联")
            print(f"信号长度: {len(signal)} 样本")

            # 加载标注文件
            annotation = wfdb.rdann(root + num, 'atr')
            fs = annotation.fs
            ann_len = annotation.ann_len
            # MIT-BIH标注从1开始，需要转换为0-based索引
            sig_sample = annotation.sample[1:]

            total_annotation_num += ann_len
            processed_count += 1

            # 计算标注的平均心率（基于标注的R-R间期）
            if len(sig_sample) > 1:
                annotation_rr_intervals = np.diff(sig_sample) * 1000 / fs  # 转换为ms
                annotation_avg_rr = np.mean(annotation_rr_intervals)
                # 心率计算：1分钟=60秒=60000ms，心率=60000ms / 平均R-R间期
                annotation_heart_rate = 60000 / annotation_avg_rr  # bpm
                print(f"标注平均心率: {annotation_heart_rate:.2f} bpm")
            else:
                print("标注数量不足，无法计算平均心率")

            # 创建QRS检测器实例
            qrs_detector = PanTomkinsQRSDetector()
            qrs_detector.fs = fs

            # 进行QRS检测
            qrs_peaks = qrs_detector.detect_qrs_peaks(signal, signal_name=target_lead)
            heart_rate, rr_intervals = qrs_detector.calculate_heart_rate()

            total_detection_num += len(qrs_peaks)
            print(f"检测到 {len(qrs_peaks)} 个QRS波")
            print(f"算法检测平均心率: {heart_rate:.1f} bpm")

            # 绘制结果（只处理前几个记录以避免过多图表）
            if processed_count <= 0:
                print(f"\n绘制记录 {num} 的检测结果...")
                qrs_detector.plot_results(signal, start_idx=0, num_samples=3000,
                                          annotation_samples=sig_sample, record_number=num)

                print(f"\n绘制记录 {num} 的错误分析图...")
                error_stats = qrs_detector.plot_error_analysis(signal, sig_sample, start_idx=0,
                                                               num_samples=3000, record_number=num)

            # 使用比较函数进行详细统计
            stats = qrs_detector.compare_with_annotations(sig_sample)
            qrs_detector.print_comparison_stats(stats, f"{target_lead} (Record {num})")

            # 收集详细统计数据
            detailed_stats["tp"] += stats['true_positives']
            detailed_stats["fp"] += stats['false_positives']
            detailed_stats["fn"] += stats['false_negatives']
            # 收集时间误差
            for pair in stats['matched_pairs']:
                detailed_stats["time_errors"].append(pair['time_diff_ms'])

            # 打印基本统计信息
            print(f"\n=== 基本检测统计信息 ===")
            print(f"QRS波标注数量: {stats['total_annotations']}")
            print(f"QRS波检测数量: {stats['total_detections']}")
            print(f"算法检测平均心率: {heart_rate:.2f} bpm")
            if len(rr_intervals) > 0:
                print(f"R-R间期均值: {np.mean(rr_intervals):.2f} ms")
                print(f"R-R间期标准差: {np.std(rr_intervals):.2f} ms")

        except Exception as e:
            print(f"处理记录 {num} 时出错: {str(e)}")
            continue

    print(f"\n" + "=" * 60)
    print("总体检测统计信息:")

    # 计算总体统计数据
    total_tp = detailed_stats["tp"]
    total_fp = detailed_stats["fp"]
    total_fn = detailed_stats["fn"]
    all_time_errors = detailed_stats["time_errors"]

    # 计算性能指标
    total_annotations = total_tp + total_fn
    total_detections = total_tp + total_fp

    if total_annotations > 0:
        overall_sensitivity = total_tp / total_annotations
        overall_detection_rate = overall_sensitivity * 100
    else:
        overall_sensitivity = 0
        overall_detection_rate = 0

    if total_detections > 0:
        overall_precision = total_tp / total_detections
    else:
        overall_precision = 0

    if overall_sensitivity + overall_precision > 0:
        overall_f1 = 2 * (overall_sensitivity * overall_precision) / (overall_sensitivity + overall_precision)
    else:
        overall_f1 = 0

    print(f"\n总体检测性能 ({target_lead} 导联):")
    print(f"  处理记录数: {processed_count}")
    print(f"  总标注数: {total_annotations}")
    print(f"  总检测数: {total_detections}")
    print(f"  总正确检测: {total_tp}")
    print(f"  总误检: {total_fp}")
    print(f"  总漏检: {total_fn}")
    print(f"  总体检测率: {overall_detection_rate:.2f}%")
    print(f"  总体敏感度: {overall_sensitivity:.4f}")
    print(f"  总体精确率: {overall_precision:.4f}")
    print(f"  总体F1分数: {overall_f1:.4f}")

    # 总体时间误差统计
    if all_time_errors:
        print(f"\n总体时间误差统计:")
        print(f"  平均误差: {np.mean(all_time_errors):.2f} ms")
        print(f"  标准差: {np.std(all_time_errors):.2f} ms")
        print(f"  最大误差: {np.max(all_time_errors):.2f} ms")
        print(f"  最小误差: {np.min(all_time_errors):.2f} ms")

        # 计算误差分布
        error_under_25ms = sum(1 for e in all_time_errors if e <= 25)
        error_25_to_50ms = sum(1 for e in all_time_errors if 25 < e <= 50)
        error_over_50ms = sum(1 for e in all_time_errors if e > 50)
        total_matches = len(all_time_errors)

        print(f"\n时间误差分布:")
        print(f"  ≤25ms: {error_under_25ms}/{total_matches} ({error_under_25ms / total_matches * 100:.1f}%)")
        print(f"  25-50ms: {error_25_to_50ms}/{total_matches} ({error_25_to_50ms / total_matches * 100:.1f}%)")
        print(f"  >50ms: {error_over_50ms}/{total_matches} ({error_over_50ms / total_matches * 100:.1f}%)")
    else:
        print(f"\n总体时间误差统计: 无匹配数据")

    print(f"\n算法参数:")
    print(f"  目标导联: {target_lead}")
    print(f"  匹配容差: 50ms")
    print(f"  滑动窗口大小: 8秒")
    print(f"  窗口重叠: 4秒")
    print(f"  自适应不应期: V1(150ms), V2/V3(170ms), 其他(200ms)")

    print("\n" + "=" * 60)


if __name__ == "__main__":

    if 1:
        # 参数调试模式 - 单个导联
        # 设置要测试的导联和记录
        target_lead = "MLII"  # 可以修改为 "V1", "V2", "V5" 等其他导联
        print(f"运行{target_lead}导联参数调试...")
        # 可以指定要测试的记录子集，例如：['100', '101', '103']
        debug_lead_parameters(target_lead=target_lead, tolerance_ms=50)
    elif 0:
        # 参数调试模式 - 所有导联
        print("运行所有导联参数调试...")
        # 使用较少的记录进行快速测试
        test_records_small = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112']
        debug_all_leads(record_subset=test_records_small, tolerance_ms=50)
    elif 0:
        # 标准检测模式
        main()

    print("\n" + "=" * 60)
