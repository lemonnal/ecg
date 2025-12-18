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
# numberSet = ['100']


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
            # # 肢体导联-加压单极导联
            # 'aVR': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 'aVL': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 'aVF': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # # 肢体导联-标准双极导联
            # 'I': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 'MLII': {'low': 5, 'high': 25.0, 'threshold_factor': 1.4},
            # 'MLIII': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # # V1特殊处理
            # 'V1': {'low': 1, 'high': 50.0, 'threshold_factor': 1.2},
            # # V1导联特点：R波小，S波深，需要更低频率捕获，更高频率保留细节
            #
            # # 胸前导联 - 过渡区
            # 'V2': {'low': 3, 'high': 30.0, 'threshold_factor': 1.3},
            # # V2导联特点：介于V1和V3之间，中等参数
            
            # 胸前导联 - 左心前区
            # 'V3': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 'V4': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'V5': {'low': 5, 'high': 20.0, 'threshold_factor': 1.4},
            # 'V6': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
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

        # 使用5阶Butterworth滤波器 - 平衡滤波效果和信号保留
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

        # 步骤1: 带通滤波 - 传递signal_name以使用自适应参数
        self.filtered_signal = self.bandpass_filter(signal_data, signal_name=signal_name)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: QRS检测 - 传递信号名称用于自适应阈值
        self.qrs_peaks = self._threshold_detection(signal_name=signal_name)

        return self.qrs_peaks

    def _threshold_detection(self, signal_name="MLII"):
        """
        滑动窗口阈值检测算法
        使用自适应的滑动窗口来适应信号变化
        """
        if self.integrated_signal is None:
            return []

        # 获取该导联的自适应参数
        params = self.get_filter_parameters(signal_name)
        threshold_factor = params.get('threshold_factor', 1.5)

        # 设置滑动窗口参数
        window_size = int(8 * self.fs)  # 10秒窗口
        overlap_size = int(4 * self.fs)   # 5秒重叠

        # 设置自适应不应期
        if signal_name == 'V1':
            refractory_period = int(0.15 * self.fs)  # 150ms
        elif signal_name in ['V2', 'V3']:
            refractory_period = int(0.17 * self.fs)  # 170ms
        else:
            refractory_period = int(0.2 * self.fs)   # 200ms

        all_peaks = []

        # 滑动窗口处理
        for start_idx in range(0, len(self.integrated_signal), overlap_size):
            end_idx = min(start_idx + window_size, len(self.integrated_signal))

            if end_idx - start_idx < overlap_size:  # 最后一个窗口太小就跳过
                break

            # 提取当前窗口的信号
            window_signal = self.integrated_signal[start_idx:end_idx]

            # 计算当前窗口的阈值
            window_mean = np.mean(window_signal)
            window_std = np.std(window_signal)
            current_threshold = window_mean + threshold_factor * window_std

            # 在窗口内检测峰值
            window_peaks = []
            for i in range(len(window_signal)):
                actual_idx = start_idx + i
                current_value = window_signal[i]

                # 检查是否超过阈值
                if current_value > current_threshold:
                    # 检查是否在不应期内（相对于整个信号的峰值）
                    if len(all_peaks) == 0 or (actual_idx - all_peaks[-1]) > refractory_period:
                        # 检查是否在窗口峰值的不应期内
                        if len(window_peaks) == 0 or (i - window_peaks[-1]) > int(0.2 * self.fs):
                            # 在小窗口内寻找真正的峰值
                            search_window = min(10, len(window_signal) - i - 1)
                            local_peak_idx = i

                            for j in range(max(0, i - 5), min(len(window_signal), i + search_window + 1)):
                                if window_signal[j] > window_signal[local_peak_idx]:
                                    local_peak_idx = j

                            # 添加找到的峰值
                            if local_peak_idx not in window_peaks:
                                window_peaks.append(local_peak_idx)
                                all_peaks.append(start_idx + local_peak_idx)

        # 峰值精确定位 - 在原始滤波信号上找到最大值
        refined_peaks = []
        search_window = int(0.03 * self.fs)  # 30ms搜索窗口

        for peak in all_peaks:
            search_start = max(0, peak - search_window)
            search_end = min(len(self.filtered_signal), peak + search_window)

            if search_start < search_end:
                search_segment = self.filtered_signal[search_start:search_end]
                if len(search_segment) > 0:
                    local_max_idx = np.argmax(search_segment) + search_start
                    # 避免重复检测
                    if len(refined_peaks) == 0 or (local_max_idx - refined_peaks[-1]) > int(0.15 * self.fs):
                        refined_peaks.append(local_max_idx)

        return refined_peaks

    def _threshold_detection_global(self, signal_name="MLII"):
        """
        全局阈值检测算法（原版保留）
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

    def plot_error_analysis(self, signal_data, annotation_samples, start_idx=0, num_samples=3000, tolerance_ms=150, record_number=""):
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
                ax1.axvline(x=ann/self.fs, color='green', linestyle='--', alpha=0.7, linewidth=2)

        # 绘制检测结果
        for det in self.qrs_peaks:
            if start_idx <= det < end_idx:
                ax1.axvline(x=det/self.fs, color='red', linestyle=':', alpha=0.7, linewidth=2)

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
            ax2.plot(det/self.fs, signal_data[det], 'go', markersize=10, label='True Positive' if len(true_positives) > 0 and (ann, det) == true_positives[0] else "")
            ax2.annotate(f'+{diff*1000/self.fs:.1f}ms', (det/self.fs, signal_data[det]),
                        xytext=(5, -15), textcoords='offset points', fontsize=7, color='green')

        # 标记漏检
        for fn in false_negatives:
            ax2.plot(fn/self.fs, signal_data[fn], 'rx', markersize=12, markeredgewidth=3,
                    label='False Negative' if len(false_negatives) > 0 and fn == false_negatives[0] else "")
            ax2.annotate('FN', (fn/self.fs, signal_data[fn]),
                        xytext=(5, -20), textcoords='offset points', fontsize=7,
                        color='red', weight='bold')

        # 标记误检
        for fp in false_positives:
            ax2.plot(fp/self.fs, signal_data[fp], 'r^', markersize=10,
                    label='False Positive' if len(false_positives) > 0 and fp == false_positives[0] else "")
            ax2.annotate('FP', (fp/self.fs, signal_data[fp]),
                        xytext=(5, 15), textcoords='offset points', fontsize=7,
                        color='red', weight='bold')

        ax2.set_title(f'Error Analysis: TP={len(true_positives)}, FN={len(false_negatives)}, FP={len(false_positives)} (Record {record_number})')
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


def main():
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

    # 详细统计数据结构
    detailed_stats = {
        "I": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "MLII": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "MLIII": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "V1": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "V2": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "V3": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "V4": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []},
        "V5": {"tp": 0, "fp": 0, "fn": 0, "time_errors": []}
    }

    for num in numberSet:
        # 加载数据文件
        print(f"file:{num}")
        input_data = wfdb.rdrecord(root + num)
        sig_name = input_data.sig_name
        print(f"{sig_name}")
        signal1 = wfdb.rdrecord(root + num, channel_names=[sig_name[0]]).p_signal.flatten()
        signal2 = wfdb.rdrecord(root + num, channel_names=[sig_name[1]]).p_signal.flatten()

        # 加载标注文件
        annotation = wfdb.rdann(root + num, 'atr')
        fs = annotation.fs
        ann_len = annotation.ann_len
        # MIT-BIH标注从1开始，需要转换为0-based索引
        sig_sample = annotation.sample[1:]
        sig_symbol = annotation.symbol


        if "MLII" not in sig_name:
            continue

        total_annotation_num[sig_name[0]] += ann_len
        total_annotation_num[sig_name[1]] += ann_len

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
        qrs_detector1 = PanTomkinsQRSDetector(fs=fs)
        qrs_detector2 = PanTomkinsQRSDetector(fs=fs)

        # 对第一列信号进行QRS检测 - 传递导联名称以使用自适应参数
        qrs_peaks1 = qrs_detector1.detect_qrs_peaks(signal1, sig_name[0])
        heart_rate1, rr_intervals1 = qrs_detector1.calculate_heart_rate()

        # 对第二列信号进行QRS检测 - 传递导联名称以使用自适应参数
        qrs_peaks2 = qrs_detector2.detect_qrs_peaks(signal2, sig_name[1])
        heart_rate2, rr_intervals2 = qrs_detector2.calculate_heart_rate()

        total_detection_num[sig_name[0]] += len(qrs_peaks1)
        total_detection_num[sig_name[1]] += len(qrs_peaks2)
        print(f"第一列信号检测到 {len(qrs_peaks1)} 个QRS波")
        print(f"平均心率: {heart_rate1:.1f} bpm")
        print(f"第二列信号检测到 {len(qrs_peaks2)} 个QRS波")
        print(f"平均心率: {heart_rate2:.1f} bpm")


        if num in numberSet[0:11]:
            # 绘制详细结果图
            # print("\n绘制第一列信号的QRS检测结果...")
            qrs_detector1.plot_results(signal1, start_idx=0, num_samples=3000, annotation_samples=sig_sample, record_number=num)

            # print("\n绘制第二列信号的QRS检测结果...")
            qrs_detector2.plot_results(signal2, start_idx=0, num_samples=3000, annotation_samples=sig_sample, record_number=num)

            # 绘制错误分析图
            # print("\n绘制第一列信号的错误分析图...")
            error_stats1 = qrs_detector1.plot_error_analysis(signal1, sig_sample, start_idx=0, num_samples=3000, record_number=num)

            # print("\n绘制第二列信号的错误分析图...")
            error_stats2 = qrs_detector2.plot_error_analysis(signal2, sig_sample, start_idx=0, num_samples=3000, record_number=num)

        # 使用新的比较函数进行详细统计
        stats1 = qrs_detector1.compare_with_annotations(sig_sample)
        qrs_detector1.print_comparison_stats(stats1, f"信号1({sig_name[0]})")

        stats2 = qrs_detector2.compare_with_annotations(sig_sample)
        qrs_detector2.print_comparison_stats(stats2, f"信号2({sig_name[1]})")

        # 收集详细统计数据
        if sig_name[0] in detailed_stats:
            detailed_stats[sig_name[0]]["tp"] += stats1['true_positives']
            detailed_stats[sig_name[0]]["fp"] += stats1['false_positives']
            detailed_stats[sig_name[0]]["fn"] += stats1['false_negatives']
            # 收集时间误差
            for pair in stats1['matched_pairs']:
                detailed_stats[sig_name[0]]["time_errors"].append(pair['time_diff_ms'])

        if sig_name[1] in detailed_stats:
            detailed_stats[sig_name[1]]["tp"] += stats2['true_positives']
            detailed_stats[sig_name[1]]["fp"] += stats2['false_positives']
            detailed_stats[sig_name[1]]["fn"] += stats2['false_negatives']
            # 收集时间误差
            for pair in stats2['matched_pairs']:
                detailed_stats[sig_name[1]]["time_errors"].append(pair['time_diff_ms'])

        # 打印基本统计信息
        print(f"\n=== 基本检测统计信息 ===")
        print(f"信号1 - QRS波标注数量: {stats1['total_annotations']}")
        print(f"信号1 - QRS波检测数量: {stats1['total_detections']}")
        print(f"信号1 - 平均心率: {heart_rate1:.2f} bpm")
        if len(rr_intervals1) > 0:
            print(f"信号1 - R-R间期均值: {np.mean(rr_intervals1):.2f} ms")
            print(f"信号1 - R-R间期标准差: {np.std(rr_intervals1):.2f} ms")

        print(f"\n信号2 - QRS波标注数量: {stats2['total_annotations']}")
        print(f"信号2 - QRS波检测数量: {stats2['total_detections']}")
        print(f"信号2 - 平均心率: {heart_rate2:.2f} bpm")
        if len(rr_intervals2) > 0:
            print(f"信号2 - R-R间期均值: {np.mean(rr_intervals2):.2f} ms")
            print(f"信号2 - R-R间期标准差: {np.std(rr_intervals2):.2f} ms")

    print("\n" + "=" * 60)
    print("总体检测统计信息:")

    # 计算总体统计数据
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_time_errors = []

    print("各导联详细统计:")
    for key in detailed_stats.keys():
        tp = detailed_stats[key]["tp"]
        fp = detailed_stats[key]["fp"]
        fn = detailed_stats[key]["fn"]
        time_errors = detailed_stats[key]["time_errors"]

        # 累计总体统计
        total_tp += tp
        total_fp += fp
        total_fn += fn
        all_time_errors.extend(time_errors)

        # 计算性能指标
        total_annotations = tp + fn
        total_detections = tp + fp

        if total_annotations > 0:
            sensitivity = tp / total_annotations  # 敏感度（召回率）
            detection_rate = sensitivity * 100
        else:
            sensitivity = 0
            detection_rate = 0

        if total_detections > 0:
            precision = tp / total_detections  # 精确率
        else:
            precision = 0

        if sensitivity + precision > 0:
            f1_score = 2 * (sensitivity * precision) / (sensitivity + precision)  # F1分数
        else:
            f1_score = 0

        print(f"\n导联 {key}:")
        print(f"  标注总数: {total_annotations}")
        print(f"  检测总数: {total_detections}")
        print(f"  ┌─ 正确检测 (TP): {tp}")
        print(f"  ├─ 误检 (FP): {fp}")
        print(f"  └─ 漏检 (FN): {fn}")
        print(f"  检测率: {detection_rate:.2f}%")
        print(f"  敏感度 (Sensitivity): {sensitivity:.4f}")
        print(f"  精确率 (Precision): {precision:.4f}")
        print(f"  F1分数: {f1_score:.4f}")

        # 时间误差统计
        if time_errors:
            print(f"  时间误差统计:")
            print(f"    平均误差: {np.mean(time_errors):.2f} ms")
            print(f"    标准差: {np.std(time_errors):.2f} ms")
            print(f"    最大误差: {np.max(time_errors):.2f} ms")
        else:
            print(f"  时间误差统计: 无匹配数据")

        print("-" * 60)

    # 总体性能统计
    print(f"\n{'='*20} 总体性能统计 {'='*20}")
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

    print(f"\n总体检测性能:")
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
        print(f"  ≤25ms: {error_under_25ms}/{total_matches} ({error_under_25ms/total_matches*100:.1f}%)")
        print(f"  25-50ms: {error_25_to_50ms}/{total_matches} ({error_25_to_50ms/total_matches*100:.1f}%)")
        print(f"  >50ms: {error_over_50ms}/{total_matches} ({error_over_50ms/total_matches*100:.1f}%)")
    else:
        print(f"\n总体时间误差统计: 无匹配数据")

    print(f"\n算法参数:")
    print(f"  匹配容差: 50ms")
    print(f"  滑动窗口大小: 8秒")
    print(f"  窗口重叠: 4秒")
    print(f"  自适应不应期: V1(150ms), V2/V3(170ms), 其他(200ms)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
    print("\n" + "=" * 60)
