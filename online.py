from collections import deque
import numpy as np
from scipy import signal as scipy_signal
import asyncio
from bleak import BleakScanner
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
import matplotlib.pyplot as plt
import struct
from signal_params import get_signal_params

QINGXUN_UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-68716563686f"
QINGXUN_UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-68716563686f"
QINGXUN_UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-68716563686f"


# 设备配置 - 当前选择的设备
DEVICE_NAME = "AAA-TEST"
if DEVICE_NAME == "AAA-TEST":
    device_param = {
        "name": DEVICE_NAME,
        "address": "EC:7A:26:9D:81:3F",
        "service_uuid": QINGXUN_UART_SERVICE_UUID,
        "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
        "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
    }
elif DEVICE_NAME == "PW-ECG-SL":
    device_param = {
        "name": DEVICE_NAME,
        "address": "E2:1B:A5:DB:DE:EA",
        "service_uuid": QINGXUN_UART_SERVICE_UUID,
        "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
        "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
    }

# ============================================================================
# 初始化matplotlib图形显示系统
# ============================================================================

# 开启交互模式，允许图形实时更新而不阻塞程序
plt.ion()

# ----------------------------------------------------------------------------
# 配置子图信息：每个子图对应Pan-Tomkins算法的一个处理阶段
# ----------------------------------------------------------------------------
SUBPLOT_CONFIG = [
    {'ylabel': 'original signal', 'color': 'b', 'description': 'Original ECG Signal'},
    {'ylabel': 'filtered signal', 'color': 'g', 'description': 'Bandpass Filtered Signal'},
    {'ylabel': 'differentiated signal', 'color': 'm', 'description': 'Differentiated Signal'},
    {'ylabel': 'squared signal', 'color': 'y', 'description': 'Squared Signal'},
    {'ylabel': 'integrated signal', 'color': 'k', 'description': 'Moving Window Integrated Signal'}
]

# ----------------------------------------------------------------------------
# 配置PQRST波形标记样式：定义每种波形的颜色、标记形状和说明
# ----------------------------------------------------------------------------
WAVE_MARKER_CONFIG = {
    'r': {'color': 'red', 'marker': 'o', 'label': 'R', 'size': 64, 'desc': 'R Peak - QRS Complex Main Peak'},
    'q': {'color': 'blue', 'marker': '^', 'label': 'Q', 'size': 64, 'desc': 'Q Wave - Negative Wave Before R Peak'},
    's': {'color': 'green', 'marker': 'v', 'label': 'S', 'size': 64, 'desc': 'S Wave - Negative Wave After R Peak'},
    'p': {'color': 'magenta', 'marker': 's', 'label': 'P', 'size': 64, 'desc': 'P Wave - Atrial Depolarization'},
    't': {'color': 'cyan', 'marker': 'D', 'label': 'T', 'size': 64, 'desc': 'T Wave - Ventricular Repolarization'}
}

# ----------------------------------------------------------------------------
# 创建主图形窗口和5个垂直排列的子图
# - 5个子图共享x轴（时间轴），便于对比不同处理阶段的信号
# - figsize=(10, 8): 设置图形窗口大小为10x8英寸
# ----------------------------------------------------------------------------
NUM_SUBPLOTS = len(SUBPLOT_CONFIG)
fig, axes = plt.subplots(NUM_SUBPLOTS, 1, figsize=(10, 8), sharex=True)

# 确保axes始终是列表（单子图时plt.subplots返回单个Axes对象）
if NUM_SUBPLOTS == 1:
    axes = [axes]

# ----------------------------------------------------------------------------
# 批量创建每个子图的信号曲线对象
# ----------------------------------------------------------------------------
lines = []
for ax, config in zip(axes, SUBPLOT_CONFIG):
    line, = ax.plot([], f"{config['color']}-", label=config['description'])
    ax.set_ylabel(config['ylabel'])
    lines.append(line)

# 为第一个子图添加图例
axes[0].legend(loc='upper right', fontsize=8)

# ============================================================================
# 批量创建scatter对象用于标记PQRST波的特征点
# 优势：避免每次更新时重复创建/删除对象，大幅提升性能
# ============================================================================
scatter_objects = {}
for wave_type, marker_cfg in WAVE_MARKER_CONFIG.items():
    scatter_list = []
    for i, ax in enumerate(axes):
        scatter = ax.scatter(
            [], [], 
            c=marker_cfg['color'],
            s=marker_cfg['size'],
            marker=marker_cfg['marker'],
            label=marker_cfg['label'] if i == 0 else None,  # 只在第一个子图显示标签
            zorder=5
        )
        scatter_list.append(scatter)
    scatter_objects[wave_type] = scatter_list

# ----------------------------------------------------------------------------
# 为了向后兼容，保留原有的变量名
# ----------------------------------------------------------------------------
ax1, ax2, ax3, ax4, ax5 = axes
line1, line2, line3, line4, line5 = lines



class RealTimeECGDetector:
    """
    基于Pan-Tomkins算法的实时ECG波形检测器

    实现完整的ECG波形检测，包括:
    - R峰检测 (QRS复合波的峰值)
    - Q波检测 (R峰前的负向波)
    - S波检测 (R峰后的负向波)
    - P波检测 (QRS波之前的正向小波，代表心房去极化)
    - T波检测 (QRS波之后的正向宽波，代表心室复极化)

    算法流程:
    1. 带通滤波 - 去除基线漂移和高频噪声
    2. 微分 - 突出QRS波的陡峭斜率
    3. 平方 - 使所有值为正并放大高斜率区域
    4. 移动窗口积分 - 平滑信号并提取QRS波特征
    5. 自适应阈值检测 - 使用滑动窗口和EMA平滑检测R峰
    6. 相位延迟补偿 - 补偿滤波和积分引入的延迟
    """

    def __init__(self, signal_name="MLII"):
        """
        初始化ECG波形检测器

        参数:
            signal_name: ECG导联名称 (如 "MLII", "V1", "V2", "I", "aVR" 等)
        """
        # 采样参数
        self.fs = 250  # 采样频率 (Hz)
        self.signal_len = 750  # 信号缓冲区长度 (采样点数)

        # 信号缓冲区 (使用deque实现滑动窗口)
        self.signal = deque([], self.signal_len)
        self.filtered_signal = None      # 带通滤波后的信号
        self.differentiated_signal = None # 微分后的信号
        self.squared_signal = None       # 平方后的信号
        self.integrated_signal = None    # 移动窗口积分后的信号

        # 检测结果存储
        self.qrs_peaks = []  # R峰位置列表
        self.q_waves = []    # Q波位置列表
        self.s_waves = []    # S波位置列表
        self.p_waves = []    # P波位置列表
        self.t_waves = []    # T波位置列表

        # 获取导联相关的处理参数并提取为直接属性
        params = get_signal_params('online', signal_name)
        
        # 滤波参数
        self.low = params['low']
        self.high = params['high']
        self.filter_order = params['filter_order']
        self.original_weight = params['original_weight']
        self.filtered_weight = params['filtered_weight']
        
        # 相位延迟补偿参数
        self.phase_delay_compensation_samples = int(params['compensation_ms'] * self.fs)

        # 阈值平滑参数（指数移动平均 EMA）
        self.ema_threshold = None
        self.ema_alpha = params['ema_alpha']  # 平滑系数，越小变化越慢

        # QRS检测参数
        self.integration_window_size = params['integration_window_size']
        self.refractory_period = int(params['refractory_period'] * self.fs)
        self.threshold_factor = params['threshold_factor']

        # Q波检测参数 (R峰前的负向波)
        self.q_wave_search_start = int(params['q_wave_search_start'] * self.fs)
        self.q_wave_search_end = int(params['q_wave_search_end'] * self.fs)
        self.q_wave_min_amplitude = params['q_wave_min_amplitude']

        # S波检测参数 (R峰后的负向波)
        self.s_wave_search_start = int(params['s_wave_search_start'] * self.fs)
        self.s_wave_search_end = int(params['s_wave_search_end'] * self.fs)
        self.s_wave_min_amplitude = params['s_wave_min_amplitude']

        # P波检测参数 (QRS波之前的正向小波，心房去极化)
        self.p_wave_search_start = int(params['p_wave_search_start'] * self.fs)
        self.p_wave_search_end = int(params['p_wave_search_end'] * self.fs)
        self.p_wave_min_amplitude = params['p_wave_min_amplitude']
        self.p_wave_max_width = int(params['p_wave_max_width'] * self.fs)

        # T波检测参数 (QRS波之后的正向宽波，心室复极化)
        self.t_wave_search_start = int(params['t_wave_search_start'] * self.fs)
        self.t_wave_search_end = int(params['t_wave_search_end'] * self.fs)
        self.t_wave_min_amplitude = params['t_wave_min_amplitude']
        self.t_wave_max_width = int(params['t_wave_max_width'] * self.fs)

    def bandpass_filter(self, signal_data):
        """
        自适应带通滤波器
        根据不同导联使用不同的频率参数，去除基线漂移和高频噪声

        参数:
            signal_data: 输入ECG信号数组

        返回:
            combined_signal: 滤波后与原始信号加权组合的信号
        """
        # 设计带通滤波器
        nyquist = 0.5 * self.fs
        low = self.low / nyquist
        high = self.high / nyquist

        # 使用 n 阶 Butterworth 滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(self.filter_order, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # 添加原始信号的加权
        combined_signal = (self.original_weight * signal_data
                           + self.filtered_weight * filtered_signal)
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
        h = 1 / self.fs
        # 使用5点中心差分公式
        # f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)
        for i in range(2, len(signal_data) - 2):
            differentiated_signal[i] = (-signal_data[i + 2] + 8 * signal_data[i + 1]
                                        - 8 * signal_data[i - 1] + signal_data[i - 2]) / (12 * h)

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

    def moving_window_integration(self, signal_data):
        """
        移动窗口积分器
        对微分平方后的信号进行平滑，突出QRS波特征

        参数:
            signal_data: 输入信号 (通常是微分平方后的信号)

        返回:
            integrated_signal: 移动平均积分后的信号
        """
        # 窗口中的采样点数量
        window_sample = int(self.integration_window_size * self.fs)

        # 使用卷积实现移动平均积分
        window = np.ones(window_sample) / window_sample
        integrated_signal = np.convolve(signal_data, window, mode='same')

        return integrated_signal

    def threshold_detection(self, signal_data):
        """
        滑动窗口阈值检测算法
        使用自适应的滑动窗口来适应信号变化，检测QRS波峰值

        参数:
            signal_data: 输入积分信号

        返回:
            refined_peaks: 定位的QRS波峰值位置列表
        """
        if signal_data is None or len(signal_data) == 0:
            return []

        # 设置滑动窗口参数
        window_size = int(self.signal_len * (1 / 3))  # 检测窗口 - 信号窗口 - 1秒
        overlap_size = int(self.signal_len * (1 / 6))    # 重叠窗口大小 - 信号窗口 - 0.5秒

        # 使用预存储的实例属性（性能优化）
        refractory_period = self.refractory_period  # 不应期
        threshold_factor = self.threshold_factor    # 阈值系数

        all_peaks = [] # 检测到的R-peaks

        # 滑动窗口处理
        for start_idx in range(0, len(signal_data), overlap_size):
            end_idx = min(start_idx + window_size, len(signal_data))

            if end_idx - start_idx < overlap_size:  # 最后一个窗口太小就跳过
                break

            # 提取当前窗口的信号
            window_signal = signal_data[start_idx:end_idx]

            # 计算当前窗口的自适应阈值
            window_mean = np.mean(window_signal)
            window_std = np.std(window_signal)
            raw_threshold = window_mean + threshold_factor * window_std

            # 使用指数移动平均平滑阈值，避免突变
            if self.ema_threshold is None:
                self.ema_threshold = raw_threshold
            else:
                self.ema_threshold = (self.ema_alpha * raw_threshold +
                                      (1 - self.ema_alpha) * self.ema_threshold)

            current_threshold = self.ema_threshold
            # print('window_mean: {}, window_std: {}'.format(window_mean, window_std))
            # print('raw_threshold: {}, ema_threshold: {}'.format(raw_threshold, self.ema_threshold))

            # 在窗口内检测候选峰值
            window_peaks = []
            for i in range(len(window_signal)):
                actual_idx = start_idx + i
                current_value = window_signal[i]
                # 第一级过滤: 检查是否超过阈值
                if current_value > current_threshold:
                    # 第二级过滤: 检查是否在不应期内
                    if len(all_peaks) == 0 or (actual_idx - all_peaks[-1]) > refractory_period:
                        # 在窗口内寻找峰值点
                        search_range = min(10, len(window_signal) - i - 1)
                        local_peak_idx = i

                        for j in range(max(0, i - 5), min(len(window_signal), i + search_range + 1)):
                            if window_signal[j] > window_signal[local_peak_idx]:
                                local_peak_idx = j

                        # 添加找到的峰值 (避免重复)
                        if local_peak_idx not in window_peaks:
                            window_peaks.append(local_peak_idx)
                            all_peaks.append(start_idx + local_peak_idx)

        return all_peaks

    def apply_delay_compensation(self, peaks):
        """
        相位延迟补偿
        由于带通滤波和移动窗口积分会引入相位延迟，
        需要将检测到的峰值位置向前移动若干采样点以对齐真实R峰位置。

        参数:
            peaks: 检测到的QRS波峰值位置列表

        返回:
            compensated_peaks: 补偿后的峰值位置列表
        """
        if not peaks:
            return []

        compensated_peaks = []
        for peak in peaks:
            # 向前移动补偿采样点数
            compensated_idx = peak + self.phase_delay_compensation_samples

            # 确保补偿后的索引不越界
            if compensated_idx >= 0:
                # 在补偿位置附近搜索真实峰值（原始信号中的最大值）
                # 搜索范围为补偿位置前后±5个采样点
                search_start = max(0, compensated_idx - 5)
                search_end = min(len(self.signal), compensated_idx + 6)

                signal_array = np.array(list(self.signal))
                local_peak_idx = search_start + np.argmax(signal_array[search_start:search_end])

                compensated_peaks.append(local_peak_idx)
            else:
                # 如果补偿后越界，保持原位置或设为0
                compensated_peaks.append(max(0, peak))

        return compensated_peaks

    def detect_q_waves(self, r_peaks, signal_array):
        """
        检测Q波
        Q波在QRS波之前，R峰前的负向波

        参数:
            r_peaks: R峰位置列表
            signal_array: 原始ECG信号数组

        返回:
            q_waves: Q波位置列表
        """
        q_waves = []

        if not r_peaks or len(signal_array) == 0:
            return q_waves

        for r_peak in r_peaks:
            # 定义Q波搜索窗口 (R峰前)
            search_start = max(0, r_peak - self.q_wave_search_start)
            search_end = max(0, r_peak - self.q_wave_search_end)

            if search_end <= search_start:
                continue

            # 在搜索窗口内找最小值点（Q波是负向波）
            window = signal_array[search_start:search_end]
            if len(window) == 0:
                continue

            min_idx = np.argmin(window)
            q_peak_candidate = search_start + min_idx

            # 获取R峰幅值作为参考
            r_amplitude = signal_array[r_peak]
            q_amplitude = signal_array[q_peak_candidate]

            # 验证Q波特征:
            # 1. Q波幅值应明显小于R峰（负向偏转）
            # 2. 幅值差应超过最小阈值
            amplitude_diff = r_amplitude - q_amplitude

            if (amplitude_diff > self.q_wave_min_amplitude and
                q_amplitude < r_amplitude * 0.7):  # Q波应明显低于R峰
                q_waves.append(q_peak_candidate)

        return q_waves

    def detect_s_waves(self, r_peaks, signal_array):
        """
        检测S波
        S波在QRS波之后，R峰后的负向波

        参数:
            r_peaks: R峰位置列表
            signal_array: 原始ECG信号数组

        返回:
            s_waves: S波位置列表
        """
        s_waves = []

        if not r_peaks or len(signal_array) == 0:
            return s_waves

        for r_peak in r_peaks:
            # 定义S波搜索窗口 (R峰后)
            search_start = min(len(signal_array) - 1, r_peak + self.s_wave_search_start)
            search_end = min(len(signal_array), r_peak + self.s_wave_search_end)

            if search_end <= search_start:
                continue

            # 在搜索窗口内找最小值点（S波是负向波）
            window = signal_array[search_start:search_end]
            if len(window) == 0:
                continue

            min_idx = np.argmin(window)
            s_peak_candidate = search_start + min_idx

            # 获取R峰幅值作为参考
            r_amplitude = signal_array[r_peak]
            s_amplitude = signal_array[s_peak_candidate]

            # 验证S波特征:
            # 1. S波幅值应明显小于R峰（负向偏转）
            # 2. 幅值差应超过最小阈值
            amplitude_diff = r_amplitude - s_amplitude

            if (amplitude_diff > self.s_wave_min_amplitude and
                s_amplitude < r_amplitude * 0.7):  # S波应明显低于R峰
                s_waves.append(s_peak_candidate)

        return s_waves

    def detect_p_waves(self, r_peaks, signal_array):
        """
        检测P波
        P波在QRS波之前，代表心房去极化

        参数:
            r_peaks: R峰位置列表
            signal_array: 原始ECG信号数组

        返回:
            p_waves: P波位置列表
        """
        p_waves = []

        if not r_peaks or len(signal_array) == 0:
            return p_waves

        for r_peak in r_peaks:
            # 定义P波搜索窗口 (R峰前)
            search_start = max(0, r_peak - self.p_wave_search_start)
            search_end = max(0, r_peak - self.p_wave_search_end)

            if search_end <= search_start:
                continue

            # 在搜索窗口内找最大值点（P波通常是正向小波）
            window = signal_array[search_start:search_end]
            if len(window) == 0:
                continue

            max_idx = np.argmax(window)
            p_peak_candidate = search_start + max_idx

            # 获取P波和R峰幅值
            r_amplitude = signal_array[r_peak]
            p_amplitude = signal_array[p_peak_candidate]

            # 计算基线（窗口两端平均值）
            baseline_start = signal_array[search_start]
            baseline_end = signal_array[search_end - 1] if search_end < len(signal_array) else baseline_start
            baseline = (baseline_start + baseline_end) / 2

            # P波幅值（相对于基线）
            p_amplitude_from_baseline = p_amplitude - baseline

            # 验证P波特征:
            # 1. P波幅值应超过最小阈值
            # 2. P波应明显小于R峰
            if self.p_wave_min_amplitude < p_amplitude_from_baseline < r_amplitude * 0.25:  # P波应远小于R峰
                p_waves.append(p_peak_candidate)

        return p_waves

    def detect_t_waves(self, r_peaks, signal_array):
        """
        检测T波
        T波在QRS波之后，代表心室复极化

        参数:
            r_peaks: R峰位置列表
            signal_array: 原始ECG信号数组

        返回:
            t_waves: T波位置列表
        """
        t_waves = []

        if not r_peaks or len(signal_array) == 0:
            return t_waves

        for r_peak in r_peaks:
            # 定义T波搜索窗口 (R峰后)
            search_start = min(len(signal_array) - 1, r_peak + self.t_wave_search_start)
            search_end = min(len(signal_array), r_peak + self.t_wave_search_end)

            if search_end <= search_start:
                continue

            # 在搜索窗口内找最大值点（T波通常是正向宽波）
            window = signal_array[search_start:search_end]
            if len(window) == 0:
                continue

            max_idx = np.argmax(window)
            t_peak_candidate = search_start + max_idx

            # 获取T波和R峰幅值
            r_amplitude = signal_array[r_peak]
            t_amplitude = signal_array[t_peak_candidate]

            # 计算基线（窗口两端平均值）
            baseline_start = signal_array[search_start]
            baseline_end = signal_array[search_end - 1] if search_end < len(signal_array) else baseline_start
            baseline = (baseline_start + baseline_end) / 2

            # T波幅值（相对于基线）
            t_amplitude_from_baseline = t_amplitude - baseline

            # 验证T波特征:
            # 1. T波幅值应超过最小阈值
            # 2. T波通常小于R峰但大于P波
            if self.t_wave_min_amplitude < t_amplitude_from_baseline < r_amplitude * 0.6:  # T波应小于R峰
                t_waves.append(t_peak_candidate)

        return t_waves

    def detect_wave(self):
        """
        执行完整的ECG波形检测流程

        按Pan-Tomkins算法依次执行:
        1. 带通滤波 - 去除基线漂移和高频噪声
        2. 微分 - 突出QRS波的陡峭斜率
        3. 平方 - 使所有值为正并放大高斜率区域
        4. 移动窗口积分 - 平滑信号并提取QRS波特征
        5. R峰检测 - 使用自适应阈值检测
        6. 相位延迟补偿 - 补偿滤波引入的延迟
        7. Q波检测 - R峰前的负向波
        8. S波检测 - R峰后的负向波
        9. P波检测 - 心房去极化波
        10. T波检测 - 心室复极化波

        返回:
            r_peaks: R峰位置索引列表
        """

        # 将deque转换为numpy数组，以便进行数值运算
        signal_array = np.array(list(self.signal))

        # 步骤1: 带通滤波
        self.filtered_signal = self.bandpass_filter(signal_array)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: R峰检测
        self.qrs_peaks = self.threshold_detection(self.integrated_signal)

        # 步骤6: 相位延迟补偿
        self.qrs_peaks = self.apply_delay_compensation(self.qrs_peaks)

        # # 步骤7-10: 检测PQRST波
        # # 使用原始信号检测
        # self.q_waves = self.detect_q_waves(self.qrs_peaks, signal_array)
        # self.s_waves = self.detect_s_waves(self.qrs_peaks, signal_array)
        # self.p_waves = self.detect_p_waves(self.qrs_peaks, signal_array)
        # self.t_waves = self.detect_t_waves(self.qrs_peaks, signal_array)

        # 使用滤波后信号检测
        self.q_waves = self.detect_q_waves(self.qrs_peaks, self.filtered_signal)
        self.s_waves = self.detect_s_waves(self.qrs_peaks, self.filtered_signal)
        self.p_waves = self.detect_p_waves(self.qrs_peaks, self.filtered_signal)
        self.t_waves = self.detect_t_waves(self.qrs_peaks, self.filtered_signal)

        return self.qrs_peaks

    def update_signal_and_plot(self, samples):
        """
        更新信号缓冲区并刷新显示
        接收蓝牙回调的新数据并添加到信号队列中，进行波形检测和可视化

        参数:
            samples: 新接收的采样数据列表 (单位: mV)
        """
        sample_show_cnt = 0
        for sample in samples:
            sample_show_cnt += 1
            # 将新样本添加到deque中，自动淘汰旧数据，若在读取期间有所失常则用上一个数据源补充
            if len(self.signal) > 500 and sample  < 2.0:
                sample = self.signal[-1]
            self.signal.append(sample)

            if len(self.signal) > 500 and sample_show_cnt % 10 == 0:
                peaks = self.detect_wave()
                # print(f"R peaks: {peaks}")
                # print(f"Q waves: {self.q_waves}")
                # print(f"S waves: {self.s_waves}")
                # print(f"P waves: {self.p_waves}")
                # print(f"T waves: {self.t_waves}")

                # 转换为numpy数组方便计算ylim
                signal_array = np.array(list(self.signal))
                
                # 更新原始信号子图
                line1.set_ydata(signal_array)
                line1.set_xdata(range(len(signal_array)))
                ax1.set_ylim(np.min(signal_array), np.max(signal_array))

                # 准备5个子图的信号数据
                signals = [
                    signal_array,
                    self.filtered_signal,
                    self.differentiated_signal,
                    self.squared_signal,
                    self.integrated_signal
                ]
                lines = [line1, line2, line3, line4, line5]
                axes = [ax1, ax2, ax3, ax4, ax5]

                # 批量更新其他信号子图
                for i in range(1, 5):
                    if signals[i] is not None:
                        lines[i].set_ydata(signals[i])
                        lines[i].set_xdata(range(len(signals[i])))
                        axes[i].set_ylim(np.min(signals[i]), np.max(signals[i]))

                # 使用高效的scatter对象更新标记点（不需要清除和重新创建）
                # 更新R峰标记
                if len(peaks) > 0:
                    for i, scatter in enumerate(scatter_objects['r']):
                        if signals[i] is not None:
                            scatter.set_offsets(np.c_[peaks, signals[i][peaks]])
                else:
                    for scatter in scatter_objects['r']:
                        scatter.set_offsets(np.empty((0, 2)))

                # 更新Q波标记
                if len(self.q_waves) > 0:
                    for i, scatter in enumerate(scatter_objects['q']):
                        if signals[i] is not None:
                            scatter.set_offsets(np.c_[self.q_waves, signals[i][self.q_waves]])
                else:
                    for scatter in scatter_objects['q']:
                        scatter.set_offsets(np.empty((0, 2)))

                # 更新S波标记
                if len(self.s_waves) > 0:
                    for i, scatter in enumerate(scatter_objects['s']):
                        if signals[i] is not None:
                            scatter.set_offsets(np.c_[self.s_waves, signals[i][self.s_waves]])
                else:
                    for scatter in scatter_objects['s']:
                        scatter.set_offsets(np.empty((0, 2)))

                # 更新P波标记
                if len(self.p_waves) > 0:
                    for i, scatter in enumerate(scatter_objects['p']):
                        if signals[i] is not None:
                            scatter.set_offsets(np.c_[self.p_waves, signals[i][self.p_waves]])
                else:
                    for scatter in scatter_objects['p']:
                        scatter.set_offsets(np.empty((0, 2)))

                # 更新T波标记
                if len(self.t_waves) > 0:
                    for i, scatter in enumerate(scatter_objects['t']):
                        if signals[i] is not None:
                            scatter.set_offsets(np.c_[self.t_waves, signals[i][self.t_waves]])
                else:
                    for scatter in scatter_objects['t']:
                        scatter.set_offsets(np.empty((0, 2)))

                # 只对第一个子图更新布局（其他子图共享x轴，会自动更新）
                ax1.relim()
                ax1.autoscale_view(scalex=True, scaley=False)

                # 使用draw_idle()替代pause()，更高效
                fig.canvas.draw_idle()
                fig.canvas.flush_events()


class BlueToothCollector:
    def __init__(self, client=None):
        self.latest_samples = []
        self.data = []
        self.qrs_detector = RealTimeECGDetector(signal_name="MLII")

    def handle_disconnect(self, client):  # 断开连接回调函数
        print(f"设备已断开连接")

    def match_nus_device(self, device: BLEDevice, adv: AdvertisementData):
        # 优先通过MAC地址匹配
        if device.address == "EC:7A:26:9D:81:3F":
            print(f"通过MAC地址匹配到设备: {device.name or '未知'} ({device.address})")
            return True
        # 优先通过设备名称匹配
        if device.name and "AAA-TEST" in device.name:
            print(f"通过名称匹配到设备: {device.name} ({device.address})")
            return True
        # 如果名称和MAC地址都匹配失败，尝试UUID匹配
        if adv and adv.service_uuids and device_param["service_uuid"].lower() in [uuid.lower() for uuid in adv.service_uuids]:
            print(f"通过UUID匹配到设备: {device.name or '未知'} ({device.address})")
            print(f"  服务UUIDs: {adv.service_uuids}")
            return True
        return False

    def build_protocol_packet(self, func_code, data):
        """
        构建轻迅协议V1.0.1数据包
        格式: [功能码(2字节)] [数据长度(2字节)] [数据内容] [CRC16(2字节)]

        Args:
            func_code: 功能码 (int)
            data: 数据内容 (bytes/bytearray)

        Returns:
            完整的协议包 (bytearray)
        """
        packet = bytearray()
        # 1. 功能码 (2字节小端格式)
        packet.extend(struct.pack('<H', func_code))
        # 2. 数据长度 (2字节小端格式)
        packet.extend(struct.pack('<H', len(data)))
        # 3. 数据内容
        packet.extend(data)

        def calculate_crc16(data, offset=0, length=None):
            """
            计算CRC16-CCITT-FALSE校验值
            多项式: 0x1021
            初始值: 0xFFFF
            结果异或值: 0x0000
            输入输出反转: 无
            """

            wCRCin = 0xFFFF
            wCPoly = 0x1021

            for i in range(offset, offset + length):
                byte = data[i]
                for j in range(8):
                    bit = ((byte >> (7 - j)) & 1) == 1
                    c15 = ((wCRCin >> 15) & 1) == 1
                    wCRCin = wCRCin << 1
                    if c15 ^ bit:
                        wCRCin = wCRCin ^ wCPoly

            return wCRCin & 0xFFFF

        # 4. CRC检验
        crc_value = calculate_crc16(packet, 0, len(packet))
        packet.extend(struct.pack('<H', crc_value))

        return packet

    async def start_collection(self, client, collect_enable=1, timestamp=0):
        """
        开启采集
        功能码: 0x0001
        数据格式: [功能码(2字节)] [数据长度(2字节)] [数据内容 [采集开关(1字节)] [时间戳(8字节)] (9字节)] [CRC16(2字节)]
        """
        data = bytearray()
        # 采集开关(1字节)
        data.extend(struct.pack('B', collect_enable))
        # 时间戳(8字节)
        data.extend(struct.pack('<Q', timestamp))
        packet = self.build_protocol_packet(0x0001, data)
        print(f"发送开启采集指令: {[f'0x{b:02X}' for b in packet]}")
        respones = await client.write_gatt_char(device_param["rx_uuid"], packet)
        print("开始采集指令发送成功")
        return respones

    def packet_decode(self, data):
        # 只提取数据内容
        samples = []
        sample_data_start = 4  # 采样数据起始位置
        for i in range(119):
            sample_offset = sample_data_start + i * 2
            if sample_offset + 2 > len(data) - 2:  # 减去校验和
                break

            # 读取小端格式的16位整数
            sample_value = struct.unpack('<H', data[sample_offset:sample_offset + 2])[0]

            # 转换为电压值 (μV) - 单导联 0.288 12导联 0.318
            voltage_mV = sample_value * 0.288 / 1000.0
            samples.append(voltage_mV)

        return samples

    def handle_rx(self, sender, data): # 接收数据回调函数
        data_samples = self.packet_decode(data)
        data_samples = data_samples[3:-2]

        # 将数据传递给QRS检测器
        if len(data_samples) > 0:
            self.qrs_detector.update_signal_and_plot(data_samples)


async def main():
    # 首先扫描并输出所有附近的蓝牙设备
    print("正在扫描所有附近的蓝牙设备...")
    all_devices = await BleakScanner.discover(timeout=1.0)
    print(f"\n找到 {len(all_devices)} 个蓝牙设备:\n")

    for d in all_devices:
        print(f"设备名称: {d.name or '未知'}")
        print(f"MAC地址: {d.address}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("开始搜索目标设备...")

    # 搜索设备, 查看是否匹配NUS UUID，找到后可尝试建立连接，进行读写。
    Collector = BlueToothCollector()
    device = await BleakScanner.find_device_by_filter(Collector.match_nus_device)
    if not device:
        print("未找到目标设备")
        return
    else:
        print(f"\n成功找到设备: {device.address}")

    # 创建BleakClient客户端，连接后进行串口操作
    async with BleakClient(device, disconnected_callback=Collector.handle_disconnect) as client:
        # 发送开始监听指令
        await client.start_notify(device_param["tx_uuid"], Collector.handle_rx)
        print("Enable listening Callback Function")
        # 发送开始采集指令
        await Collector.start_collection(client, collect_enable=1, timestamp=0)
        print("Enable Collector Callback Function")
        # 持续接收数据的循环
        try:
            print("开始持续接收数据")
            while True:
                # 保持连接并等待数据
                await asyncio.sleep(0.01)  # 防止CPU占用过高，同时维持连接

        except KeyboardInterrupt:
            print("\n收到中断信号，正在断开连接...")
        except Exception as e:
            print(f"发生错误: {e}")
        finally:
            print("连接已断开")


if __name__ == "__main__":
    asyncio.run(main())