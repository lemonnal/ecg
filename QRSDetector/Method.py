from collections import deque
import numpy as np
from scipy import signal as scipy_signal
import wfdb


def get_signal_params(signal_name):
    # 基于导联特性的参数
    if signal_name == 'V1':
        signal_params = {
            'low': 1, 'high': 50.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.2,
        }
    elif signal_name == 'V2':
        signal_params = {
            'low': 3, 'high': 30.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.3
        }
    elif signal_name == 'V3':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'V4':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'V5':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'V6':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'I':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'MLII':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'MLIII':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'aVR':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'aVL':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    elif signal_name == 'aVF':
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
    else:
        signal_params = {
            'low': 5, 'high': 15.0, 'filter_order': 5, 'original_weight': 0.2, 'filtered_weight': 0.8,
            'integration_window_size': 0.080,
            'detection_window_size': 8.0, 'overlap_window_size': 4.0, 'refractory_period': 0.20,
            'threshold_factor': 1.4
        }
        
    return signal_params

class PanTomkinsQRSDetectorOffline:
    """
    基于Pan-Tomkins算法的QRS波检测器
    """

    def __init__(self, signal_name="MLII"):
        """
        初始化QRS检测器

        参数:
            fs: 采样频率 (Hz)
            signal_name: ECG导联名称 (如 "MLII", "V1", "V2" 等)
        """
        self.fs = 360
        self.signal = None
        self.filtered_signal = None
        self.differentiated_signal = None
        self.squared_signal = None
        self.integrated_signal = None
        self.qrs_peaks = []
        self.params = get_signal_params(signal_name=signal_name)

    def bandpass_filter(self, signal_data):
        """
        自适应带通滤波器
        根据不同导联使用不同的频率参数

        参数:
            signal_data: 输入ECG信号

        返回:
            combined_signal: 滤波后与原始信号加权组合的信号
        """
        # 获取该导联的滤波参数

        # 设计带通滤波器
        nyquist = 0.5 * self.fs
        low = self.params['low'] / nyquist
        high = self.params['high'] / nyquist
        order = self.params['filter_order']

        # 使用 n 阶 Butterworth 滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(order, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # 添加原始信号的加权
        combined_signal = (self.params["original_weight"] * signal_data
                           + self.params["filtered_weight"] * filtered_signal)
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
            differentiated_signal[i] = (-signal_data[i + 2] + 8 * signal_data[i + 1]
                                        - 8 * signal_data[i - 1] + signal_data[i - 2]) / 12

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
        window_sample = int(self.params['integration_window_size'] * self.fs)

        # 使用卷积实现高效的移动平均积分
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
        window_size = int(self.params['detection_window_size'] * self.fs)  # 检测窗口大小 (秒)
        overlap_size = int(self.params['overlap_window_size'] * self.fs)    # 重叠窗口大小 (秒)

        # 设置不应期 (避免同一QRS波被重复检测)
        refractory_period = int(self.params['refractory_period'] * self.fs)  # 不应期（秒）

        # 获取该导联的阈值系数
        threshold_factor = self.params['threshold_factor']

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
            current_threshold = window_mean + threshold_factor * window_std

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

    def detect_qrs_peaks(self, signal_data):
        """
        检测QRS波峰值

        参数:
            signal_data: 输入ECG信号

        返回:
            qrs_peaks: QRS波峰值位置索引
        """

        # 步骤1: 带通滤波
        self.filtered_signal = self.bandpass_filter(signal_data)

        # 步骤2: 微分
        self.differentiated_signal = self.derivative(self.filtered_signal)

        # 步骤3: 平方
        self.squared_signal = self.squaring(self.differentiated_signal)

        # 步骤4: 移动窗口积分
        self.integrated_signal = self.moving_window_integration(self.squared_signal)

        # 步骤5: QRS检测
        self.qrs_peaks = self.threshold_detection(self.integrated_signal)

        return self.qrs_peaks


class PanTomkinsQRSDetectorOnline:
    """
    基于Pan-Tomkins算法的实时QRS波检测器
    """

    def __init__(self, signal_name="MLII"):
        """
        初始化QRS检测器

        参数:
            fs: 采样频率 (Hz)
            signal_name: ECG导联名称 (如 "MLII", "V1", "V2" 等)
        """
        self.fs = 360
        self.signal_len = 300
        self.signal = deque([0], self.signal_len)
        self.filtered_signal = None
        self.differentiated_signal = None
        self.squared_signal = None
        self.integrated_signal = None
        self.qrs_peaks = []
        self.params = get_signal_params(signal_name=signal_name)

    def update_signal(self, samples):
        """
        更新信号缓冲区
        接收蓝牙回调的新数据并添加到信号队列中

        参数:
            samples: 新接收的采样数据列表 (单位: mV)
        """
        # 将新样本添加到deque中，自动淘汰旧数据
        for sample in samples:
            self.signal.append(sample)
        print(len(self.signal))
        print(self.signal)

        # 可选: 当信号缓冲区更新时自动进行QRS检测
        # self.detect_qrs_peaks()

    def bandpass_filter(self, signal_data):
        """
        自适应带通滤波器
        根据不同导联使用不同的频率参数

        参数:
            signal_data: 输入ECG信号

        返回:
            combined_signal: 滤波后与原始信号加权组合的信号
        """
        # 获取该导联的滤波参数

        # 设计带通滤波器
        nyquist = 0.5 * self.fs
        low = self.params['low'] / nyquist
        high = self.params['high'] / nyquist
        order = self.params['filter_order']

        # 使用 n 阶 Butterworth 滤波器 - 平衡滤波效果和信号保留
        b, a = scipy_signal.butter(order, [low, high], btype='band')

        # 应用零相位滤波
        filtered_signal = scipy_signal.filtfilt(b, a, signal_data)

        # 添加原始信号的加权
        combined_signal = (self.params["original_weight"] * signal_data
                           + self.params["filtered_weight"] * filtered_signal)
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
            differentiated_signal[i] = (-signal_data[i + 2] + 8 * signal_data[i + 1]
                                        - 8 * signal_data[i - 1] + signal_data[i - 2]) / 12

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
        window_sample = int(self.params['integration_window_size'] * self.fs)

        # 使用卷积实现高效的移动平均积分
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
        window_size = int(self.signal_len)  # 检测窗口即为信号窗口
        overlap_size = int(self.signal_len / 2)    # 重叠窗口大小 (秒)

        # 设置不应期 (避免同一QRS波被重复检测)
        refractory_period = int(self.params['refractory_period'] * self.fs)  # 不应期（秒）

        # 获取该导联的阈值系数
        threshold_factor = self.params['threshold_factor']

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
            current_threshold = window_mean + threshold_factor * window_std

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

    def detect_qrs_peaks(self):
        """
        检测QRS波峰值

        参数:
            signal_data: 输入ECG信号

        返回:
            qrs_peaks: QRS波峰值位置索引
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

        # 步骤5: QRS检测
        self.qrs_peaks = self.threshold_detection(self.integrated_signal)

        return self.qrs_peaks

import asyncio
from bleak import BleakScanner
from bleak import BleakClient
from bleak.backends.device import BLEDevice
from bleak.backends.scanner import AdvertisementData
import struct
import matplotlib.pyplot as plt
from collections import deque
import struct
QINGXUN_UART_SERVICE_UUID = "6e400001-b5a3-f393-e0a9-68716563686f"
QINGXUN_UART_RX_CHAR_UUID = "6e400002-b5a3-f393-e0a9-68716563686f"
QINGXUN_UART_TX_CHAR_UUID = "6e400003-b5a3-f393-e0a9-68716563686f"
device = "AAA-TEST"
if device == "AAA-TEST":
    device_param = {
        "name": device,
        "address": "EC:7A:26:9D:81:3F",
        "service_uuid": QINGXUN_UART_SERVICE_UUID,
        "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
        "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
    }
elif device == "PW-ECG-SL":
    device_param = {
        "name": device,
        "address": "E2:1B:A5:DB:DE:EA",
        "service_uuid": QINGXUN_UART_SERVICE_UUID,
        "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
        "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
    }
voltage_mV_max = -0xffffff
voltage_mV_min = 0xffffff
# 创建空的数据列表
data_list = deque(maxlen=300)  # 保持最新的200个数据点
# 创建一个图形窗口
plt.ion()  # 开启交互模式
fig, ax = plt.subplots()
line, = ax.plot([])
ax.set_ylim(0.0, 0.05)  # 设置y轴范围

class QingXunBlueToothCollector:
    def __init__(self, client=None):
        self.data_queue = deque(maxlen=300)
        self.latest_samples = []
        self.data = []
        self.qrs_detector = PanTomkinsQRSDetectorOnline()

    # 断开连接回调函数
    def handle_disconnect(self, client):
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
    # 接收数据回调函数
    def handle_rx(self, sender, data):
        global voltage_mV_max
        global voltage_mV_min

        data_samples = self.packet_decode(data)

        # 将数据传递给QRS检测器
        if len(data_samples) > 0:
            self.qrs_detector.update_signal(data_samples)

        try:
            voltage_mV_max = max(data_samples) if voltage_mV_max < max(data_samples) else voltage_mV_max
            voltage_mV_min = min(data_samples) if voltage_mV_min > min(data_samples) else voltage_mV_min
            # print(voltage_mV_max, voltage_mV_min)
            voltage_delta = voltage_mV_max - voltage_mV_min

            for sample in data_samples:
                value = sample
                data_list.append(value)
                line.set_ydata(data_list)
                line.set_xdata(range(len(data_list)))
                ax.set_ylim(voltage_mV_min - 0.2 * voltage_delta, voltage_mV_max + 0.2 * voltage_delta)
                ax.relim()
                ax.autoscale_view()
                plt.pause(0.01)  # 更新图形，可以根据需要调整刷新频率
        except ValueError:
            pass


async def main():
    # 首先扫描并输出所有附近的蓝牙设备
    print("正在扫描所有附近的蓝牙设备...")
    all_devices = await BleakScanner.discover(timeout=10.0)
    print(f"\n找到 {len(all_devices)} 个蓝牙设备:\n")

    for d in all_devices:
        print(f"设备名称: {d.name or '未知'}")
        print(f"MAC地址: {d.address}")
        print("-" * 60)

    print("\n" + "=" * 60)
    print("开始搜索目标设备...")

    # 搜索设备, 查看是否匹配NUS UUID，找到后可尝试建立连接，进行读写。
    Collector = QingXunBlueToothCollector()
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
    off_line = 0
    on_line = 1

    if off_line:
        # MIT-BIH - 单通道检测
        # 路径
        root = "/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/"
        # numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
        #              '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
        #              '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
        #              '231', '232', '233', '234']
        numberSet = ['100']

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
                # 创建QRS检测器实例
                qrs_detector = PanTomkinsQRSDetectorOffline(signal_name=target_lead)
                # 进行QRS检测
                qrs_peaks = qrs_detector.detect_qrs_peaks(signal)

                print(np.array(sig_sample))
                print(np.array(sig_sample).shape)
                print(np.array(qrs_peaks))
                print(np.array(qrs_peaks).shape)

            except Exception as e:
                print(e)
    elif on_line:
        asyncio.run(main())



