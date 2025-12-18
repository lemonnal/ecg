import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
import wfdb


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
            'aVR': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'aVL': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'aVF': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            # 肢体导联-标准双极导联
            'I': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'MLII': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'MLIII': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},

            # 胸前导联 -
            # V1特殊处理
            'V1': {'low': 1, 'high': 50.0, 'threshold_factor': 1.2},
            # V1导联特点：R波小，S波深，需要更低频率捕获，更高频率保留细节

            # 胸前导联 - 过渡区
            'V2': {'low': 3, 'high': 30.0, 'threshold_factor': 1.3},
            # V2导联特点：介于V1和V3之间，中等参数

            # 胸前导联 - 左心前区
            'V3': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'V4': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'V5': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
            'V6': {'low': 5, 'high': 15.0, 'threshold_factor': 1.4},
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

if __name__ == "__main__":
    PanTomkinsQRSDetector()