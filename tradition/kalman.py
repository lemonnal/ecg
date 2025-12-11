import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    """
    卡尔曼滤波器实现

    参数:
        dim_x: 状态向量维度
        dim_z: 观测向量维度
    """

    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z

        # 初始化矩阵
        self.x = np.zeros((dim_x, 1))        # 状态向量
        self.P = np.eye(dim_x)               # 状态协方差矩阵
        self.Q = np.eye(dim_x)               # 过程噪声协方差矩阵
        self.R = np.eye(dim_z)               # 观测噪声协方差矩阵
        self.F = np.eye(dim_x)               # 状态转移矩阵
        self.H = np.zeros((dim_z, dim_x))    # 观测矩阵

    def predict(self):
        """
        预测步骤
        预测下一个状态
        """
        # 状态预测: x_k|k-1 = F * x_k-1|k-1
        self.x = np.dot(self.F, self.x)

        # 协方差预测: P_k|k-1 = F * P_k-1|k-1 * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return self.x

    def update(self, z):
        """
        更新步骤
        使用观测值更新状态估计

        参数:
            z: 观测值向量
        """
        # 计算卡尔曼增益
        # S = H * P * H^T + R (新息协方差)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # K = P * H^T * S^(-1) (卡尔曼增益)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # 计算新息 (观测值 - 预测观测值)
        y = z - np.dot(self.H, self.x)

        # 状态更新: x_k|k = x_k|k-1 + K * y
        self.x = self.x + np.dot(K, y)

        # 协方差更新: P_k|k = (I - K * H) * P_k|k-1
        I_KH = np.eye(self.dim_x) - np.dot(K, self.H)
        self.P = np.dot(I_KH, self.P)

        return self.x


def apply_kalman_filter(signal_data, process_noise=0.01, measurement_noise=0.1,
                        initial_state_uncertainty=1.0, use_smoothing=False):
    """
    应用卡尔曼滤波器到一维信号

    参数:
        signal_data: 输入信号数组 (N,)
        process_noise: 过程噪声方差 q
        measurement_noise: 观测噪声方差 r
        initial_state_uncertainty: 初始状态不确定性
        use_smoothing: 是否使用RTS平滑 (Rauch-Tung-Striebel smoother)

    返回:
        filtered_signal: 滤波后的信号
        kalman_filter: 卡尔曼滤波器对象 (可用于进一步分析)
    """
    # 确保输入是一维数组
    if signal_data.ndim == 1:
        signal_data = signal_data.reshape(-1, 1)

    n_samples = signal_data.shape[0]

    # 创建卡尔曼滤波器 (状态维度=2: 位置和速度, 观测维度=1)
    kf = KalmanFilter(dim_x=2, dim_z=1)

    # 设置状态转移矩阵 (常速度模型)
    dt = 1.0  # 时间步长
    kf.F = np.array([[1, dt],
                     [0, 1]])

    # 设置观测矩阵 (只能观测位置)
    kf.H = np.array([[1, 0]])

    # 设置噪声协方差矩阵
    kf.Q = process_noise * np.array([[dt**3/3, dt**2/2],
                                     [dt**2/2, dt]])  # 过程噪声
    kf.R = measurement_noise * np.eye(1)  # 观测噪声
    kf.P = initial_state_uncertainty * np.eye(2)  # 初始不确定性

    # 存储滤波结果
    filtered_states = []
    predictions = []

    # 前向滤波
    for i in range(n_samples):
        # 预测
        prediction = kf.predict()
        predictions.append(prediction[0, 0])

        # 更新
        measurement = np.array([[signal_data[i, 0]]])
        state = kf.update(measurement)
        filtered_states.append(state[0, 0])

    filtered_signal = np.array(filtered_states)

    if use_smoothing and n_samples > 1:
        # RTS平滑 (可选)
        smoothed_signal = apply_rts_smoother(kf, signal_data, filtered_states, predictions)
        return smoothed_signal, kf
    else:
        return filtered_signal, kf


def apply_rts_smoother(kf, measurements, filtered_states, predictions):
    """
    应用Rauch-Tung-Striebel (RTS) 平滑器

    参数:
        kf: 卡尔曼滤波器对象
        measurements: 原始测量值
        filtered_states: 前向滤波的状态估计
        predictions: 预测值

    返回:
        smoothed_signal: 平滑后的信号
    """
    n_samples = len(measurements)

    # 重新运行滤波过程，存储中间结果
    x_history = []
    P_history = []

    # 重置滤波器
    kf.x = np.zeros((2, 1))
    kf.P = np.eye(2)

    # 前向滤波并存储历史
    for i in range(n_samples):
        kf.predict()
        measurement = np.array([[measurements[i, 0]]])
        kf.update(measurement)
        x_history.append(kf.x.copy())
        P_history.append(kf.P.copy())

    # 反向平滑
    x_smooth = x_history[-1].copy()
    smoothed_signal = np.zeros(n_samples)
    smoothed_signal[-1] = x_smooth[0, 0]

    for i in range(n_samples - 2, -1, -1):
        # 平滑增益
        C = np.dot(np.dot(x_history[i], kf.F.T), np.linalg.inv(kf.P))

        # 平滑状态
        x_smooth = x_history[i] + np.dot(C, x_smooth - np.dot(kf.F, x_history[i]))

        smoothed_signal[i] = x_smooth[0, 0]

    return smoothed_signal


def analyze_kalman_performance(original_signal, filtered_signal, window_size=100):
    """
    分析卡尔曼滤波器的性能

    参数:
        original_signal: 原始信号
        filtered_signal: 滤波后信号
        window_size: 窗口大小用于计算局部方差
    返回:
        performance_metrics: 性能指标字典
    """
    # 计算均方误差
    mse = np.mean((original_signal - filtered_signal) ** 2)

    # 计算信噪比改善
    original_power = np.var(original_signal)
    noise_power = np.var(original_signal - filtered_signal)
    snr_improvement = 10 * np.log10(original_power / noise_power) if noise_power > 0 else float('inf')

    # 计算信号平滑度 (相邻样本差的方差)
    original_smoothness = np.var(np.diff(original_signal))
    filtered_smoothness = np.var(np.diff(filtered_signal))
    smoothness_ratio = original_smoothness / filtered_smoothness if filtered_smoothness > 0 else float('inf')

    # 计算局部方差变化
    if len(original_signal) > window_size:
        original_local_var = []
        filtered_local_var = []

        for i in range(0, len(original_signal) - window_size, window_size // 2):
            window_original = original_signal[i:i + window_size]
            window_filtered = filtered_signal[i:i + window_size]
            original_local_var.append(np.var(window_original))
            filtered_local_var.append(np.var(window_filtered))

        variance_reduction = np.mean(original_local_var) / np.mean(filtered_local_var)
    else:
        variance_reduction = 1.0

    performance_metrics = {
        'mse': mse,
        'snr_improvement_db': snr_improvement,
        'smoothness_improvement': smoothness_ratio,
        'variance_reduction': variance_reduction,
        'original_std': np.std(original_signal),
        'filtered_std': np.std(filtered_signal)
    }

    return performance_metrics


# 读取数据文件
data_path = 'mit-bih-dataset/ecg_102.txt'

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

# 应用卡尔曼滤波器
print("Applying Kalman filter to ECG signals...")

# 定义多个参数配置方案
configurations = {
    'default': {
        'process_noise': 0.01,
        'measurement_noise': 0.1,
        'initial_state_uncertainty': 1.0,
        'description': '默认配置 - 通用平衡'
    },
    'smooth': {
        'process_noise': 0.005,
        'measurement_noise': 0.5,
        'initial_state_uncertainty': 2.0,
        'description': '平滑优先 - 更多平滑，但响应较慢'
    },
    'responsive': {
        'process_noise': 0.02,
        'measurement_noise': 0.05,
        'initial_state_uncertainty': 0.5,
        'description': '响应优先 - 快速跟踪变化'
    },
    'very_smooth': {
        'process_noise': 0.001,
        'measurement_noise': 1.0,
        'initial_state_uncertainty': 5.0,
        'description': '极度平滑 - 适用于噪声很大的信号'
    },
    'minimal_smoothing': {
        'process_noise': 0.05,
        'measurement_noise': 0.02,
        'initial_state_uncertainty': 0.1,
        'description': '最小平滑 - 保持原始信号特征'
    },
    'adaptive_ecg': {
        'process_noise': 0.008,
        'measurement_noise': 0.08,
        'initial_state_uncertainty': 0.8,
        'description': 'ECG优化 - 专门针对心电信号'
    }
}

# 使用默认配置
current_config = 'adaptive_ecg'
config = configurations[current_config]
process_noise = config['process_noise']
measurement_noise = config['measurement_noise']
initial_state_uncertainty = config['initial_state_uncertainty']

print(f"\n可用的Kalman滤波器配置:")
for key, cfg in configurations.items():
    marker = "→ " if key == current_config else "  "
    print(f"{marker}{key}: {cfg['description']}")
    print(f"    process_noise={cfg['process_noise']}, measurement_noise={cfg['measurement_noise']}")

print(f"\n当前使用配置: {current_config}")
print(f"Process Noise: {process_noise}, Measurement Noise: {measurement_noise}")
print(f"Initial State Uncertainty: {initial_state_uncertainty}")

  # 对两列信号分别应用卡尔曼滤波器
filtered_column1, kf1 = apply_kalman_filter(
    column1,
    process_noise=process_noise,
    measurement_noise=measurement_noise
)
filtered_column2, kf2 = apply_kalman_filter(
    column2,
    process_noise=process_noise,
    measurement_noise=measurement_noise
)

print("Kalman filtering completed!")

# 分析滤波性能
perf1 = analyze_kalman_performance(column1, filtered_column1)
perf2 = analyze_kalman_performance(column2, filtered_column2)

print(f"\nColumn 1 Performance:")
print(f"  MSE: {perf1['mse']:.6f}")
print(f"  SNR Improvement: {perf1['snr_improvement_db']:.2f} dB")
print(f"  Smoothness Improvement: {perf1['smoothness_improvement']:.2f}x")

print(f"\nColumn 2 Performance:")
print(f"  MSE: {perf2['mse']:.6f}")
print(f"  SNR Improvement: {perf2['snr_improvement_db']:.2f} dB")
print(f"  Smoothness Improvement: {perf2['smoothness_improvement']:.2f}x")

# 创建滤波前后的对比图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Kalman Filter ECG Signal Processing', fontsize=16)

# 第一列：原始信号 vs 滤波后信号 (前1000个样本)
ax1.plot(column1[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
ax1.plot(filtered_column1[:1000], 'r-', label='Kalman Filtered', linewidth=2)
ax1.set_title('Column 1 - Original vs Kalman Filtered (First 1000 samples)')
ax1.set_xlabel('Sample Index')
ax1.set_ylabel('Amplitude')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 第二列：原始信号 vs 滤波后信号 (前1000个样本)
ax2.plot(column2[:1000], 'b-', alpha=0.7, label='Original', linewidth=1)
ax2.plot(filtered_column2[:1000], 'r-', label='Kalman Filtered', linewidth=2)
ax2.set_title('Column 2 - Original vs Kalman Filtered (First 1000 samples)')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Amplitude')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 第一列：详细对比 (前200个样本，放大显示)
ax3.plot(column1[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
ax3.plot(filtered_column1[:200], 'r-', label='Kalman Filtered', linewidth=2)
ax3.set_title('Column 1 - Detailed Comparison (First 200 samples)')
ax3.set_xlabel('Sample Index')
ax3.set_ylabel('Amplitude')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 第二列：详细对比 (前200个样本，放大显示)
ax4.plot(column2[:200], 'b-', alpha=0.7, label='Original', linewidth=1)
ax4.plot(filtered_column2[:200], 'r-', label='Kalman Filtered', linewidth=2)
ax4.set_title('Column 2 - Detailed Comparison (First 200 samples)')
ax4.set_xlabel('Sample Index')
ax4.set_ylabel('Amplitude')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 打印基本信息
print(f"\n数据信息:")
print(f"数据形状: {data.shape}")
print(f"第一列范围: {np.min(column1):.6f} 到 {np.max(column1):.6f}")
print(f"第二列范围: {np.min(column2):.6f} 到 {np.max(column2):.6f}")
print(f"滤波后第一列范围: {np.min(filtered_column1):.6f} 到 {np.max(filtered_column1):.6f}")
print(f"滤波后第二列范围: {np.min(filtered_column2):.6f} 到 {np.max(filtered_column2):.6f}")

# 打印滤波器参数信息
print(f"\n卡尔曼滤波器参数:")
print(f"过程噪声协方差矩阵 Q:\n{kf1.Q}")
print(f"观测噪声协方差矩阵 R:\n{kf1.R}")
print(f"状态转移矩阵 F:\n{kf1.F}")
print(f"观测矩阵 H:\n{kf1.H}")

print("\n" + "="*50)
print("卡尔曼滤波器应用完成!")
print("="*50)