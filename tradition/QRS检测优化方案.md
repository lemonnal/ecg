# QRS检测率优化方案

## 当前检测结果分析
- **总体检测率**: 84.32% (169322/200812)
- **MLII导联**: 90.55% - 相对较好
- **V1导联**: 75.82% - 问题最严重
- **其他导联**: 92-99% - 良好

## 主要问题识别
1. **V1导联检测率过低** (75.82%)
2. **滤波器参数可能不适用于所有导联**
3. **阈值设置可能过于保守**
4. **不应期设置可能不适合某些心率变化**

## 优化方案

### 方案1: 调整滤波器参数
**目标**: 改善V1导联的信号质量

**具体修改**:
```python
# 在bandpass_filter方法中
# 原参数: low=0.5, high=40.0 Hz
# 建议修改为:
low = 0.5 / nyquist      # 保持不变
high = 45.0 / nyquist    # 提高到45Hz以保留更多高频信息

# 或者针对不同导联使用不同参数
if "V1" in signal_name or "V2" in signal_name:
    low = 0.3 / nyquist   # 降低低频截止
    high = 50.0 / nyquist # 提高高频截止
else:
    low = 0.5 / nyquist
    high = 40.0 / nyquist
```

**原理**: V1导联的QRS波形态可能与MLII不同，需要调整频率范围。

### 方案2: 优化阈值检测算法
**目标**: 提高检测敏感性

**具体修改**:
```python
# 在_threshold_detection方法中
# 原参数: threshold = signal_mean + 1.5 * signal_std
# 建议修改为:
threshold = signal_mean + 1.2 * signal_std  # 降低阈值

# 或使用动态阈值
signal_median = np.median(self.integrated_signal)
threshold = signal_median + 1.0 * signal_std  # 基于中位数

# 或使用分位数法
threshold = np.percentile(self.integrated_signal, 85)  # 85分位数作为阈值
```

### 方案3: 调整不应期参数
**目标**: 减少漏检

**具体修改**:
```python
# 原参数: refractory_period = int(0.2 * self.fs)  # 200ms
# 建议修改为:
refractory_period = int(0.15 * self.fs)  # 150ms，支持更高心率

# 或自适应不应期
rr_avg = 0.8  # 默认平均RR间期(秒)
if len(peaks) >= 2:
    rr_avg = np.mean(np.diff(peaks[-2:])) / self.fs
refractory_period = int(rr_avg * 0.3 * self.fs)  # 30%的平均RR间期
```

### 方案4: 添加信号预处理
**目标**: 改善信号质量

**具体修改**:
```python
def preprocess_signal(self, signal_data):
    """信号预处理"""
    # 去除基线漂移
    from scipy.signal import medfilt
    baseline = medfilt(signal_data, kernel_size=201)
    signal_no_baseline = signal_data - baseline

    # 去除工频干扰
    from scipy.signal import iirnotch
    b_notch, a_notch = iirnotch(60, 30, self.fs)  # 60Hz工频干扰
    signal_clean = scipy_signal.filtfilt(b_notch, a_notch, signal_no_baseline)

    return signal_clean
```

### 方案5: 改进峰值检测逻辑
**目标**: 更准确的峰值定位

**具体修改**:
```python
# 在阈值检测中添加更严格的峰值验证
def _is_valid_peak(self, signal, peak_idx, window_size=5):
    """验证是否为有效峰值"""
    # 检查局部最大值
    window_start = max(0, peak_idx - window_size)
    window_end = min(len(signal), peak_idx + window_size + 1)
    window = signal[window_start:window_end]

    # 检查是否为局部最大值
    local_max = np.argmax(window)
    if local_max != peak_idx - window_start:
        return False

    # 检查峰值的显著性
    noise_floor = np.mean(signal[window_start:window_end])
    peak_height = signal[peak_idx] - noise_floor

    return peak_height > (np.std(signal) * 0.5)
```

### 方案6: 多尺度检测
**目标: 不同形态的QRS波都能检测**

**具体修改**:
```python
def multi_scale_detection(self, signal_data):
    """多尺度QRS检测"""
    all_peaks = []

    # 不同窗口大小的积分
    window_sizes = [int(0.06*self.fs), int(0.08*self.fs), int(0.10*self.fs)]

    for window_size in window_sizes:
        integrated = self.moving_window_integration(signal_data, window_size)
        peaks = self._find_peaks(integrated, threshold_factor=1.2)
        all_peaks.extend(peaks)

    # 合并并去重
    all_peaks = sorted(list(set(all_peaks)))
    refined_peaks = self._merge_close_peaks(all_peaks, min_distance=int(0.1*self.fs))

    return refined_peaks
```

### 方案7: 导联自适应参数
**目标: 针对不同导联使用最优参数**

**具体修改**:
```python
def get_adaptive_params(self, lead_name):
    """根据导联获取自适应参数"""
    params = {
        'MLII': {'threshold_factor': 1.5, 'refractory': 0.2, 'filter_low': 0.5, 'filter_high': 40},
        'V1':   {'threshold_factor': 1.2, 'refractory': 0.15, 'filter_low': 0.3, 'filter_high': 50},
        'V2':   {'threshold_factor': 1.3, 'refractory': 0.18, 'filter_low': 0.4, 'filter_high': 45},
        'V4':   {'threshold_factor': 1.4, 'refractory': 0.19, 'filter_low': 0.5, 'filter_high': 42},
        'V5':   {'threshold_factor': 1.4, 'refractory': 0.2, 'filter_low': 0.5, 'filter_high': 40},
    }
    return params.get(lead_name, params['MLII'])
```

### 方案8: 后处理优化
**目标: 通过后处理减少误检和漏检**

**具体修改**:
```python
def post_process_peaks(self, peaks, signal):
    """后处理优化检测结果"""
    # 1. 移除过密的峰值（误检）
    min_distance = int(0.25 * self.fs)  # 250ms最小间距
    filtered_peaks = []

    for peak in peaks:
        if not filtered_peaks or (peak - filtered_peaks[-1]) >= min_distance:
            filtered_peaks.append(peak)

    # 2. 在长间隔中搜索漏检的峰值
    for i in range(len(filtered_peaks) - 1):
        interval = filtered_peaks[i+1] - filtered_peaks[i]
        if interval > int(1.5 * self.fs):  # 1.5秒长间隔
            missed = self._search_missing_peaks(signal, filtered_peaks[i], filtered_peaks[i+1])
            filtered_peaks.extend(missed)

    return sorted(filtered_peaks)
```

### 方案9: 机器学习方法
**目标: 使用简单的机器学习提高检测准确性**

**具体修改**:
```python
def ml_enhanced_detection(self, signal_data, peaks):
    """使用机器学习增强检测结果"""
    features = []
    for peak in peaks:
        # 提取特征
        window_start = max(0, peak - int(0.05*self.fs))
        window_end = min(len(signal_data), peak + int(0.05*self.fs))
        window = signal_data[window_start:window_end]

        # 计算特征
        features.append([
            np.max(window),           # 峰值
            np.min(window),           # 谷值
            np.ptp(window),          # 峰峰值
            np.std(window),          # 标准差
            len(window),             # 宽度特征
        ])

    # 使用简单的规则分类
    # 这里可以训练一个简单的分类器或使用启发式规则
    valid_peaks = []
    for i, feat in enumerate(features):
        if feat[0] > 0.5 and feat[2] > 0.3:  # 示例阈值
            valid_peaks.append(peaks[i])

    return valid_peaks
```

### 方案10: 综合优化策略
**实施顺序**:
1. 首先尝试方案1和方案2（滤波器和阈值）
2. 如果V1导联仍不理想，实施方案7（导联自适应）
3. 添加方案8（后处理优化）
4. 最后考虑更复杂的方案（如方案6多尺度检测）

## 预期改进效果
- **V1导联**: 从75.82%提升到85-90%
- **MLII导联**: 从90.55%提升到94-96%
- **总体检测率**: 从84.32%提升到90-92%

## 注意事项
1. 修改参数后需要在所有数据集上测试
2. 注意避免过度拟合（误检率增加）
3. 不同导联的QRS形态差异很大，需要针对性优化
4. 保留原始算法作为对比基准

## 测试建议
1. 逐个方案测试，记录改进效果
2. 重点关注V1导联的改进
3. 平衡检测率和误检率
4. 使用可视化工具分析漏检案例