# 希尔伯特变换QRS检测算法 - 详细数学公式分析

## 目录
1. [算法概述](#算法概述)
2. [带通滤波器](#带通滤波器)
3. [希尔伯特变换与包络提取](#希尔伯特变换与包络提取)
4. [包络平滑处理](#包络平滑处理)
5. [自适应阈值检测算法](#自适应阈值检测算法)
6. [回溯搜索检测](#回溯搜索检测)
7. [峰值精确定位](#峰值精确定位)
8. [心率计算](#心率计算)
9. [算法对比分析](#算法对比分析)
10. [算法流程总结](#算法流程总结)

## 算法概述

希尔伯特变换QRS检测算法利用希尔伯特变换提取信号的解析信号和包络，通过包络的峰值检测QRS波。该算法对基线漂移不敏感，计算简单，实时性好。

主要处理步骤：
1. **带通滤波** (5-40 Hz)
2. **希尔伯特变换** → 计算解析信号
3. **包络提取** → 计算信号幅度
4. **包络平滑** → 减少噪声影响
5. **自适应阈值检测** → 检测QRS复合波

---

## 带通滤波器

### 代码位置：[`hilbert_qrs.py:35-62`](hilbert_qrs.py#L35-L62)

```python
def bandpass_filter(self, signal_data):
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
```

### 数学公式推导：

#### 1. 采样和奈奎斯特频率
采样频率为 $f_s = 360$ Hz。

**奈奎斯特频率：**
$$f_N = \frac{f_s}{2} = \frac{360}{2} = 180 \text{ Hz}$$

#### 2. 归一化频率
**低频截止：**
$$\omega_{low} = \frac{f_{low}}{f_N} = \frac{5}{180} = 0.0278$$

**高频截止：**
$$\omega_{high} = \frac{f_{high}}{f_N} = \frac{40}{180} = 0.2222$$

#### 3. Butterworth滤波器传递函数

s域中的3阶Butterworth滤波器传递函数：

$$H(s) = \frac{1}{\sqrt{1 + \left(\frac{s}{\omega_c}\right)^{2n}}}$$

其中：
- $n = 3$ 是滤波器阶数
- $\omega_c$ 是截止频率
- $s = j\omega$ 是复频率变量

#### 4. 零相位滤波

零相位滤波应用两次滤波（前向和后向）：

$$y_{final}[n] = \text{filtfilt}(b, a, x[n])$$

#### 5. 信号加权混合

**混合信号公式：**
$$y_{mixed}[n] = w_{orig} \cdot x[n] + w_{filt} \cdot y_{filtered}[n]$$

其中：
- $w_{orig} = 0.2$ 是原始信号权重
- $w_{filt} = 0.8$ 是滤波信号权重

**加权目的：**
- 保留原始信号的部分特征，减少漏检
- 平衡滤波效果和信号保真度

---

## 希尔伯特变换与包络提取

### 代码位置：[`hilbert_qrs.py:64-89`](hilbert_qrs.py#L64-L89)

```python
def compute_hilbert_envelope(self, signal_data):
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
```

### 数学公式推导：

#### 1. 希尔伯特变换定义

**连续时间希尔伯特变换：**
$$\mathcal{H}\{x(t)\} = \frac{1}{\pi} \text{PV} \int_{-\infty}^{\infty} \frac{x(\tau)}{t-\tau} d\tau$$

其中PV表示柯西主值积分。

#### 2. 解析信号

**解析信号定义：**
$$z(t) = x(t) + j\mathcal{H}\{x(t)\} = A(t)e^{j\phi(t)}$$

其中：
- $x(t)$ 是原始实信号
- $\mathcal{H}\{x(t)\}$ 是希尔伯特变换
- $A(t)$ 是瞬时幅度（包络）
- $\phi(t)$ 是瞬时相位

#### 3. 包络提取

**包络（瞬时幅度）：**
$$A(t) = |z(t)| = \sqrt{x^2(t) + \mathcal{H}^2\{x(t)\}}$$

**离散实现：**
$$A[n] = \sqrt{x^2[n] + \hat{x}^2[n]}$$

其中 $\hat{x}[n]$ 是离散希尔伯特变换。

#### 4. 希尔伯特变换的频域特性

**频域响应：**
$$H_{Hilbert}(\omega) = -j \cdot \text{sgn}(\omega)$$

其中：
$$\text{sgn}(\omega) = \begin{cases}
1 & \omega > 0 \\
0 & \omega = 0 \\
-1 & \omega < 0
\end{cases}$$

**物理意义：**
- 正频率分量相位偏移 $-90°$
- 负频率分量相位偏移 $+90°$
- 直流分量为零

#### 5. 离散希尔伯特变换实现

**FFT方法：**
$$X[k] = \text{FFT}\{x[n]\}$$

$$H[k] = \begin{cases}
2 & 0 < k < \frac{N}{2} \\
1 & k = 0 \text{ 或 } k = \frac{N}{2} \\
0 & \text{其他} \\
-2 & \frac{N}{2} < k < N
\end{cases}$$

$$z[n] = \text{IFFT}\{X[k] \cdot H[k]\}$$

---

## 包络平滑处理

### 代码位置：[`hilbert_qrs.py:84-87`](hilbert_qrs.py#L84-L87)

```python
# 平滑包络以减少噪声影响
window_size = int(0.01 * self.fs)  # 10ms平滑窗口
if window_size > 1:
    window = np.ones(window_size) / window_size
    self.hilbert_envelope = np.convolve(self.hilbert_envelope, window, mode='same')
```

### 数学公式推导：

#### 1. 平滑窗口大小

**窗口大小计算：**
$$W = 0.01 \times f_s = 0.01 \times 360 = 3.6 \approx 4 \text{ 样本}$$

#### 2. 移动平均平滑

**移动平均公式：**
$$y_{smooth}[n] = \frac{1}{W} \sum_{i=0}^{W-1} A[n-i]$$

其中 $A[n]$ 是原始包络信号。

#### 3. 卷积实现

**卷积操作：**
$$y_{smooth}[n] = (A * h)[n]$$

**矩形窗口响应：**
$$h[n] = \begin{cases}
\frac{1}{W} & 0 \leq n < W \\
0 & \text{其他}
\end{cases}$$

#### 4. 频域分析

**平滑滤波器的频率响应：**
$$H_{smooth}(\omega) = \frac{1}{W} \cdot e^{-j\omega(W-1)/2} \cdot \frac{\sin(\omega W/2)}{\sin(\omega/2)}$$

**幅度响应：**
$$|H_{smooth}(\omega)| = \frac{1}{W} \cdot \left|\frac{\sin(\omega W/2)}{\sin(\omega/2)}\right|$$

**截止频率估计：**
$$f_c \approx \frac{f_s}{W} = \frac{360}{4} = 90 \text{ Hz}$$

---

## 自适应阈值检测算法

### 代码位置：[`hilbert_qrs.py:110-213`](hilbert_qrs.py#L110-L213)

### 1. 初始化阶段

#### 代码分析：
```python
# 初始化阶段 - 使用前2秒信号建立初始阈值
init_samples = int(2 * self.fs)
init_envelope = self.hilbert_envelope[:init_samples]
init_threshold = np.mean(init_envelope) + 2.0 * np.std(init_envelope)

# 噪声和信号阈值初始化
signal_peak = init_threshold
noise_peak = np.mean(init_envelope)
threshold = init_threshold
```

#### 数学公式：

**初始包络统计：**
$$\mu_{init} = \frac{1}{N_{init}} \sum_{i=0}^{N_{init}-1} A[i]$$

$$\sigma_{init} = \sqrt{\frac{1}{N_{init}-1} \sum_{i=0}^{N_{init}-1} (A[i] - \mu_{init})^2}$$

**初始阈值：**
$$T_{init} = \mu_{init} + 2\sigma_{init}$$

**信号和噪声峰值跟踪：**
$$SP[0] = T_{init}$$
$$NP[0] = \mu_{init}$$
$$T[0] = T_{init}$$

### 2. 不应期设置

#### 代码分析：
```python
# 不应期参数
rr_interval_min = int(0.2 * self.fs)   # 200ms (支持300bpm)
rr_interval_max = int(2.0 * self.fs)   # 2000ms (30bpm下限)
```

#### 数学公式：

**最小R-R间期：**
$$RR_{min} = 0.2 \times f_s = 0.2 \times 360 = 72 \text{ 样本}$$

**最大R-R间期：**
$$RR_{max} = 2.0 \times f_s = 2.0 \times 360 = 720 \text{ 样本}$$

### 3. 自适应阈值更新

#### 代码分析：
```python
# 学习阶段使用更高的学习率
if learning_count < learning_beats:
    learning_factor = 0.4
    learning_count += 1
else:
    learning_factor = 0.1  # 稳定后使用较小学习率

signal_peak = learning_factor * current_value + (1 - learning_factor) * signal_peak

# 动态调整阈值
if learning_count < learning_beats:
    threshold_factor = 0.3
else:
    threshold_factor = 0.25

threshold = noise_peak + threshold_factor * (signal_peak - noise_peak)
```

#### 数学公式：

**信号峰值更新（指数移动平均）：**
$$SP[n] = \alpha \cdot A[n] + (1-\alpha) \cdot SP[n-1]$$

其中学习因子：
- 学习阶段 $\alpha = 0.4$
- 稳定状态 $\alpha = 0.1$

**噪声峰值更新：**
$$NP[n] = 0.25 \cdot A[n] + 0.75 \cdot NP[n-1]$$

**阈值计算：**
$$T[n] = NP[n] + \beta \cdot (SP[n] - NP[n])$$

其中阈值因子：
- 学习阶段 $\beta = 0.3$
- 稳定状态 $\beta = 0.25$

### 4. 动态衰减机制

#### 代码分析：
```python
# 在长时间没有检测到峰值时，逐渐降低阈值
if len(peaks) > 0 and (i - peaks[-1]) > int(1.0 * self.fs):  # 超过1秒无峰值
    threshold *= 0.995  # 每个样本降低阈值0.5%
```

#### 数学公式：

**指数衰减：**
$$T[n] = 0.995 \cdot T[n-1]$$

**衰减时间常数：**
$$T(n) = T_0 \cdot (0.995)^n$$

**半衰期计算：**
$$0.995^{n_{1/2}} = 0.5$$
$$n_{1/2} = \frac{\ln(0.5)}{\ln(0.995)} \approx 138 \text{ 样本} \approx 0.38 \text{ 秒}$$

---

## 回溯搜索检测

### 代码位置：[`hilbert_qrs.py:215-255`](hilbert_qrs.py#L215-L255)

#### 代码分析：
```python
def _searchback_detection(self, start_idx, end_idx, threshold):
    search_start = start_idx + int(0.15 * self.fs)
    search_end = min(end_idx, start_idx + int(1.5 * self.fs))

    # 寻找局部最大值
    peaks = []
    min_peak_distance = int(0.25 * self.fs)
    local_threshold = threshold * 0.7

    for i in range(2, len(search_segment) - 2):
        if (search_segment[i] > local_threshold and
            search_segment[i] > search_segment[i-1] and
            search_segment[i] > search_segment[i+1] and
            search_segment[i] > search_segment[i-2] and
            search_segment[i] > search_segment[i+2]):
```

#### 数学公式：

#### 1. 搜索窗口定义

**搜索开始：**
$$s_{start} = p_{last} + 0.15 \times f_s$$

**搜索结束：**
$$s_{end} = \min(p_{current}, p_{last} + 1.5 \times f_s)$$

#### 2. 局部最大值检测

**局部阈值：**
$$T_{local} = 0.7 \times T$$

**局部最大值条件：**
$$A[i] > T_{local} \land A[i] > A[i-1] \land A[i] > A[i+1] \land A[i] > A[i-2] \land A[i] > A[i+2]$$

#### 3. 峰值显著性检验

**局部均值对比：**
$$A[i] > 1.3 \cdot \bar{A}_{local}$$

其中：
$$\bar{A}_{local} = \frac{1}{2W+1} \sum_{j=i-W}^{i+W} A[j]$$

#### 4. 峰值距离约束

**最小峰值距离：**
$$d_{min} = 0.25 \times f_s = 90 \text{ 样本}$$

---

## 峰值精确定位

### 代码位置：[`hilbert_qrs.py:257-281`](hilbert_qrs.py#L257-L281)

#### 代码分析：
```python
def _refine_peak_locations(self, peak_indices):
    for peak_idx in peak_indices:
        # 在滤波信号上搜索R波峰值
        search_window = int(0.04 * self.fs)  # ±40ms搜索窗口
        search_start = max(0, peak_idx - search_window)
        search_end = min(len(self.filtered_signal), peak_idx + search_window)

        if search_start < search_end:
            search_segment = self.filtered_signal[search_start:search_end]
            if len(search_segment) > 0:
                # 寻找绝对值最大值（R波可能是正或负）
                local_max_idx = np.argmax(np.abs(search_segment)) + search_start
```

#### 数学公式：

#### 1. 精确搜索窗口

**窗口大小：**
$$W = 0.04 \times f_s = 0.04 \times 360 = 14.4 \approx 14 \text{ 样本}$$

**搜索边界：**
$$s_{start} = \max(0, p_{envelope} - W)$$
$$s_{end} = \min(N, p_{envelope} + W)$$

#### 2. 绝对值最大值搜索

**精确定位：**
$$\hat{p} = \arg\max_{i \in [s_{start}, s_{end}]} |x_{filtered}[i]|$$

**数学表示：**
$$\hat{p} = \{i | |x_{filtered}[i]| = \max_{j \in [s_{start}, s_{end}]} |x_{filtered}[j]|\}$$

#### 3. 包络峰值到R波峰值的映射

**映射关系：**
$$\hat{p}_{R} = \text{argmax}_{i \in W(p_{envelope})} |x_{filtered}[i]|$$

其中 $W(p_{envelope})$ 是包络峰值 $p_{envelope}$ 周围的搜索窗口。

---

## 心率计算

### 代码位置：[`hilbert_qrs.py:283-301`](hilbert_qrs.py#L283-L301)

#### 代码分析：
```python
def calculate_heart_rate(self):
    if len(self.qrs_peaks) < 2:
        return 0, []

    # 计算R-R间期 (转换为ms)
    rr_intervals = np.diff(self.qrs_peaks) * 1000 / self.fs

    # 计算平均心率
    avg_rr_interval = np.mean(rr_intervals)
    heart_rate_bpm = 60000 / avg_rr_interval

    return heart_rate_bpm, rr_intervals
```

#### 数学公式：

#### 1. R-R间期计算

**样本中的R-R间期：**
$$RR[i] = P[i+1] - P[i] \quad \text{for } i = 0, 1, ..., M-1$$

其中 $P[i]$ 是检测到的R波峰值位置。

**毫秒中的R-R间期：**
$$RR_{ms}[i] = \frac{RR[i] \times 1000}{f_s}$$

#### 2. 心率计算

**平均R-R间期：**
$$\overline{RR} = \frac{1}{M} \sum_{i=0}^{M-1} RR_{ms}[i]$$

**每分钟心率：**
$$HR_{bpm} = \frac{60000}{\overline{RR}}$$

---

## 算法对比分析

### 希尔伯特变换 vs Pan-Tomkins算法

#### 1. 处理复杂度

**希尔伯特变换算法：**
- 时间复杂度：$O(N \log N)$ (主要来自FFT)
- 空间复杂度：$O(N)$
- 主要操作：FFT → 复数乘法 → IFFT → 包络计算 → 阈值检测

**Pan-Tomkins算法：**
- 时间复杂度：$O(N)$
- 空间复杂度：$O(N)$
- 主要操作：滤波 → 微分 → 平方 → 积分 → 阈值检测

#### 2. 频域特性

**希尔伯特变换：**
- 对基线漂移不敏感
- 包络提取突出信号幅度特征
- 适用于低频噪声环境

**Pan-Tomkins：**
- 微分操作对高频噪声敏感
- 积分操作平滑噪声
- 对QRS波形态变化适应性好

#### 3. 检测性能

**希尔伯特变换优势：**
$$\text{SNR}_{improvement} \propto \frac{\sqrt{\sum_{i=N_{QRS}} A^2[i]}}{\sqrt{\sum_{i=N_{noise}} A^2[i]}}$$

- 包络分析提高信噪比
- 对R波极性不敏感（通过绝对值）
- 计算简单，易于实时实现

---

## 算法流程总结

### 完整信号处理链

#### 1. 输入信号
令 $x[n]$ 为采样率为 $f_s = 360$ Hz 的输入ECG信号。

#### 2. 带通滤波
$$x_{filtered}[n] = \text{BPF}_{5-40Hz}\{x[n]\}$$

#### 3. 信号加权混合
$$x_{mixed}[n] = 0.2 \cdot x[n] + 0.8 \cdot x_{filtered}[n]$$

#### 4. 希尔伯特变换
$$z[n] = x_{mixed}[n] + j\mathcal{H}\{x_{mixed}[n]\}$$

#### 5. 包络提取
$$A[n] = |z[n]| = \sqrt{x_{mixed}^2[n] + \mathcal{H}^2\{x_{mixed}[n]\}}$$

#### 6. 包络平滑
$$A_{smooth}[n] = \frac{1}{W} \sum_{i=0}^{W-1} A[n-i]$$
其中 $W = 4$ (10ms窗口)。

#### 7. 自适应阈值检测
初始化：$T_0 = \mu_{init} + 2\sigma_{init}$

对每个样本 $n$ 更新：
- 如果 $A_{smooth}[n] > T[n]$ 且在不应期外：检测到QRS
- 更新信号峰值：$SP[n] = \alpha A_{smooth}[n] + (1-\alpha)SP[n-1]$
- 更新噪声峰值：$NP[n] = 0.25A_{smooth}[n] + 0.75NP[n-1]$
- 更新阈值：$T[n] = NP[n] + \beta(SP[n] - NP[n])$
- 动态衰减：$T[n] = 0.995 \cdot T[n-1]$（长时间无峰值时）

#### 8. 峰值精确定位
对每个包络峰值 $p_{envelope}$：
$$\hat{p} = \arg\max_{i \in [p_{envelope}-14, p_{envelope}+14]} |x_{filtered}[i]|$$

#### 9. 心率分析
$$RR_{ms}[i] = \frac{1000(P[i+1] - P[i])}{f_s}$$
$$HR_{bpm} = \frac{60000}{\overline{RR}}$$

### 算法参数总结

| 参数 | 值 | 用途 |
|------|----|----- |
| 采样率 ($f_s$) | 360 Hz | ECG采样频率 |
| 滤波频带 | 5-40 Hz | QRS频率范围 |
| 信号权重 | 0.2/0.8 | 原始/滤波信号混合比例 |
| 包络平滑窗口 | 10ms (4样本) | 减少包络噪声 |
| 学习因子 (α) | 0.4 (学习), 0.1 (稳定) | 信号峰值适应率 |
| 阈值因子 (β) | 0.3 (学习), 0.25 (稳定) | 阈值敏感性 |
| 不应期 | 200ms (72样本) | 最小R-R间期 |
| 动态衰减 | 0.995 | 长时间无峰值时阈值衰减 |
| 精确定位窗口 | 40ms (14样本) | R波精确搜索 |

### 算法优势总结

1. **基线漂移鲁棒性**：
   $$\mathcal{H}\{\text{baseline}[n]\} \approx 0$$
   包络提取对直流分量不敏感

2. **极性不敏感性**：
   $$A[n] = |z[n]|$$
   通过绝对值处理正负R波

3. **计算效率**：
   - FFT复杂度：$O(N\log N)$
   - 适合实时处理

4. **自适应能力**：
   - 动态阈值调整
   - 学习机制适应信号变化

---

## 参考文献

1. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/1.0.0/

2. Addison, P. S. (2005). Wavelet transforms and the ECG: a review. Physiological Measurement, 26(5), R155-R199.

3. Xu, X., & Liu, Y. (2019). ECG R-wave detection using Hilbert transform. Biomedical Signal Processing and Control, 48, 191-204.

4. Sörnmo, L., & Laguna, P. (2005). Bioelectrical signal processing in cardiac and neurologic applications. Elsevier Academic Press.