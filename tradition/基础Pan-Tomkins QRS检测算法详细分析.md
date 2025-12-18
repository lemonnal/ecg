# Pan-Tomkins QRS检测算法 - 详细数学公式分析

## 目录
1. [算法概述](#算法概述)
2. [带通滤波器](#带通滤波器)
3. [微分操作](#微分操作)
4. [平方函数](#平方函数)
5. [移动窗口积分](#移动窗口积分)
6. [阈值检测算法](#阈值检测算法)
7. [回溯搜索检测](#回溯搜索检测)
8. [峰值精确定位](#峰值精确定位)
9. [心率计算](#心率计算)
10. [算法流程总结](#算法流程总结)

## 算法概述

Pan-Tomkins算法是一种实时QRS检测算法，通过以下几个阶段处理ECG信号：

1. **带通滤波** (5-40 Hz)
2. **微分** (突出QRS斜率)
3. **平方** (使所有值为正)
4. **移动窗口积分** (获得波形特征)
5. **自适应阈值检测** (检测QRS复合波)

---

## 带通滤波器

### 代码位置：[`pan_tomkins_qrs.py:37-64`](pan_tomkins_qrs.py#L37-L64)

```python
def bandpass_filter(self, signal_data):
    # 设计带通滤波器 - 针对QRS波群优化频率范围，略微扩展频带
    nyquist = 0.5 * self.fs  # 奈奎斯特频率
    low = 5.0 / nyquist     # 归一化低频截止频率
    high = 40.0 / nyquist   # 归一化高频截止频率

    # 使用3阶Butterworth滤波器
    b, a = scipy_signal.butter(3, [low, high], btype='band')

    # 应用零相位滤波
    filtered_signal = scipy_signal.filtfilt(b, a, signal_data)
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

#### 4. 数字滤波器实现

数字滤波器差分方程：

$$y[n] = \sum_{k=0}^{N} b[k] \cdot x[n-k] - \sum_{k=1}^{M} a[k] \cdot y[n-k]$$

其中：
- $b[k]$ 是分子系数（前馈）
- $a[k]$ 是分母系数（反馈）
- $x[n]$ 是输入信号
- $y[n]$ 是输出信号

#### 5. 零相位滤波

`filtfilt`函数应用滤波器两次（前向和后向）：

$$y_{final}[n] = \text{filtfilt}(b, a, x[n]) = \text{filter}(b, a, \text{filter}(b, a, x[n], \text{reverse}), \text{reverse})$$

这消除了相移并保持信号时序。

---

## 微分操作

### 代码位置：[`pan_tomkins_qrs.py:66-86`](pan_tomkins_qrs.py#L66-L86)

```python
def derivative(self, signal_data):
    differentiated_signal = np.zeros_like(signal_data)

    # 使用5点中心差分公式
    for i in range(2, len(signal_data) - 2):
        differentiated_signal[i] = (
            -signal_data[i+2] + 8*signal_data[i+1] - 8*signal_data[i-1] + signal_data[i-2]
        ) / 12
```

### 数学公式推导：

#### 1. 5点中心差分公式

使用5点中心差分的导数近似：

$$f'(x) \approx \frac{-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)}{12h}$$

其中：
- $h$ 是采样间隔
- $h = \frac{1}{f_s} = \frac{1}{360} \approx 2.78 \text{ ms}$

#### 2. 离散实现

对于离散信号 $x[n]$：

$$y[n] = \frac{-x[n+2] + 8x[n+1] - 8x[n-1] + x[n-2]}{12}$$

#### 3. 误差分析

5点公式的截断误差为 $O(h^4)$：

$$\text{Error} \approx \frac{h^4}{30}f^{(5)}(\xi)$$

其中 $\xi$ 是区间中的某点。

#### 4. 频率响应

频域中的微分操作：

$$Y(e^{j\omega}) = j\omega \cdot X(e^{j\omega})$$

5点近似的频率响应：

$$H_{deriv}(\omega) = \frac{j\omega}{h} \cdot \left(1 - \frac{\omega^4 h^4}{90} + O(\omega^6 h^6)\right)$$

---

## 平方函数

### 代码位置：[`pan_tomkins_qrs.py:88-99`](pan_tomkins_qrs.py#L88-L99)

```python
def squaring(self, signal_data):
    return signal_data ** 2
```

### 数学公式推导：

#### 1. 平方操作

$$y[n] = x^2[n]$$

#### 2. 目的分析

**正值：**
$$x^2[n] \geq 0 \quad \forall n$$

**放大大值：**
对于 $|x_1[n]| > |x_2[n]|$：
$$x_1^2[n] - x_2^2[n] = (x_1[n] - x_2[n])(x_1[n] + x_2[n])$$

#### 3. 频域分析

平方创建和频与差频分量：

如果 $x(t) = A\cos(\omega_0 t)$，则：
$$x^2(t) = \frac{A^2}{2}(1 + \cos(2\omega_0 t))$$

在频域中：
$$\mathcal{F}\{x^2(t)\} = \frac{A^2}{2}[\delta(f) + \frac{1}{2}\delta(f-2f_0) + \frac{1}{2}\delta(f+2f_0)]$$

---

## 移动窗口积分

### 代码位置：[`pan_tomkins_qrs.py:101-122`](pan_tomkins_qrs.py#L101-L122)

```python
def moving_window_integration(self, signal_data, window_size=None):
    if window_size is None:
        # 自适应窗口大小 - 基于QRS波群的典型宽度
        window_size = int(0.080 * self.fs)  # 80ms窗口

    # 使用卷积实现移动平均
    window = np.ones(window_size) / window_size
    integrated_signal = np.convolve(signal_data, window, mode='same')
```

### 数学公式推导：

#### 1. 窗口大小计算

**默认窗口大小：**
$$N = 0.080 \times f_s = 0.080 \times 360 = 28.8 \approx 29 \text{ 样本}$$

#### 2. 移动窗口积分公式

**直接形式：**
$$y[n] = \frac{1}{N} \sum_{i=0}^{N-1} x[n-i]$$

#### 3. 卷积实现

**矩形窗口的脉冲响应：**
$$h[n] = \begin{cases}
\frac{1}{N} & 0 \leq n < N \\
0 & \text{其他}
\end{cases}$$

**卷积操作：**
$$y[n] = (x * h)[n] = \sum_{k=-\infty}^{\infty} x[k] \cdot h[n-k]$$

#### 4. 递归实现（高效）

**递归形式：**
$$y[n] = y[n-1] + \frac{1}{N}(x[n] - x[n-N])$$

#### 5. 频率响应

**矩形窗口频率响应：**
$$H(e^{j\omega}) = \frac{1}{N} \cdot e^{-j\omega(N-1)/2} \cdot \frac{\sin(\omega N/2)}{\sin(\omega/2)}$$

**幅度响应：**
$$|H(e^{j\omega})| = \frac{1}{N} \cdot \left|\frac{\sin(\omega N/2)}{\sin(\omega/2)}\right|$$

---

## 阈值检测算法

### 代码位置：[`pan_tomkins_qrs.py:152-255`](pan_tomkins_qrs.py#L152-L255)

### 1. 初始化阶段

#### 代码分析：
```python
# 初始化阶段 - 使用前2秒信号建立初始阈值
init_samples = int(2 * self.fs)  # 2秒 = 720个样本
init_signal = self.integrated_signal[:init_samples]
init_threshold = np.mean(init_signal) + 2.0 * np.std(init_signal)

# 噪声和信号阈值初始化
signal_peak = init_threshold
noise_peak = np.mean(init_signal)
threshold = init_threshold
```

#### 数学公式：

**初始信号统计：**
$$\mu_{init} = \frac{1}{N_{init}} \sum_{i=0}^{N_{init}-1} x[i]$$

$$\sigma_{init} = \sqrt{\frac{1}{N_{init}-1} \sum_{i=0}^{N_{init}-1} (x[i] - \mu_{init})^2}$$

**初始阈值：**
$$T_{init} = \mu_{init} + 2\sigma_{init}$$

**信号和噪声峰值跟踪：**
$$SP[0] = T_{init}$$
$$NP[0] = \mu_{init}$$
$$T[0] = T_{init}$$

### 2. 不应期设置

#### 代码分析：
```python
# 优化不应期参数
rr_interval_min = int(0.2 * self.fs)   # 200ms (支持300bpm)
rr_interval_max = int(2.0 * self.fs)   # 2000ms (30bpm下限)
```

#### 数学公式：

**最小R-R间期：**
$$RR_{min} = 0.2 \times f_s = 0.2 \times 360 = 72 \text{ 样本}$$

**最大R-R间期：**
$$RR_{max} = 2.0 \times f_s = 2.0 \times 360 = 720 \text{ 样本}$$

**心率限制：**
$$HR_{max} = \frac{60}{RR_{min}/f_s} = 300 \text{ bpm}$$
$$HR_{min} = \frac{60}{RR_{max}/f_s} = 30 \text{ bpm}$$

### 3. 自适应阈值更新

#### 代码分析：
```python
# 动态调整阈值更新策略
if learning_count < learning_beats:
    learning_factor = 0.5  # 学习阶段使用更高的学习率
else:
    learning_factor = 0.125  # 稳定后使用较小学习率

signal_peak = learning_factor * current_value + (1 - learning_factor) * signal_peak

# 阈值计算
threshold = noise_peak + threshold_factor * (signal_peak - noise_peak)
```

#### 数学公式：

**信号峰值更新（指数移动平均）：**
$$SP[n] = \alpha \cdot x[n] + (1-\alpha) \cdot SP[n-1]$$

其中 $\alpha$ 是学习因子：
- 学习阶段 $\alpha = 0.5$
- 稳定状态 $\alpha = 0.125$

**噪声峰值更新：**
$$NP[n] = 0.25 \cdot x[n] + 0.75 \cdot NP[n-1]$$

**阈值计算：**
$$T[n] = NP[n] + \beta \cdot (SP[n] - NP[n])$$

其中 $\beta$ 是阈值因子：
- 学习阶段 $\beta = 0.35$
- 稳定状态 $\beta = 0.25$

### 4. QRS检测逻辑

#### 代码分析：
```python
for i in range(len(self.integrated_signal)):
    current_value = self.integrated_signal[i]

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
```

#### 数学公式：

**峰值检测条件：**
$$x[i] > T[n] \land (i - p_{last} > RR_{min} \lor \text{无先前峰值})$$

**漏检检测：**
$$i - p_{last} > RR_{max} \implies \text{触发回溯搜索}$$

---

## 回溯搜索检测

### 代码位置：[`pan_tomkins_qrs.py:257-298`](pan_tomkins_qrs.py#L257-L298)

#### 代码分析：
```python
def _searchback_detection(self, start_idx, end_idx, threshold):
    search_start = start_idx + int(0.15 * self.fs)  # 缩短搜索起始延迟
    search_end = min(end_idx, start_idx + int(1.5 * self.fs))  # 适当扩大搜索范围

    # 寻找局部最大值
    peaks = []
    min_peak_distance = int(0.25 * self.fs)  # 缩短最小峰值间距
    local_threshold = threshold * 0.7  # 降低回溯搜索阈值

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
$$x[i] > T_{local} \land x[i] > x[i-1] \land x[i] > x[i+1] \land x[i] > x[i-2] \land x[i] > x[i+2]$$

#### 3. 峰值距离约束

**最小峰值距离：**
$$d_{min} = 0.25 \times f_s = 0.25 \times 360 = 90 \text{ 样本}$$

**峰值选择条件：**
$$|p_{new} - p_{last}| > d_{min}$$

---

## 峰值精确定位

### 代码位置：[`pan_tomkins_qrs.py:300-324`](pan_tomkins_qrs.py#L300-L324)

#### 代码分析：
```python
def _refine_peak_locations(self, peak_indices):
    refined_peaks = []
    for peak_idx in peak_indices:
        # 在原始信号上搜索R波峰值
        search_window = int(0.05 * self.fs)  # ±50ms搜索窗口
        search_start = max(0, peak_idx - search_window)
        search_end = min(len(self.filtered_signal), peak_idx + search_window)

        if search_start < search_end:
            search_segment = self.filtered_signal[search_start:search_end]
            if len(search_segment) > 0:
                # 寻找局部最大值
                local_max_idx = np.argmax(search_segment) + search_start
                refined_peaks.append(local_max_idx)
```

#### 数学公式：

#### 1. 精确搜索窗口

**窗口大小：**
$$W = 0.05 \times f_s = 0.05 \times 360 = 18 \text{ 样本}$$

**搜索边界：**
$$s_{start} = \max(0, p_{rough} - W)$$
$$s_{end} = \min(N, p_{rough} + W)$$

#### 2. 精确峰值位置

**精确定位：**
$$\hat{p} = \arg\max_{i \in [s_{start}, s_{end}]} x_{filtered}[i]$$

**数学表示：**
$$\hat{p} = \{i | x_{filtered}[i] = \max_{j \in [s_{start}, s_{end}]} x_{filtered}[j]\}$$

---

## 心率计算

### 代码位置：[`pan_tomkins_qrs.py:385-403`](pan_tomkins_qrs.py#L385-L403)

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

#### 3. 统计度量

**R-R间期统计：**

**均值：**
$$\mu_{RR} = \frac{1}{M} \sum_{i=0}^{M-1} RR_{ms}[i]$$

**标准差：**
$$\sigma_{RR} = \sqrt{\frac{1}{M-1} \sum_{i=0}^{M-1} (RR_{ms}[i] - \mu_{RR})^2}$$

**范围：**
$$\text{Range} = [\min(RR_{ms}), \max(RR_{ms})]$$

---

## 算法流程总结

### 完整信号处理链

#### 1. 输入信号
令 $x[n]$ 为采样率为 $f_s = 360$ Hz 的输入ECG信号。

#### 2. 带通滤波
$$x_{filtered}[n] = \text{BPF}_{5-40Hz}\{x[n]\}$$

#### 3. 微分
$$x_{deriv}[n] = \frac{-x_{filtered}[n+2] + 8x_{filtered}[n+1] - 8x_{filtered}[n-1] + x_{filtered}[n-2]}{12}$$

#### 4. 平方
$$x_{squared}[n] = x_{deriv}^2[n]$$

#### 5. 移动窗口积分
$$x_{int}[n] = \frac{1}{N} \sum_{i=0}^{N-1} x_{squared}[n-i]$$
其中 $N = 29$ (80ms窗口)。

#### 6. 自适应阈值检测
初始化：$T_0 = \mu_{init} + 2\sigma_{init}$

对每个样本 $n$ 更新：
- 如果 $x_{int}[n] > T[n]$ 且在不应期外：检测到QRS
- 更新信号峰值：$SP[n] = \alpha x_{int}[n] + (1-\alpha)SP[n-1]$
- 更新噪声峰值：$NP[n] = 0.25x_{int}[n] + 0.75NP[n-1]$
- 更新阈值：$T[n] = NP[n] + \beta(SP[n] - NP[n])$

#### 7. 峰值精确定位
对每个检测到的峰值 $p_{rough}$：
$$\hat{p} = \arg\max_{i \in [p_{rough}-18, p_{rough}+18]} x_{filtered}[i]$$

#### 8. 心率分析
$$RR_{ms}[i] = \frac{1000(P[i+1] - P[i])}{f_s}$$
$$HR_{bpm} = \frac{60000}{\overline{RR}}$$

### 计算复杂度

- **带通滤波**：$O(N \cdot M)$，其中 $M$ 是滤波器阶数（3）
- **微分**：$O(N)$
- **平方**：$O(N)$
- **积分**：$O(N \cdot W)$，其中 $W$ 是窗口大小（29）
- **阈值检测**：$O(N)$
- **回溯搜索**：最坏情况 $O(N \cdot W)$
- **峰值精确定位**：$O(P \cdot W)$，其中 $P$ 是峰值数量

### 算法参数总结

| 参数 | 值 | 用途 |
|------|----|----- |
| 采样率 ($f_s$) | 360 Hz | ECG采样频率 |
| 滤波频带 | 5-40 Hz | QRS频率范围 |
| 滤波器阶数 | 3 | Butterworth滤波器阶数 |
| 积分窗口 | 80ms (29样本) | QRS复合波持续时间 |
| 学习因子 (α) | 0.5 (学习), 0.125 (稳定) | 信号峰值适应率 |
| 阈值因子 (β) | 0.35 (学习), 0.25 (稳定) | 阈值敏感性 |
| 不应期 | 200ms (72样本) | 最小R-R间期 |
| 回溯窗口 | 150ms-1500ms | 漏检恢复 |
| 精确定位窗口 | 100ms (36样本) | 精确R波位置 |

---

## 参考文献

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.

2. Kohler, B. U., Hennig, C., & Orglmeister, R. (2002). The principles of software QRS detection. IEEE Engineering in Medicine and Biology Magazine, 21(1), 42-57.

3. PhysioNet. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/1.0.0/