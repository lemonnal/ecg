# 综合ECG特征点检测算法 - P、Q、S、T波检测详细分析

## 目录
1. [算法概述](#算法概述)
2. [Q点和S点检测算法](#q点和s点检测算法)
3. [P波特征检测算法](#p波特征检测算法)
4. [T波特征检测算法](#t波特征检测算法)
5. [波形起始和结束点检测](#波形起始和结束点检测)
6. [搜索窗口和时间约束](#搜索窗口和时间约束)
7. [算法优化策略](#算法优化策略)
8. [检测准确性评估](#检测准确性评估)
9. [算法参数总结](#算法参数总结)
10. [应用场景和限制](#应用场景和限制)

## 算法概述

综合ECG特征点检测器基于已验证的Pan-TomkinsR波检测算法，在其基础上扩展了P、Q、S、T波的特征检测功能。该算法采用分层检测策略：

1. **R波检测**：使用成熟的Pan-Tomkins算法
2. **QRS复合波检测**：基于R波位置检测Q点和S点
3. **P波检测**：在RR间期前半段检测P波特征
4. **T波检测**：在S波后检测T波特征
5. **精确定位**：通过梯度分析确定波形边界

核心思想：**以R波为基准点，利用时间约束和生理学知识指导其他波形检测**

---

## Q点和S点检测算法

### 代码位置：[`comprehensive_ecg_detector.py:294-318`](comprehensive_ecg_detector.py#L294-L318)

```python
def detect_qrs_points(self):
    """
    检测Q点和S点
    """
    self.q_points = []
    self.s_points = []

    for r_peak in self.r_peaks:
        # 检测Q点 (R波前)
        q_search_start = max(0, r_peak - self.qrs_window // 2)
        q_search_end = r_peak

        if q_search_start < q_search_end:
            q_segment = self.filtered_signal[q_search_start:q_search_end]
            if len(q_segment) > 0:
                # Q点通常是R波前的最小值
                q_local_idx = np.argmin(q_segment) + q_search_start
                self.q_points.append(q_local_idx)

        # 检测S点 (R波后)
        s_search_start = r_peak
        s_search_end = min(len(self.filtered_signal), r_peak + self.qrs_window // 2)

        if s_search_start < s_search_end:
            s_segment = self.filtered_signal[s_search_start:s_search_end]
            if len(s_segment) > 0:
                # S点通常是R波后的最小值
                s_local_idx = np.argmin(s_segment) + s_search_start
                self.s_points.append(s_local_idx)
```

### 数学公式推导：

#### 1. Q点检测窗口

**窗口定义：**
$$Q_{window} = [R - W_{QRS}/2, R]$$

其中：
- $R$ 是R波峰值位置
- $W_{QRS}$ 是QRS窗口大小（100ms）

**窗口大小计算：**
$$W_{QRS} = 0.1 \times f_s = 0.1 \times 360 = 36 \text{ 样本}$$

**Q点搜索范围：**
$$Q_{search} = [R - 18, R]$$

#### 2. Q点定位算法

**最小值搜索：**
$$Q = \arg\min_{i \in Q_{window}} x_{filtered}[i]$$

**数学表示：**
$$Q = \{i | x_{filtered}[i] = \min_{j \in Q_{window}} x_{filtered}[j]\}$$

#### 3. S点检测窗口

**窗口定义：**
$$S_{window} = [R, R + W_{QRS}/2]$$

**S点搜索范围：**
$$S_{search} = [R, R + 18]$$

#### 4. S点定位算法

**最小值搜索：**
$$S = \arg\min_{i \in S_{window}} x_{filtered}[i]$$

#### 5. 检测准确性分析

**QRS复合波的典型时间特征：**
- Q点 → R点：约30-40ms
- R点 → S点：约60-80ms
- QRS总时长：约80-120ms

**搜索窗口合理性：**
- Q点搜索窗口：50ms（足够覆盖正常Q点位置）
- S点搜索窗口：50ms（足够覆盖正常S点位置）

---

## P波特征检测算法

### 代码位置：[`comprehensive_ecg_detector.py:320-358`](comprehensive_ecg_detector.py#L320-L358)

```python
def detect_p_waves(self):
    """
    检测P波特征点 (P_onset, P_peak, P_end)
    """
    self.p_peaks = []
    self.p_onsets = []
    self.p_ends = []

    # 使用低通滤波突出P波
    p_filtered = self.lowpass_filter(self.signal, cutoff=10)

    for i, r_peak in enumerate(self.r_peaks):
        # P波在R波前的搜索窗口
        if i == 0:
            # 第一个心跳，从信号开始搜索
            p_search_start = 0
        else:
            # 正常情况，从前一个R波后开始搜索
            p_search_start = self.r_peaks[i-1] + int(0.2 * self.fs)

        p_search_end = self.q_points[i] if i < len(self.q_points) else r_peak - int(0.05 * self.fs)

        if p_search_start < p_search_end:
            p_segment = p_filtered[p_search_start:p_search_end]
            if len(p_segment) > 0:
                # P波峰值 - 通常是最大值
                p_peak_local_idx = np.argmax(p_segment) + p_search_start
                self.p_peaks.append(p_peak_local_idx)

                # P波起始点 - 查找上升沿起点
                p_onset_start = p_search_start
                p_onset_end = p_peak_local_idx

                if p_onset_start < p_onset_end:
                    p_onset_segment = p_filtered[p_onset_start:p_onset_end]
                    # 寻找斜率变化最大点
                    p_onset_local = self._find_onset_point(p_onset_segment)
                    if p_onset_local is not None:
                        self.p_onsets.append(p_onset_start + p_onset_local)
                    else:
                        self.p_onsets.append(p_search_start)

                # P波结束点 - 查找到基线
                p_end_start = p_peak_local_idx
                p_end_end = p_search_end

                if p_end_start < p_end_end:
                    p_end_segment = p_filtered[p_end_start:p_end_end]
                    # 寻找返回基线的点
                    p_end_local = self._find_offset_point(p_end_segment)
                    if p_end_local is not None:
                        self.p_ends.append(p_end_start + p_end_local)
                    else:
                        self.p_ends.append(p_search_end)
```

### 数学公式推导：

#### 1. P波检测时间窗口

**搜索窗口定义：**
$$P_{search} = [T_{start}, T_{end}]$$

其中：
- $T_{start} = \max(R_{i-1} + T_{PR}, 0)$ （第一个心跳从0开始）
- $T_{end} = Q_i - T_{margin}$

**时间参数：**
- $T_{PR} = 0.2 \times f_s = 72 \text{ 样本}$ (PR间期最小值)
- $T_{margin} = 0.05 \times f_s = 18 \text{ 样本}$ (安全间隔)

#### 2. 低通滤波器设计

**滤波器类型：** 3阶Butterworth低通滤波器

**截止频率：** 10 Hz

**归一化截止频率：**
$$\omega_c = \frac{10}{f_s/2} = \frac{10}{180} = 0.0556$$

**传递函数：**
$$H_{LP}(s) = \frac{1}{\sqrt{1 + (s/\omega_c)^{2n}}}$$

#### 3. P波峰值检测

**峰值定位：**
$$P_{peak} = \arg\max_{i \in P_{search}} x_{LPF}[i]$$

**数学表示：**
$$P_{peak} = \{i | x_{LPF}[i] = \max_{j \in P_{search}} x_{LPF}[j]\}$$

#### 4. P波起始点检测

**梯度计算：**
$$g[i] = \frac{d}{dt}x_{LPF}[i] \approx x_{LPF}[i+1] - x_{LPF}[i]$$

**起始点检测算法：**
$$P_{onset} = \min\{i | g[i] > T_{gradient} \land g[i] > g[i-1]\}$$

其中梯度阈值：
$$T_{gradient} = \sigma_g \times 0.5$$
$$\sigma_g = \sqrt{\frac{1}{N-1}\sum_{k=0}^{N-1}(g[k] - \mu_g)^2}$$

#### 5. P波结束点检测

**结束点检测算法：**
$$P_{end} = \min\{i \in [P_{peak}, P_{search\_end}] | |g[i]| < T_{baseline}\}$$

**基线阈值：**
$$T_{baseline} = \sigma_g \times 0.3$$

---

## T波特征检测算法

### 代码位置：[`comprehensive_ecg_detector.py:360-388`](comprehensive_ecg_detector.py#L360-L388)

```python
def detect_t_waves(self):
    """
    检测T波特征点 (T_peak, T_end)
    """
    self.t_peaks = []
    self.t_ends = []

    # 使用更低的截止频率突出T波
    t_filtered = self.lowpass_filter(self.signal, cutoff=8)

    for i, r_peak in enumerate(self.r_peaks):
        # T波在S波后的搜索窗口
        t_search_start = self.s_points[i] + int(0.05 * self.fs) if i < len(self.s_points) else r_peak + int(0.1 * self.fs)

        if i < len(self.r_peaks) - 1:
            # 不是最后一个心跳，到下一个R波前
            t_search_end = self.r_peaks[i+1] - int(0.1 * self.fs)
        else:
            # 最后一个心跳，到信号结束
            t_search_end = len(t_filtered)

        t_search_end = min(t_search_end, t_search_start + self.t_window)

        if t_search_start < t_search_end:
            t_segment = t_filtered[t_search_start:t_search_end]
            if len(t_segment) > 0:
                # T波峰值 - 可能是正波或负波，取绝对值最大
                t_peak_local_idx = np.argmax(np.abs(t_segment)) + t_search_start
                self.t_peaks.append(t_peak_local_idx)

                # T波结束点
                t_end_start = t_peak_local_idx
                t_end_end = t_search_end

                if t_end_start < t_end_end:
                    t_end_segment = t_filtered[t_end_start:t_end_end]
                    # 寻找返回基线的点
                    t_end_local = self._find_offset_point(t_end_segment)
                    if t_end_local is not None:
                        self.t_ends.append(t_end_start + t_end_local)
                    else:
                        self.t_ends.append(t_search_end)
```

### 数学公式推导：

#### 1. T波检测时间窗口

**搜索窗口定义：**
$$T_{search} = [T_{start}, T_{end}]$$

其中：
- $T_{start} = S_i + T_{ST\_margin}$ （或 $R_i + T_{RS\_margin}$）
- $T_{end} = \min(R_{i+1} - T_{next\_margin}, S_i + T_{T\_max})$

**时间参数：**
- $T_{ST\_margin} = 0.05 \times f_s = 18 \text{ 样本}$
- $T_{RS\_margin} = 0.1 \times f_s = 36 \text{ 样本}$
- $T_{next\_margin} = 0.1 \times f_s = 36 \text{ 样本}$
- $T_{T\_max} = 0.4 \times f_s = 144 \text{ 样本}$

#### 2. T波滤波器设计

**滤波器类型：** 3阶Butterworth低通滤波器

**截止频率：** 8 Hz（比P波滤波更低）

**归一化截止频率：**
$$\omega_c = \frac{8}{180} = 0.0444$$

#### 3. T波峰值检测

**考虑极性的峰值检测：**
$$T_{peak} = \arg\max_{i \in T_{search}} |x_{LPF}[i]|$$

**数学表示：**
$$T_{peak} = \{i | |x_{LPF}[i]| = \max_{j \in T_{search}} |x_{LPF}[j]|\}$$

**极性判断：**
- 如果 $x_{LPF}[T_{peak}] > 0$：正向T波
- 如果 $x_{LPF}[T_{peak}] < 0$：负向T波

#### 4. T波结束点检测

**结束点检测算法：**
$$T_{end} = \min\{i \in [T_{peak}, T_{search\_end}] | |g[i]| < T_{baseline\_T}\}$$

**T波基线阈值：**
$$T_{baseline\_T} = \sigma_{g\_T} \times 0.3$$

#### 5. T波形态分析

**T波持续时间：**
$$T_{duration} = T_{end} - T_{peak}$$

**T波幅度：**
$$T_{amplitude} = |x_{LPF}[T_{peak}] - x_{LPF}[T_{end}]|$$

---

## 波形起始和结束点检测

### 代码位置：[`comprehensive_ecg_detector.py:390-424`](comprehensive_ecg_detector.py#L390-L424)

#### 起始点检测算法：

```python
def _find_onset_point(self, segment):
    """
    查找波形的起始点
    """
    if len(segment) < 10:
        return None

    # 计算梯度
    gradient = np.gradient(segment)

    # 寻找梯度开始显著增加的点
    threshold = np.std(gradient) * 0.5

    for i in range(1, len(gradient) - 1):
        if gradient[i] > threshold and gradient[i] > gradient[i-1]:
            return i

    return 0
```

#### 结束点检测算法：

```python
def _find_offset_point(self, segment):
    """
    查找波形的结束点
    """
    if len(segment) < 10:
        return None

    # 计算梯度
    gradient = np.gradient(segment)

    # 寻找梯度接近零的点
    threshold = np.std(gradient) * 0.3

    for i in range(len(gradient) - 1, 0, -1):
        if abs(gradient[i]) < threshold:
            return i

    return len(segment) - 1
```

### 数学公式推导：

#### 1. 梯度计算

**数值梯度（中心差分）：**
$$g[i] = \frac{x[i+1] - x[i-1]}{2}$$

**SciPy梯度实现：**
$$g[i] = \frac{d}{dt}x[i]$$

#### 2. 起始点检测

**梯度统计特性：**
$$\mu_g = \frac{1}{N}\sum_{i=0}^{N-1} g[i]$$

$$\sigma_g = \sqrt{\frac{1}{N-1}\sum_{i=0}^{N-1}(g[i] - \mu_g)^2}$$

**起始点条件：**
$$i_{onset} = \min\{i | g[i] > T_{onset} \land g[i] > g[i-1]\}$$

其中：
$$T_{onset} = \sigma_g \times 0.5$$

#### 3. 结束点检测

**结束点条件：**
$$i_{offset} = \min\{i \in [k, N-1] | |g[i]| < T_{offset}\}$$

其中从末端向前搜索：
$$T_{offset} = \sigma_g \times 0.3$$

#### 4. 检测优化策略

**多条件约束：**
1. **梯度幅值约束**：确保信号变化足够显著
2. **梯度方向约束**：起始点要求正向变化，结束点要求接近零
3. **时间连续性约束**：确保检测结果符合生理学时序

**自适应阈值：**
$$T_{adaptive} = \alpha \cdot \sigma_g + (1-\alpha) \cdot T_{fixed}$$

---

## 搜索窗口和时间约束

### 生理学时间约束

#### 1. 正常ECG时间间隔

**标准时间参数（以360Hz采样率计算）：**

| 波形 | 正常范围 | 采样数 | 搜索窗口 |
|------|----------|--------|----------|
| P波 | 60-100ms | 22-36 | 108样本 |
| PR间期 | 120-200ms | 43-72 | - |
| QRS波 | 80-120ms | 29-43 | 36样本 |
| ST段 | 80-120ms | 29-43 | - |
| T波 | 120-200ms | 43-72 | 144样本 |
| QT间期 | 350-450ms | 126-162 | - |

#### 2. 搜索窗口设计原则

**窗口大小计算：**
$$W_{search} = k \times T_{expected}$$

其中扩展因子 $k$ 取值：
- P波：$k = 1.5$（考虑变异性）
- T波：$k = 1.8$（考虑变异性）

#### 3. 安全间隔设置

**防止波形重叠：**
$$T_{safety} = 0.05 \times f_s = 18 \text{ 样本}$$

**窗口边界调整：**
$$P_{end} = \min(P_{end}, Q_{onset} - T_{safety})$$
$$T_{end} = \min(T_{end}, P_{onset}^{next} - T_{safety})$$

#### 4. 自适应窗口调整

**基于心率变化：**
$$RR_{current} = R_i - R_{i-1}$$

$$\text{Adjustment Factor} = \frac{RR_{current}}{RR_{average}}$$

$$W_{adjusted} = W_{base} \times \text{Adjustment Factor}$$

---

## 算法优化策略

### 1. 多尺度滤波策略

#### 代码实现：
```python
# R波检测：带通滤波 (5-40 Hz)
self.filtered_signal = self.bandpass_filter(signal_data)

# P波检测：低通滤波 (10 Hz)
p_filtered = self.lowpass_filter(self.signal, cutoff=10)

# T波检测：低通滤波 (8 Hz)
t_filtered = self.lowpass_filter(self.signal, cutoff=8)
```

#### 数学原理：

**频率分离：**
- **QRS波**：5-40 Hz（主要能量集中）
- **P波**：0.5-10 Hz（低频为主）
- **T波**：0.5-8 Hz（更低频）

**滤波器响应对比：**
$$H_{QRS}(f) = \begin{cases}
1 & 5 \leq f \leq 40 \\
0 & \text{其他}
\end{cases}$$

$$H_{P}(f) = \begin{cases}
1 & f \leq 10 \\
0 & f > 10
\end{cases}$$

$$H_{T}(f) = \begin{cases}
1 & f \leq 8 \\
0 & f > 8
\end{cases}$$

### 2. 梯度分析优化

#### 多阶梯度计算：

**一阶梯度：**
$$g_1[i] = x[i] - x[i-1]$$

**二阶梯度：**
$$g_2[i] = x[i+1] - 2x[i] + x[i-1]$$

**组合梯度特征：**
$$g_{combined}[i] = \alpha \cdot |g_1[i]| + \beta \cdot |g_2[i]|$$

### 3. 噪声鲁棒性

#### 信号平滑预处理：

**移动平均平滑：**
$$x_{smooth}[i] = \frac{1}{2W+1}\sum_{j=-W}^{W} x[i+j]$$

**中值滤波去噪：**
$$x_{median}[i] = \text{median}\{x[i-W], ..., x[i+W]\}$$

#### 阈值自适应调整：

**基于信号质量的阈值调整：**
$$T_{adaptive} = T_{base} \cdot (1 + \gamma \cdot \text{SNR}^{-1})$$

其中SNR为信噪比估计。

---

## 检测准确性评估

### 1. 定位精度分析

#### 时间分辨率：
$$\Delta t = \frac{1}{f_s} = \frac{1}{360} \approx 2.78 \text{ ms}$$

#### 检测误差评估：
$$\text{Error}_{P} = |P_{detected} - P_{true}|$$
$$\text{Error}_{Q} = |Q_{detected} - Q_{true}|$$
$$\text{Error}_{S} = |S_{detected} - S_{true}|$$
$$\text{Error}_{T} = |T_{detected} - T_{true}|$$

#### 统计精度指标：

**平均绝对误差（MAE）：**
$$\text{MAE} = \frac{1}{N}\sum_{i=1}^{N}|\text{Error}_i|$$

**均方根误差（RMSE）：**
$$\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}\text{Error}_i^2}$$

### 2. 检测成功率

#### 真阳性率（Sensitivity）：
$$\text{Sensitivity} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

#### 假阳性率（False Positive Rate）：
$$\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$$

#### F1分数：
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

### 3. 时序一致性验证

#### RR间期一致性：
$$\text{RR\_consistency} = \frac{\sum_{i} |\text{RR}_i - \overline{\text{RR}}|}{N \cdot \overline{\text{RR}}}$$

#### 心率变异性约束：
$$\text{HRV\_constraint} = \begin{cases}
1 & \text{if } 0.5 \leq \frac{\text{RR}_i}{\text{RR}_{i-1}} \leq 2.0 \\
0 & \text{otherwise}
\end{cases}$$

---

## 算法参数总结

### 核心参数配置

| 参数类别 | 参数名称 | 数值 | 生理学依据 |
|----------|----------|------|------------|
| **采样参数** | 采样频率 | 360 Hz | 标准ECG采样 |
| **滤波参数** | QRS带通 | 5-40 Hz | QRS能量集中 |
| | P波低通 | 10 Hz | P波低频特性 |
| | T波低通 | 8 Hz | T波更低频 |
| **时间窗口** | QRS窗口 | 100ms (36样本) | QRS复合波典型宽度 |
| | P波窗口 | 300ms (108样本) | 最大P波搜索范围 |
| | T波窗口 | 400ms (144样本) | 最大T波搜索范围 |
| **安全间隔** | 波形间隔 | 18ms | 防止重叠 |
| **梯度阈值** | 起始点 | 0.5×标准差 | 显著性要求 |
| | 结束点 | 0.3×标准差 | 基线收敛要求 |
| **时间约束** | PR间期最小 | 72样本 (200ms) | 生理学下限 |
| | QT间期最大 | 162样本 (450ms) | 生理学上限 |

### 自适应参数

| 参数 | 自适应策略 | 调整范围 |
|------|------------|----------|
| 搜索窗口 | 基于RR间期比例 | ±30% |
| 检测阈值 | 基于信号质量 | ±20% |
| 梯度阈值 | 基于噪声水平 | 动态计算 |

---

## 应用场景和限制

### 1. 适用场景

#### 理想条件：
- **信号质量**：SNR > 10 dB
- **心率范围**：40-180 bpm
- **心律**：窦性心律为主
- **导联**：标准肢体导联

#### 临床应用：
- **心律监测**：实时心率变异性分析
- **诊断辅助**：P波、T波形态分析
- **运动医学**：运动心电图分析

### 2. 算法限制

#### 信号质量限制：
- **基线漂移**：严重的基线漂移影响检测精度
- **肌电干扰**：高频噪声影响Q点和S点检测
- **50/60Hz工频干扰**：需要额外滤波处理

#### 心律失常限制：
- **房颤**：P波消失，无法检测
- **室性早搏**：QRS形态改变，影响Q/S点定位
- **传导阻滞**：P-R间期异常，影响时间约束

#### 生理变异性：
- **年龄相关**：老年人P波幅度降低
- **性别差异**：T波形态性别差异
- **个体差异**：心脏位置和解剖变异

### 3. 性能优化建议

#### 实时处理优化：
- **滑动窗口**：避免全局重计算
- **增量更新**：基于前一检测结果更新
- **并行处理**：多导联同时处理

#### 鲁棒性提升：
- **多尺度融合**：结合多个尺度的检测结果
- **置信度评估**：为每个检测结果分配置信度
- **异常检测**：识别异常波形并标记

---

## 参考文献

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. IEEE Transactions on Biomedical Engineering, 32(3), 230-236.

2. Laguna, P., Jané, R., & Caminal, P. (1996). Automatic detection of wave boundaries in multilead ECG signals: Validation with the CSE database. Computers and Biomedical Research, 29(3), 305-316.

3. Martínez, J. P., Almeida, R., Olmos, S., Rocha, A. P., & Laguna, P. (2004). A wavelet-based ECG delineator: evaluation on standard databases. IEEE Transactions on Biomedical Engineering, 51(4), 570-581.

4. PhysioNet. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/1.0.0/

5. ANSI/AAMI EC57:2012. Testing and reporting performance results of cardiac rhythm and ST segment measurement algorithms. Association for the Advancement of Medical Instrumentation.