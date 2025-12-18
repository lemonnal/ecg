MIT-BIH 数据库中所有symbol的意义，请全部给出

以下是MIT-BIH心律失常数据库中所有注释符号（symbol）的完整意义：

## 标准注释符号

### 正常和异常心拍

- **N** - 正常心拍 (Normal beat)
- **L** - 左束支传导阻滞 (Left bundle branch block)
- **R** - 右束支传导阻滞 (Right bundle branch block)
- **A** - 房性早搏 (Atrial premature beat)
- **a** - 异常房性早搏 (Aberrated atrial premature beat)
- **J** - 交界性早搏 (Nodal (junctional) premature beat)
- **S** - 室上性早搏 (Supraventricular premature beat)
- **V** - 室性早搏 (Premature ventricular contraction)
- **E** - 室性逸搏 (Ventricular escape beat)
- **j** - 交界性逸搏 (Nodal (junctional) escape beat)
- **n** - 室上性逸搏 (Supraventricular escape beat)
- **/** - 起搏心拍 (Paced beat)
- **f** - 融合心拍 (Fusion of ventricular and normal beat)
- **F** - 融合心拍 (Fusion of paced and normal beat)
- **[** - 心动过速起始 (Start of ventricular flutter/fibrillation)
- **]** - 心动过速终止 (End of ventricular flutter/fibrillation)
- **!** - 室性扑动 (Ventricular flutter wave)
- **x** - 未分类心拍 (Non-conducted P-wave)
- **Q** - 无法分类的心拍 (Unclassifiable beat)

### 节律变化

- **|** - 节律变化 (Isolated QRS-like artifact)
- **~** - 信号质量变化 (Change in signal quality)
- **"** - 注释评论 (Comment annotation)
- **+** - 节律变化 (Rhythm change)

### 起搏器相关

- **p** - 起搏器脉冲 (Pacing spike)
- **u** - 起搏器未夺获 (Pacing artifact - non-captured)

### 其他符号

- **s** - 室性暂停 (ST change)
- **T** - 心室性心动过速 (Ventricular tachycardia)
- **** - 波形起始 (Waveform onset)
- **=** - 波形终止 (Waveform offset)

## 常见统计分布

在MIT-BIH数据库中，各符号的典型出现频率：

- **N** (正常心拍): 约占60-75%
- **V** (室性早搏): 约占5-20%
- **A** (房性早搏): 约占1-10%
- **L/R** (束支传导阻滞): 约占1-5%
- **其他符号**: 通常少于1%

## 临床意义

### 高危心律失常

- **V, E, F** - 室性异常，需要重点关注
- **[, ], !** - 室性快速心律失常，危险信号

### 相对安全

- **N, A, J, S** - 相对良性心律失常
- **L, R** - 传导阻滞，需结合临床

### 起搏器相关

- **/, p, u** - 起搏器功能相关
- **F** - 融合心拍，起搏器与自身心搏融合

## 1. 📊 采样频率 (Sampling Frequency)
fs = 360 Hz  # MIT-BIH ECG数据的标准采样频率
为什么是360Hz？
奈奎斯特定理: 要准确采集频率为f的信号，采样频率必须 ≥ 2f
ECG频率范围: 正常ECG信号主要成分在0.05-100Hz之间
临床标准: MIT-BIH数据集使用360Hz采样，满足ECG分析需求
频率分辨率: 360Hz意味着每秒采样360个点，时间分辨率为2.78ms
## 2. 🎛️ 通带范围 (Passband Range)
lowcut = 0.5 Hz   # 低频截止频率
highcut = 40.0 Hz # 高频截止频率
低频截止 (0.5 Hz):
基线漂移: 呼吸引起的基线漂移通常在0.1-0.5Hz
电极移动: 皮肤接触不良产生的低频噪声
保留心跳: 心率一般在1-3Hz (60-180bpm)，0.5Hz以上可保留
高频截止 (40 Hz):
ECG信号: P波、QRS波群、T波的主要频率成分都在40Hz以下
高频噪声: 肌电干扰通常在40-100Hz或更高
工频干扰: 50/60Hz电磁干扰被有效滤除
## 3. 🔢 滤波器阶数 (Filter Order)
order = 4  # 4阶Butterworth滤波器
阶数影响:
低阶滤波器 (1-2阶): 过渡平缓，衰减较慢，但相位失真小
中阶滤波器 (3-4阶): 平衡选择，常用在ECG处理中
高阶滤波器 (5+阶): 过渡陡峭，衰减快，但可能引入振铃效应
Butterworth特性:
通带内幅频响应平坦
无振铃效应
相位响应相对线性
## 4. 🔄 滤波方式 (Filtering Method)
filtered_signal = scipy_signal.filtfilt(b, a, signal_data)
零相位滤波 (filtfilt): 工作原理:
前向滤波: 从左到右通过滤波器
反向滤波: 将结果反转，再次从左到右通过滤波器
再次反转回原始时序
优势:
零相位失真: 信号时序保持不变
无延迟: 不会造成信号的时间偏移
高精度: 两次滤波提高幅度精度
对比传统滤波:
# 传统单向滤波 (有相位失真)
signal_lfilter = scipy_signal.lfilter(b, a, signal_data)

# 零相位滤波 (无相位失真)  
signal_filtfilt = scipy_signal.filtfilt(b, a, signal_data)
5. 📈 滤波器设计公式
nyquist = 0.5 * fs              # 奈奎斯特频率 = 180Hz
low = lowcut / nyquist          # 归一化低频截止 = 0.5/180 = 0.0028
high = highcut / nyquist        # 归一化高频截止 = 40/180 = 0.222
b, a = scipy_signal.butter(order, [low, high], btype='band')
归一化频率: 将实际频率转换为与奈奎斯特频率的比值，范围[0,1]
6. 🎯 实际效果
滤波前信号包含:
心电信号 (0.5-40Hz)
基线漂移 (<0.5Hz)
肌电噪声 (>40Hz)
工频干扰 (50/60Hz)
其他环境噪声
滤波后信号:
✅ 保留完整ECG波形特征
✅ 去除基线漂移
✅ 抑制高频噪声
✅ 消除工频干扰
✅ 保持时间对齐，无相位失真
这种参数组合是ECG信号处理的黄金标准，既能有效降噪，又能保留重要的诊断信息。