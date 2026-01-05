# ECG 心电图信号处理与 QRS 波检测工作空间

本工作空间专注于心电图（ECG）信号处理算法的研究与实现，核心是基于改进的 Pan-Tomkins 算法的实时 ECG 波形检测系统（RealtimeECGDetector），同时包含深度学习方法和传统算法的探索性研究。

---

## 目录

- [项目概述](#项目概述)
- [核心模块详解](#核心模块详解)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境配置](#环境配置)
  - [在线实时检测](#在线实时检测)
  - [离线文件检测](#离线文件检测)
  - [蓝牙数据采集](#蓝牙数据采集)
- [算法原理](#算法原理)
  - [Pan-Tomkins 算法详解](#pan-tomkins-算法详解)
  - [关键参数优化](#关键参数优化)
  - [完整 PQRST 波检测](#完整-pqrst-波检测)
  - [其他算法](#其他算法)
- [硬件与协议](#硬件与协议)
  - [支持的设备](#支持的设备)
  - [设备配置指南](#设备配置指南)
- [辅助模块](#辅助模块)
  - [qrs_detector 参考实现](#qrs_detector-参考实现)
  - [tradition 传统算法集合](#tradition-传统算法集合)
  - [ecg_deepl_method 深度学习方法](#ecg_deepl_method-深度学习方法)
  - [Information 技术文档](#information-技术文档)
- [标准与测试](#标准与测试)
  - [YY 9706.247-2021 标准](#yy-9706247-2021-标准)
  - [MIT-BIH 数据库](#mit-bih-数据库)
- [技术栈](#技术栈)
- [常见问题](#常见问题)
- [参考资料](#参考资料)
- [开发计划](#开发计划)

---

## 项目概述

本项目是一个完整的 ECG 信号处理解决方案，主要特点：

### 核心功能

- **实时检测**：支持通过蓝牙低功耗（BLE）设备进行实时心电信号采集与 PQRST 波检测
- **离线分析**：支持 MIT-BIH 等标准数据库的离线分析
- **多算法对比**：包含传统算法（Pan-Tomkins、希尔伯特变换）和深度学习方法
- **可视化展示**：实时显示信号处理的各个阶段，5 个子图同步显示
- **导联自适应**：针对不同 ECG 导联（MLII, V1-V6, I, II, III, aVR, aVL, aVF）优化参数
- **完整波形检测**：支持 P、Q、R、S、T 五种特征波的检测

### 核心技术

- **Pan-Tomkins 算法**：经典的 QRS 波检测算法
- **自适应带通滤波**：根据不同导联特性调整滤波参数（1-50 Hz 可调）
- **滑动窗口阈值检测**：使用指数移动平均（EMA）平滑阈值适应信号变化
- **相位延迟补偿**：补偿滤波和积分引入的相位延迟
- **异步蓝牙通信**：基于 `asyncio` 和 `bleak` 实现高效蓝牙数据采集

### 应用场景

- 动态心电图（Holter）系统开发
- 心律失常检测算法研究
- ECG 信号质量评估
- 实时心率监测设备
- 医疗器械合规性测试

---

## 核心模块详解

本项目根目录包含核心的实时 ECG 波形检测系统，提供完整的 ECG 波形检测功能。

### 系统架构

```
workspace-ecg/ (根目录)
├── online.py           # 在线实时检测主程序
├── offline.py          # 离线文件检测程序
├── BLE_data.py         # 蓝牙数据采集工具
├── Temp_Method.py      # 临时实验代码
└── test.py             # 测试脚本
```

### 模块依赖关系

```
online.py
├── RealTimeECGDetector               # 实时ECG检测器（核心）
│   ├── bandpass_filter()             # 自适应带通滤波
│   ├── derivative()                  # 5点中心差分微分
│   ├── squaring()                    # 平方运算
│   ├── moving_window_integration()   # 移动窗口积分
│   ├── threshold_detection()         # EMA平滑的自适应阈值检测
│   ├── apply_delay_compensation()    # 相位延迟补偿
│   ├── detect_q_waves()              # Q波检测
│   ├── detect_s_waves()              # S波检测
│   ├── detect_p_waves()              # P波检测
│   ├── detect_t_waves()              # T波检测
│   └── detect_wave()                 # 完整PQRST波检测流程
└── BlueToothCollector               # 蓝牙数据采集器
    ├── start_collection()            # 开启数据采集
    ├── handle_rx()                   # 接收数据回调
    ├── build_protocol_packet()       # 构建协议包
    └── packet_decode()               # 解析ECG数据包

offline.py
└── PanTomkinsQRSDetectorOffline      # 离线QRS检测器
    └── (同上信号处理方法，针对离线文件优化)
```

### 核心特性

- **扁平化结构**：核心检测文件直接位于项目根目录，便于使用和维护
- **实时蓝牙采集**：通过 BLE 设备实时接收 ECG 信号
- **自适应滤波**：根据导联类型自动调整带通滤波参数（1-50 Hz）
- **完整 PQRST 检测**：检测 P、Q、R、S、T 五种特征波
  - R 峰：QRS 复合波的主峰
  - Q 波：R 峰前的负向波
  - S 波：R 峰后的负向波
  - P 波：QRS 波前的正向小波（心房去极化）
  - T 波：QRS 波后的正向宽波（心室复极化）
- **五阶段处理可视化**：滤波 → 微分 → 平方 → 积分 → 检测
- **实时可视化**：5 个子图同步显示处理过程和波形标记
- **多导联支持**：MLII, V1-V6, I, II, III, aVR, aVL, aVF
- **滑动窗口检测**：自适应阈值适应信号变化
- **MIT-BIH 支持**：支持标准数据库批量处理
- **导联特定参数优化**：针对不同导联自动调整检测参数
- **相位延迟补偿**：补偿滤波和积分引入的延迟
- **EMA 阈值平滑**：使用指数移动平均避免阈值突变

### 信号处理流程

```
ECG 信号输入
    ↓
1. 带通滤波 (1-50 Hz 可调)
    - 去除基线漂移
    - 滤除高频噪声
    - 原始信号与滤波信号加权组合
    ↓
2. 微分处理 (5点中心差分)
    - 突出 QRS 波斜率
    - 抑制低频成分
    ↓
3. 平方运算
    - 使所有值为正
    - 放大高斜率点
    ↓
4. 移动窗口积分 (100ms)
    - 平滑信号
    - 提取 QRS 波特征
    ↓
5. 自适应阈值检测
    - 滑动窗口 (约3秒)
    - EMA 动态阈值计算
    - 不应期保护 (200-500ms)
    ↓
6. 相位延迟补偿
    - 补偿滤波和积分引入的延迟
    - 在补偿位置附近搜索真实峰值
    ↓
7. PQRST 波检测
    - Q 波检测（R峰前10-80ms）
    - S 波检测（R峰后10-100ms）
    - P 波检测（R峰前40-200ms）
    - T 波检测（R峰后150-400ms）
    ↓
输出 PQRST 波位置
```

### 实时检测功能

- 支持蓝牙低功耗（BLE）设备实时数据采集
- 实时 PQRST 波检测与标记
- 五阶段信号处理可视化
- 自动适应信号幅度变化
- 可配置的导联参数（支持 MLII, V1-V6, I, II, III, aVR, aVL, aVF）

### 离线分析功能

- 支持 MIT-BIH Arrhythmia Database 格式
- 批量处理多个记录文件
- 导联特定参数优化
- 检测结果统计与评估

### 实时显示界面

程序会创建 5 个子图窗口，实时显示：

1. **原始信号**（original signal）：接收到的原始 ECG 信号
2. **滤波信号**（filtered signal）：经过带通滤波后的信号
3. **微分信号**（differentiated signal）：微分处理后的信号
4. **平方信号**（squared signal）：平方运算后的信号
5. **积分信号**（integrated signal）：移动窗口积分后的信号

检测到的波形标记：
- **红色圆圈 (o)**：R 峰
- **蓝色向上三角形 (^)**：Q 波
- **绿色向下三角形 (v)**：S 波
- **品红色方块 (s)**：P 波
- **青色菱形 (D)**：T 波

### 采样频率

- **在线模式**：250 Hz
- **离线模式**：360 Hz（MIT-BIH 标准）

---

## 项目结构

```
workspace-ecg/
│
├── online.py                    # ★ 在线实时检测（蓝牙采集 + PQRST 波检测）
├── offline.py                   # ★ 离线文件检测（MIT-BIH 数据库）
├── BLE_data.py                  # ★ 蓝牙数据采集工具
├── Temp_Method.py               #    临时实验代码
├── test.py                      #    测试脚本
│
├── qrs_detector/                # QRS 检测器参考实现
│   ├── QRSDetectorOffline.py    #    离线检测器
│   ├── QRSDetectorOnline.py     #    在线检测器
│   └── README.md                #    模块说明文档
│
├── tradition/                   # 传统 ECG 算法集合
│   ├── pan_tomkins_qrs.py       #    Pan-Tomkins 算法完整实现
│   ├── pan_tomkins_qrs_single.py #    单导联版本
│   ├── hilbert_qrs.py           #    希尔伯特变换算法
│   ├── comprehensive_ecg_detector.py  #    综合 ECG 特征点检测（P-QRS-T）
│   ├── ecg_full_analysis.py     #    完整 ECG 分析系统
│   ├── Filter.py                #    滤波器工具
│   ├── kalman.py                #    卡尔曼滤波
│   ├── ArrhythmiaFilter.py      #    心律失常过滤
│   ├── iir.py                   #    IIR 滤波器实验
│   ├── fir.py                   #    FIR 滤波器实验
│   ├── transform_ecg.py         #    ECG 信号变换
│   └── *.md                     #    算法分析文档
│
├── ecg_deepl_method/            # 深度学习方法
│   ├── ecg_cnn_1/               #    CNN 实现
│   ├── ecg-experiment-1/        #    实验 1
│   │   ├── model.py             #       模型定义
│   │   ├── train.py             #       训练脚本
│   │   ├── predict.py           #       预测脚本
│   │   └── load.py              #       数据加载
│   ├── ecg-master/              #    主实验代码
│   │   ├── network.py           #       网络定义
│   │   ├── train.py             #       训练脚本
│   │   └── predict.py           #       预测脚本
│   ├── Dataset_Study/           #    数据集研究
│   ├── show_data.py             #    数据可视化
│   └── count_records.py         #    记录统计
│
├── Information/                 # 技术文档与资料
│   ├── MIT-BIH.md               #    MIT-BIH 数据库说明
│   ├── MIT-BIH数据库.md         #    数据库详细说明
│   ├── ECG learn.md             #    ECG 学习笔记
│   ├── documents.md             #    QRS 检测标准
│   ├── connect.md               #    12 导联电极配置
│   ├── 心电信号识别分类算法综述.pdf
│   ├── QRS 波群检测算法测试方案.pdf
│   ├── YY 9706.247-2021医用电气设备标准.pdf
│   └── 其他技术论文...
│
├── OM6626_Application_Manual_V0.8.pdf   # 芯片应用手册
├── OM6626_reference_manual_V2.1.pdf     # 芯片参考手册
│
└── README.md                    # 本文件
```

---

## 快速开始

### 环境配置

#### 硬件要求

- **蓝牙设备**：支持 BLE 的 ECG 采集设备
- **操作系统**：Linux / macOS / Windows
- **蓝牙适配器**：支持 BLE 4.0+

#### 软件要求

- **Python**：3.8+
- **依赖库**：见下方安装说明

#### 安装依赖

```bash
pip install numpy scipy matplotlib wfdb asyncio bleak jupyter
```

或使用 requirements.txt（如果存在）：

```bash
pip install -r requirements.txt
```

#### 验证安装

```bash
# 运行在线检测需要蓝牙设备
python online.py
```

---

### 在线实时检测

#### 基本用法

```bash
python online.py
```

#### 功能说明

1. **自动扫描设备**：程序启动后会自动扫描附近的蓝牙设备
2. **连接目标设备**：根据配置自动匹配并连接目标 ECG 设备
3. **实时数据采集**：接收蓝牙传输的 ECG 信号数据
4. **实时检测与显示**：在 5 个子图中实时显示处理过程

#### 配置设备

在 [online.py](online.py) 中修改设备参数：

```python
DEVICE_NAME = "YOUR_DEVICE_NAME"

if DEVICE_NAME == "YOUR_DEVICE_NAME":
    device_param = {
        "name": DEVICE_NAME,
        "address": "XX:XX:XX:XX:XX:XX",  # 替换为实际 MAC 地址
        "service_uuid": "YOUR_SERVICE_UUID",
        "rx_uuid": "YOUR_RX_UUID",
        "tx_uuid": "YOUR_TX_UUID",
    }
```

#### 更换导联类型

修改蓝牙采集器类的初始化参数：

```python
self.qrs_detector = RealTimeECGDetector(signal_name="MLII")  # 可改为 V1, V2, I 等
```

支持的导联类型：
- 肢体导联：I, MLII, MLIII, aVR, aVL, aVF
- 胸导联：V1, V2, V3, V4, V5, V6

---

### 离线文件检测

#### 基本用法

```bash
python offline.py
```

#### 配置数据集路径

在 [offline.py](offline.py) 中修改：

```python
root = "YOUR_DATABASE_PATH"  # 替换为 MIT-BIH 数据库实际路径
```

#### 选择检测记录

```python
numberSet = ['100', '101', '103', '105', '106', ...]  # 要处理的记录编号
```

#### 选择目标导联

```python
target_lead = "MLII"  # 可修改为其他导联
```

---

### 蓝牙数据采集

`[BLE_data.py](BLE_data.py)` 提供了基础的蓝牙通信功能：

#### 扫描设备

```bash
python BLE_data.py
```

此命令会列出所有附近的蓝牙设备及其 MAC 地址。

#### 主要用途

- 蓝牙设备调试
- MAC 地址获取
- 通信协议测试
- 设备连接测试

---

## 算法原理

### Pan-Tomkins 算法详解

本项目采用改进的 Pan-Tomkins 算法，这是 QRS 波检测的经典方法。

#### 算法步骤

**1. 带通滤波**（1-50 Hz 可调）
   - 去除基线漂移（< 5 Hz）
   - 滤除高频噪声（> 15-50 Hz）
   - 使用 Butterworth 滤波器
   - 原始信号与滤波信号加权组合

**2. 微分处理**
   - 5 点中心差分公式
   - 突出 QRS 波的高斜率特性
   - 公式：`f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)`

**3. 平方运算**
   - `y = x²`
   - 使所有值为正
   - 放大高斜率点

**4. 移动窗口积分**
   - 窗口大小：100 ms
   - 平滑信号
   - 提取 QRS 波特征

**5. 自适应阈值检测**
   - 滑动窗口：约3秒
   - 动态阈值：使用指数移动平均（EMA）平滑
   - 不应期：200-500ms（避免重复检测）

**6. 相位延迟补偿**
   - 补偿滤波和积分引入的延迟
   - 在补偿位置附近搜索真实峰值

#### 关键改进

- **导联自适应**：根据不同导联特性调整参数
- **加权组合**：原始信号与滤波信号加权组合
- **滑动窗口**：适应信号变化的动态阈值
- **EMA 平滑**：避免阈值突变
- **不应期保护**：避免同一 QRS 波被重复检测
- **相位延迟补偿**：提高定位精度

---

### 关键参数优化

不同导联使用不同的优化参数：

| 导联类型 | 低频截止 (Hz) | 高频截止 (Hz) | 积分窗口 (s) | 不应期 (s) | 阈值系数 |
|:---------|:-------------|:-------------|:-------------|:-----------|:---------|
| V1       | 1            | 50.0         | 0.100        | 0.20       | 1.2      |
| V2       | 3            | 30.0         | 0.100        | 0.20       | 1.3      |
| V3-V6    | 5            | 15.0         | 0.100        | 0.20       | 1.4      |
| I        | 0.5          | 40.0         | 0.100        | 0.40       | 1.3      |
| MLII     | 5            | 15.0         | 0.100        | 0.50       | 1.4      |
| MLIII    | 5            | 15.0         | 0.100        | 0.20       | 1.4      |
| aVR/aVL/aVF | 5         | 15.0         | 0.100        | 0.20       | 1.4      |
| 其他     | 5            | 15.0         | 0.100        | 0.20       | 1.4      |

这些参数经过针对不同导联特性的实验优化，确保在各种信号条件下都能获得良好的检测效果。

---

### 完整 PQRST 波检测

本系统不仅检测 QRS 波，还支持完整的 PQRST 波检测：

#### R 峰检测

- 基于 Pan-Tomkins 算法检测
- 自适应阈值 + EMA 平滑
- 不应期保护避免重复检测

#### Q 波检测（R峰前的负向波）

- 搜索窗口：R峰前10-80ms
- 检测条件：幅值明显低于R峰（< 70%）
- 最小幅值差：0.01 mV

#### S 波检测（R峰后的负向波）

- 搜索窗口：R峰后10-100ms
- 检测条件：幅值明显低于R峰（< 70%）
- 最小幅值差：0.01 mV

#### P 波检测（心房去极化波）

- 搜索窗口：R峰前40-200ms
- 检测条件：正向小波，幅值远小于R峰（< 25%）
- 最小幅值：0.02 mV
- 最大宽度：120ms

#### T 波检测（心室复极化波）

- 搜索窗口：R峰后150-400ms
- 检测条件：正向宽波，幅值小于R峰（< 60%）
- 最小幅值：0.05 mV
- 最大宽度：200ms

---

### 其他算法

#### 希尔伯特变换算法

- 利用希尔伯特变换提取信号包络
- 对噪声更鲁棒
- 适用于低信噪比信号

详见 [tradition/hilbert_qrs.py](tradition/hilbert_qrs.py)

#### 综合ECG特征点检测

- 完整的 P、Q、R、S、T 波检测
- 支持波起止点检测
- 适用于完整ECG分析

详见 [tradition/comprehensive_ecg_detector.py](tradition/comprehensive_ecg_detector.py)

#### 深度学习方法

- **CNN**：卷积神经网络自动提取特征
- **RNN/LSTM**：循环网络处理时序信号
- **Transformer**：注意力机制模型

详见 [ecg_deepl_method/](ecg_deepl_method/)

---

### 设备配置指南

#### 添加新设备

在 `online.py` 或 `BLE_data.py` 中添加新的设备配置：

```python
device = "YOUR_DEVICE_NAME"
if device == "YOUR_DEVICE_NAME":
    device_param = {
        "name": device,
        "address": "XX:XX:XX:XX:XX:XX",  # 替换为实际 MAC 地址
        "service_uuid": "YOUR_SERVICE_UUID",
        "rx_uuid": "YOUR_RX_UUID",
        "tx_uuid": "YOUR_TX_UUID",
    }
```

#### 设备匹配策略

程序按以下优先级匹配设备：

1. **MAC 地址匹配**（最可靠）
2. **设备名称匹配**
3. **服务 UUID 匹配**

---

### OM6626 芯片

- **应用手册**：[OM6626_Application_Manual_V0.8.pdf](OM6626_Application_Manual_V0.8.pdf)
- **参考手册**：[OM6626_reference_manual_V2.1.pdf](OM6626_reference_manual_V2.1.pdf)

---

## 辅助模块

### qrs_detector 参考实现

QRS 检测器的参考实现，提供另一种实现思路和代码组织方式。

- **[QRSDetectorOffline.py](qrs_detector/QRSDetectorOffline.py)**：离线检测器实现
- **[QRSDetectorOnline.py](qrs_detector/QRSDetectorOnline.py)**：在线检测器实现
- **[README.md](qrs_detector/README.md)**：详细的模块文档

---

### tradition 传统算法集合

传统 ECG 算法集合，包含多种经典算法实现：

#### 核心算法

- **[pan_tomkins_qrs.py](tradition/pan_tomkins_qrs.py)**：完整的 Pan-Tomkins 算法实现
- **[pan_tomkins_qrs_single.py](tradition/pan_tomkins_qrs_single.py)**：单导联优化版本
- **[hilbert_qrs.py](tradition/hilbert_qrs.py)**：基于希尔伯特变换的 QRS 检测
- **[comprehensive_ecg_detector.py](tradition/comprehensive_ecg_detector.py)**：综合 ECG 特征点检测（P-QRS-T）
- **[ecg_full_analysis.py](tradition/ecg_full_analysis.py)**：完整 ECG 分析系统

#### 信号处理工具

- **[Filter.py](tradition/Filter.py)**：基础滤波器（IIR、FIR）
- **[kalman.py](tradition/kalman.py)**：卡尔曼滤波器实现
- **[ArrhythmiaFilter.py](tradition/ArrhythmiaFilter.py)**：心律失常过滤算法
- **[iir.py](tradition/iir.py)**：IIR 滤波器实验
- **[fir.py](tradition/fir.py)**：FIR 滤波器实验
- **[transform_ecg.py](tradition/transform_ecg.py)**：ECG 信号变换

#### 算法分析文档

- `基础Pan-Tomkins QRS检测算法详细分析.md`
- `希尔伯特QRS检测算法详细分析.md`
- `综合ECG特征点检测算法详细分析.md`
- `QRS检测优化方案.md`

#### 运行传统算法

```bash
# Pan-Tomkins 算法
cd tradition
python pan_tomkins_qrs.py

# 希尔伯特变换算法
python hilbert_qrs.py

# 综合特征点检测
python comprehensive_ecg_detector.py
```

---

### ecg_deepl_method 深度学习方法

深度学习方法探索，包含多种实验和实现：

- **ecg_cnn_1/**：基于 CNN 的 ECG 分类
- **ecg-experiment-1/**：深度学习实验 1
  - **model.py**：三种CNN模型（Model_1: 4层CNN, Model_2: 残差网络, Model_3: 注意力机制）
  - **train.py**：模型训练
  - **predict.py**：模型预测
  - **load.py**：数据加载
- **ecg-master/**：主实验代码
- **Dataset_Study/**：数据集研究与分析
- **[show_data.py](ecg_deepl_method/show_data.py)**：数据可视化工具
- **[count_records.py](ecg_deepl_method/count_records.py)**：数据集统计工具

#### 探索深度学习方法

```bash
cd ecg_deepl_method

# 查看数据集
python show_data.py

# 统计记录数量
python count_records.py
```

---

### Information 技术文档

技术文档、标准规范和学习资料：

#### 核心文档

- **[connect.md](Information/connect.md)**：12 导联 ECG 电极配置说明
  - 威尔逊中心端原理
  - 各导联测量方法
  - 电极位置说明

- **[documents.md](Information/documents.md)**：QRS 波群检测标准（YY 9706.247-2021）
  - 检测准确性要求
  - 搏-搏比对方法
  - 性能指标计算

- **[MIT-BIH.md](Information/MIT-BIH.md)**：MIT-BIH 数据库说明
- **[MIT-BIH数据库.md](Information/MIT-BIH数据库.md)**：数据库详细说明
- **[ECG learn.md](Information/ECG%20learn.md)**：ECG 学习笔记

#### 技术论文

- `心电信号识别分类算法综述.pdf`：算法综述
- `QRS 波群检测算法测试方案.pdf`：测试方案
- `YY 9706.247-2021医用电气设备标准.pdf`：行业标准
- `1707.01836v1.pdf`：深度学习相关论文
- `applsci-13-04964-v2.pdf`：应用科学论文
- `Classification_of_ECG_signals_using_machine_learning_techniques_A_survey.pdf`：机器学习分类综述

---

## 标准与测试

### YY 9706.247-2021 标准

动态心电图系统的基本安全和基本性能专用要求，对 QRS 波检测有明确规定：

#### 核心要求

**1. 检测准确性**
   - 敏感度（Se）：正确检测的 QRS 波数占总参考 QRS 波数的比例
   - 阳性预测值（+P）：正确检测的 QRS 波数占检测总 QRS 波数的比例
   - 标准：总统计敏感度/阳性预测值均 ≥ 95%

**2. 测试数据库**
   - **AHA**：80 份记录（含室性心律失常）
   - **MIT-BIH**：48 份记录（含常见/罕见心律失常）
   - **NST**：12 份记录（含噪声抑制测试）

**3. 搏-搏比对**
   - 匹配窗口：≤ 150 ms
   - 逐一心搏匹配验证
   - 漏检（FN）和假阳性（FP）均计入统计

#### 性能指标

- QRS 敏感度（QRS Se）：`QTP / (QTP + QFN)`
- QRS 阳性预测值（QRS +P）：`QTP / (QTP + QFP)`

其中：
- `QTP`：正确检测的 QRS 波总数
- `QFN`：漏检的 QRS 波数
- `QFP`：假阳性 QRS 波数

详见：[Information/documents.md](Information/documents.md)、`YY 9706.247-2021医用电气设备 第2-47部分：动态心电图系统的基本安全和基本性能专用要求.pdf`

---

### MIT-BIH 数据库

国际标准 ECG 数据库，包含 48 份半小时长的双通道 ECG 记录。

- **官网**：https://physionet.org/content/mitdb/
- **采样频率**：360 Hz
- **导联**：通常为 MLII 和 V1/V2/V5
- **标注**：心脏病专家标注的 QRS 波位置和类型

#### 获取 MIT-BIH 数据库

1. 访问 https://physionet.org/content/mitdb/
2. 下载完整数据库
3. 修改 [offline.py](offline.py) 中的路径

详见：[Information/MIT-BIH数据库.md](Information/MIT-BIH数据库.md)

---

## 技术栈

### 编程语言

- **Python 3.8+**：主要开发语言

### 核心库

#### 信号处理

- **NumPy**：数值计算
- **SciPy**：科学计算（滤波、信号处理）
- **WFDB**：PhysioNet 数据库读写

#### 可视化

- **Matplotlib**：实时绘图和数据可视化

#### 蓝牙通信

- **asyncio**：异步编程
- **bleak**：跨平台 BLE 蓝牙库

#### 深度学习（实验性）

- **TensorFlow / Keras**：深度学习框架
- **PyTorch**：深度学习框架

---

## 常见问题

### Q1: 无法找到蓝牙设备

**解决方案**：

1. 确保设备已开启并处于可发现模式
2. 检查设备 MAC 地址是否正确
3. 增加扫描超时时间：

```python
all_devices = await BleakScanner.discover(timeout=10.0)  # 增加到 10 秒
```

4. 检查蓝牙适配器是否支持 BLE

---

### Q2: 连接后立即断开

**可能原因**：
- 设备已被其他程序连接
- 设备不支持同时连接多个客户端
- 蓝牙信号不稳定

**解决方案**：
- 关闭其他可能连接该设备的程序
- 靠近设备以增强信号
- 重启蓝牙适配器

---

### Q3: 检测到的 QRS 波数量偏少

**调整参数**：

1. 降低阈值系数：

```python
'threshold_factor': 1.2  # 从 1.4 降低到 1.2
```

2. 调整带通滤波范围：

```python
'low': 3, 'high': 30.0  # 扩大通带范围
```

---

### Q4: 检测到的 QRS 波数量过多（误检）

**调整参数**：

1. 提高阈值系数：

```python
'threshold_factor': 1.6  # 从 1.4 提高到 1.6
```

2. 增加不应期：

```python
'refractory_period': 0.25  # 从 0.20 增加到 0.25 秒
```

3. 缩小带通滤波范围：

```python
'low': 5, 'high': 15.0  # 缩小通带范围
```

---

### Q5: 实时显示卡顿

**优化方案**：

1. 减少绘图刷新频率：

```python
if len(self.signal) > 500 and sample_show_cnt % 10 == 0:  # 每10个样本更新一次
    peaks = self.detect_wave()
```

2. 减小信号缓冲区大小：

```python
self.signal_len = 500  # 从 750 减少到 500
```

3. 使用更高效的绘图库（如 PyQtGraph）

---

### Q6: 数据解析错误

**检查项目**：

1. 确认设备使用的数据格式（小端/大端）
2. 检查电压转换系数是否正确（单导联 0.288，12导联 0.318）
3. 验证 CRC 校验算法

---

## 参考资料

### 论文与文献

1. Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm." *IEEE Transactions on Biomedical Engineering*. **Pan-Tomkins 算法的原始论文**

2. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/ **标准测试数据库**

3. 心电信号识别分类算法综述. [Information/心电信号识别分类算法综述.pdf](Information/心电信号识别分类算法综述.pdf)

4. QRS 波群检测算法测试方案. [Information/QRS 波群检测算法测试方案.pdf](Information/QRS%20波群检测算法测试方案.pdf)

### 技术文档

- **12 导联电极配置**：[Information/connect.md](Information/connect.md)
- **QRS 检测标准**：[Information/documents.md](Information/documents.md)
- **MIT-BIH 数据库说明**：[Information/MIT-BIH数据库.md](Information/MIT-BIH数据库.md)
- **ECG 学习笔记**：[Information/ECG learn.md](Information/ECG%20learn.md)

### 相关项目

- **核心检测模块**：项目根目录的 [online.py](online.py)、[offline.py](offline.py)、[BLE_data.py](BLE_data.py)
- **qrs_detector**：参考实现 [qrs_detector/](qrs_detector/)
- **传统算法**：[tradition/](tradition/)
- **深度学习方法**：[ecg_deepl_method/](ecg_deepl_method/)

---

## 开发计划

### 已完成 ✅

- [x] Pan-Tomkins 算法实现（在线/离线）
- [x] 蓝牙数据采集（BLE 通信）
- [x] 实时可视化（5 子图）
- [x] 多导联参数优化
- [x] MIT-BIH 数据库支持
- [x] 自适应阈值检测（EMA 平滑）
- [x] 相位延迟补偿
- [x] 完整 PQRST 波检测
- [x] 传统算法实现（Pan-Tomkins、希尔伯特变换）
- [x] 深度学习方法探索
- [x] 完整的项目文档

### 进行中 🚧

- [ ] 性能优化（实时显示流畅度）
- [ ] 心律失常分类
- [ ] 算法性能评估与优化

### 计划中 📋

- [ ] 深度学习模型训练与部署
- [ ] YY 9706.247-2021 标准合规性测试
- [ ] 用户界面（GUI）
- [ ] 数据导出与报告生成
- [ ] 移动端适配
