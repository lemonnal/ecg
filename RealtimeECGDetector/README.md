# QRSDetector - 实时心电图 QRS 波检测系统

基于改进的 Pan-Tomkins 算法实现的实时 ECG 信号 QRS 波检测系统，支持通过蓝牙低功耗（BLE）设备进行实时心电信号采集与检测。

---

## 目录

- [项目简介](#项目简介)
- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [环境要求](#环境要求)
- [安装说明](#安装说明)
- [使用指南](#使用指南)
  - [在线实时检测](#在线实时检测)
  - [离线文件检测](#离线文件检测)
  - [蓝牙数据采集](#蓝牙数据采集)
- [算法原理](#算法原理)
- [文件说明](#文件说明)
- [设备配置](#设备配置)
- [常见问题](#常见问题)
- [参考资料](#参考资料)

---

## 项目简介

本项目是一个完整的 ECG 信号 QRS 波检测解决方案，包含：

- **在线实时检测**：通过蓝牙设备实时采集心电信号并进行 QRS 波检测
- **离线文件分析**：支持从 MIT-BIH 数据库读取信号进行离线分析
- **可视化显示**：实时展示信号处理的各个阶段（原始信号、滤波、微分、平方、积分）
- **自适应参数**：根据不同的 ECG 导联自动调整检测参数

### 核心技术

- **Pan-Tomkins 算法**：经典的 QRS 波检测算法
- **自适应带通滤波**：根据不同导联特性调整滤波参数
- **滑动窗口阈值检测**：使用自适应阈值适应信号变化
- **异步蓝牙通信**：基于 `asyncio` 和 `bleak` 实现高效蓝牙数据采集

---

## 功能特性

### 实时检测功能

- ✅ 支持蓝牙低功耗（BLE）设备实时数据采集
- ✅ 实时 QRS 波检测与标记
- ✅ 五阶段信号处理可视化
- ✅ 自动适应信号幅度变化
- ✅ 可配置的导联参数（支持 MLII, V1-V6, I, II, III, aVR, aVL, aVF）

### 离线分析功能

- ✅ 支持 MIT-BIH Arrhythmia Database 格式
- ✅ 批量处理多个记录文件
- ✅ 导联特定参数优化
- ✅ 检测结果统计与评估

### 信号处理流程

1. **带通滤波**：去除基线漂移和高频噪声
2. **微分处理**：突出 QRS 波的高斜率特性
3. **平方运算**：放大差异并使所有值为正
4. **移动窗口积分**：平滑信号并提取 QRS 波特征
5. **自适应阈值检测**：识别 QRS 波峰值位置

---

## 系统架构

```
QRSDetector/
├── online.py           # 在线实时检测主程序
├── offline.py          # 离线文件检测程序
├── BLE_data.py         # 蓝牙数据采集工具
├── Temp_Method.py      # 临时实验代码
└── test.py             # 测试脚本
```

### 模块依赖关系

```
online.py
├── PanTomkinsQRSDetectorOnline    # QRS 检测器（在线版）
│   ├── bandpass_filter()          # 带通滤波
│   ├── derivative()               # 微分处理
│   ├── squaring()                 # 平方运算
│   ├── moving_window_integration()# 移动窗口积分
│   └── threshold_detection()      # 阈值检测
└── QingXunBlueToothCollector      # 蓝牙数据采集器
    ├── start_collection()         # 开启数据采集
    ├── handle_rx()                # 接收数据回调
    └── build_protocol_packet()    # 构建协议包

offline.py
└── PanTomkinsQRSDetectorOffline   # QRS 检测器（离线版）
    └── (同上信号处理方法)
```

---

## 环境要求

### 硬件要求

- **蓝牙设备**：支持 BLE 的 ECG 采集设备（如轻迅科技的 AAA-TEST、PW-ECG-SL）
- **操作系统**：Linux / macOS / Windows
- **蓝牙适配器**：支持 BLE 4.0+

### 软件要求

- **Python**：3.8+
- **依赖库**：见下方安装说明

---

## 安装说明

### 1. 克隆项目

```bash
cd /path/to/workspace-ecg
```

### 2. 安装依赖

```bash
pip install numpy scipy matplotlib wfdb asyncio bleak
```

或使用 requirements.txt（如果存在）：

```bash
pip install -r requirements.txt
```

### 3. 验证安装

```bash
python QRSDetector/test.py
```

---

## 使用指南

### 在线实时检测

#### 基本用法

```bash
cd QRSDetector
python online.py
```

#### 功能说明

1. **自动扫描设备**：程序启动后会自动扫描附近的蓝牙设备
2. **连接目标设备**：根据配置自动匹配并连接目标 ECG 设备
3. **实时数据采集**：接收蓝牙传输的 ECG 信号数据
4. **实时检测与显示**：在 5 个子图中实时显示处理过程

#### 实时显示界面

程序会创建 5 个子图窗口，实时显示：

1. **原始信号**（original signal）：接收到的原始 ECG 信号
2. **滤波信号**（filtered signal）：经过带通滤波后的信号
3. **微分信号**（differentiated signal）：微分处理后的信号
4. **平方信号**（squared signal）：平方运算后的信号
5. **积分信号**（integrated signal）：移动窗口积分后的信号

检测到的 QRS 波会用**红色圆圈**标记在各个子图中。

#### 配置说明

在 `online.py` 中修改设备参数：

```python
device = "AAA-TEST"  # 或 "PW-ECG-SL"

device_param = {
    "name": device,
    "address": "EC:7A:26:9D:81:3F",  # 设备 MAC 地址
    "service_uuid": QINGXUN_UART_SERVICE_UUID,
    "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
    "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
}
```

#### 更换导联类型

修改 `QingXunBlueToothCollector` 类的初始化参数：

```python
self.qrs_detector = PanTomkinsQRSDetectorOnline(signal_name="MLII")  # 可改为 V1, V2, I 等
```

支持的导联类型：
- 肢体导联：I, MLII, MLIII, aVR, aVL, aVF
- 胸导联：V1, V2, V3, V4, V5, V6

---

### 离线文件检测

#### 基本用法

```bash
cd QRSDetector
python offline.py
```

#### 配置数据集路径

在 `offline.py` 中修改：

```python
root = "/path/to/mit-bih-arrhythmia-database-1.0.0/"
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

`BLE_data.py` 提供了基础的蓝牙通信功能：

#### 扫描设备

```bash
python BLE_data.py
```

此命令会列出所有附近的蓝牙设备及其 MAC 地址。

#### 协议说明

本项目使用**轻迅蓝牙通信协议 V1.0.1**：

**数据包格式**：
```
[功能码(2字节)] [数据长度(2字节)] [数据内容] [CRC16(2字节)]
```

**开启采集指令**（功能码 0x0001）：
```
[采集开关(1字节)] [时间戳(8字节)]
```

**ECG 数据包**：
- 每包包含 119 个采样点
- 每个采样点 2 字节（16位整数，小端格式）
- 电压转换系数：0.288 μV/LSB（单导联）或 0.318 μV/LSB（12导联）

---

## 算法原理

### Pan-Tomkins 算法流程

```
输入 ECG 信号
    ↓
1. 带通滤波 (5-15 Hz)
    ↓
2. 微分处理 (5点中心差分)
    ↓
3. 平方运算
    ↓
4. 移动窗口积分 (窗口大小: 0.08-0.10s)
    ↓
5. 自适应阈值检测
    ↓
输出 QRS 波位置
```

### 关键参数说明

不同导联使用不同的优化参数：

| 导联   | 低频截止 (Hz) | 高频截止 (Hz) | 积分窗口 (s) | 不应期 (s) | 阈值系数 |
|:------|:-------------|:-------------|:-------------|:-----------|:---------|
| V1    | 1            | 50.0         | 0.100        | 0.20       | 1.2      |
| V2    | 3            | 30.0         | 0.100        | 0.20       | 1.3      |
| V3-V6 | 5            | 15.0         | 0.100        | 0.20       | 1.4      |
| I     | 3            | 40.0         | 0.100        | 0.40       | 1.3      |
| MLII  | 3            | 40.0         | 0.100        | 0.40       | 1.3      |
| 其他  | 5            | 15.0         | 0.100        | 0.20       | 1.4      |

### 采样频率

- **在线模式**：250 Hz
- **离线模式**：360 Hz（MIT-BIH 标准）

---

## 文件说明

### [online.py](online.py)

在线实时检测主程序，包含：

- `PanTomkinsQRSDetectorOnline` 类：实时 QRS 检测器
  - 使用 `deque` 实现固定长度的信号缓冲区（750 个样本，约 3 秒）
  - 实时更新并显示检测过程
  - 支持 Matplotlib 实时可视化

- `QingXunBlueToothCollector` 类：蓝牙数据采集器
  - 自动设备匹配（MAC 地址、设备名称、UUID）
  - 协议包构建与解析
  - CRC16 校验
  - 异步数据接收

**主要功能**：
- [main()](online.py:587) 函数：主程序入口
- [handle_rx()](online.py:578) 函数：蓝牙数据接收回调
- [update_signal_and_plot()](online.py:363) 函数：更新信号并绘制图形

### [offline.py](offline.py)

离线文件检测程序，包含：

- `PanTomkinsQRSDetectorOffline` 类：离线 QRS 检测器
  - 支持批量处理 MIT-BIH 数据库
  - 导联特定参数优化
  - 检测结果输出与评估

**主要功能**：
- MIT-BIH 数据库读取与解析
- 多记录批量处理
- 导联选择与参数适配
- 检测结果统计

### [BLE_data.py](BLE_data.py)

蓝牙数据采集工具，包含：

- 基础的蓝牙设备扫描功能
- 蓝牙连接测试工具
- 协议解析示例代码（大部分已注释）

**用途**：
- 蓝牙设备调试
- MAC 地址获取
- 通信协议测试

### [Temp_Method.py](Temp_Method.py)

临时实验代码存储文件，用于测试新算法和功能。

### [test.py](test.py)

单元测试和功能验证脚本。

---

## 设备配置

### 支持的设备

#### 1. AAA-TEST（轻迅科技测试设备）

```
设备名称: AAA-TEST
MAC 地址: EC:7A:26:9D:81:3F
服务 UUID: 6e400001-b5a3-f393-e0a9-68716563686f
```

#### 2. PW-ECG-SL（轻迅科技 ECG 设备）

```
设备名称: PW-ECG-SL
MAC 地址: E2:1B:A5:DB:DE:EA
服务 UUID: 6e400001-b5a3-f393-e0a9-68716563686f
```

### 添加新设备

在 `online.py` 或 `BLE_data.py` 中添加新的设备配置：

```python
device = "YOUR_DEVICE_NAME"
if device == "YOUR_DEVICE_NAME":
    device_param = {
        "name": device,
        "address": "XX:XX:XX:XX:XX:XX",  # 替换为实际 MAC 地址
        "service_uuid": QINGXUN_UART_SERVICE_UUID,
        "rx_uuid": QINGXUN_UART_RX_CHAR_UUID,
        "tx_uuid": QINGXUN_UART_TX_CHAR_UUID,
    }
```

### 设备匹配策略

程序按以下优先级匹配设备：

1. **MAC 地址匹配**（最可靠）
2. **设备名称匹配**
3. **服务 UUID 匹配**

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

### Q2: 连接后立即断开

**可能原因**：

- 设备已被其他程序连接
- 设备不支持同时连接多个客户端
- 蓝牙信号不稳定

**解决方案**：

- 关闭其他可能连接该设备的程序
- 靠近设备以增强信号
- 重启蓝牙适配器

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

### Q5: 实时显示卡顿

**优化方案**：

1. 减少绘图刷新频率：

```python
plt.pause(0.05)  # 从 0.01 增加到 0.05 秒
```

2. 减小信号缓冲区大小：

```python
self.signal_len = 500  # 从 750 减少到 500
```

3. 降低绘图更新频率：

```python
if len(self.signal) > 500 and sample_index % 5 == 0:  # 每 5 个样本更新一次
    # 更新图形
```

### Q6: 数据解析错误

**检查项目**：

1. 确认设备使用的数据格式（小端/大端）
2. 检查电压转换系数是否正确
3. 验证 CRC 校验算法

---

## 参考资料

### 相关项目

- [qrs_detector/](../qrs_detector/) - 另一个 QRS 检测实现
- [tradition/](../tradition/) - 传统 ECG 算法集合
- [ecg_deepl_method/](../ecg_deepl_method/) - 深度学习方法

### 文档资源

- [connect.md](../connect.md) - 12 导联 ECG 电极配置说明
- [Information/](../Information/) - 相关技术文档和论文
  - MIT-BIH 数据库说明
  - ECG 算法综述
  - 检测算法测试方案

### 硬件文档

- `轻迅蓝牙通信协议V1.0.1_ql.docx` - 轻迅科技通信协议文档
- `OM6626_Application_Manual_V0.8.pdf` - OM6626 芯片应用手册
- `OM6626_reference_manual_V2.1.pdf` - OM6626 芯片参考手册

### 算法参考

- Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm." IEEE Transactions on Biomedical Engineering.
- MIT-BIH Arrhythmia Database: https://physionet.org/content/mitdb/

---

## 版本历史

- **v1.0** (2024-12)
  - 初始版本
  - 实现在线/离线检测功能
  - 支持蓝牙数据采集
  - 添加多导联参数优化

---

## 许可证

本项目遵循相关开源许可证。详见 [LICENSE](../LICENSE) 文件。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 提交 Issue
- 发送 Pull Request
- 查看项目文档

---

## 致谢

- MIT-BIH Arrhythmia Database 提供标准测试数据
- 轻迅科技提供硬件设备和技术支持
- Pan & Tompkins 的经典算法
- Python 科学计算社区（NumPy, SciPy, Matplotlib）

---

**最后更新**: 2024-12-31
