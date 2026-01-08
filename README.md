# ECG Signal Processing and QRS Wave Detection

This workspace focuses on the research and implementation of ECG signal processing algorithms. The core is a real-time ECG waveform detection system based on an improved Pan-Tomkins algorithm, along with exploratory research on deep learning methods and traditional algorithms.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Environment Setup](#environment-setup)
  - [Online Real-time Detection](#online-real-time-detection)
  - [Offline File Detection](#offline-file-detection)
  - [Bluetooth Test Tool](#bluetooth-test-tool)
- [Core Module Details](#core-module-details)
  - [Parameter Configuration System](#parameter-configuration-system)
  - [Online Detector](#online-detector)
  - [Offline Detector](#offline-detector)
- [Algorithm Principles](#algorithm-principles)
  - [Pan-Tomkins Algorithm Details](#pan-tomkins-algorithm-details)
  - [Key Parameter Optimization](#key-parameter-optimization)
  - [Complete PQRST Wave Detection](#complete-pqrst-wave-detection)
  - [Other Algorithms](#other-algorithms)
- [Hardware and Protocols](#hardware-and-protocols)
  - [Supported Devices](#supported-devices)
  - [Device Configuration Guide](#device-configuration-guide)
- [Auxiliary Modules](#auxiliary-modules)
  - [qrs_detector Reference Implementation](#qrs_detector-reference-implementation)
  - [tradition Traditional Algorithm Collection](#tradition-traditional-algorithm-collection)
  - [ecg_deepl_method Deep Learning Methods](#ecg_deepl_method-deep-learning-methods)
  - [Information Technical Documentation](#information-technical-documentation)
- [Standards and Testing](#standards-and-testing)
  - [YY 9706.247-2021 Standard](#yy-9706247-2021-standard)
  - [MIT-BIH Database](#mit-bih-database)
- [Tech Stack](#tech-stack)
- [FAQ](#faq)
- [References](#references)
- [Development Plan](#development-plan)

---

## Project Overview

This project is a complete ECG signal processing solution with the following main features:

### Core Features

- **Real-time Detection**: Supports real-time ECG signal acquisition and PQRST wave detection via Bluetooth Low Energy (BLE) devices
- **Offline Analysis**: Supports offline analysis of standard databases like MIT-BIH
- **Multi-algorithm Comparison**: Includes traditional algorithms (Pan-Tomkins, Hilbert transform) and deep learning methods
- **Visualization Display**: Real-time display of each stage of signal processing with 5 synchronized subplots
- **Lead Adaptive**: Optimized parameters for different ECG leads (MLII, V1-V6, I, II, III, aVR, aVL, aVF)
- **Complete Waveform Detection**: Supports detection of P, Q, R, S, T five characteristic waves
- **Parameter Configuration Separation**: Unified JSON parameter configuration file for easy tuning and maintenance

### Core Technologies

- **Pan-Tomkins Algorithm**: Classic QRS wave detection algorithm
- **Adaptive Bandpass Filtering**: Adjust filter parameters based on different lead characteristics (1-50 Hz adjustable)
- **Sliding Window Threshold Detection**: Uses Exponential Moving Average (EMA) smoothed threshold to adapt to signal changes
- **Phase Delay Compensation**: Compensates for phase delay introduced by filtering and integration
- **Asynchronous Bluetooth Communication**: Efficient Bluetooth data acquisition based on `asyncio` and `bleak`
- **Parameter Configuration System**: JSON configuration file for unified management of all lead parameters in online/offline modes

### Application Scenarios

- Dynamic ECG (Holter) system development
- Arrhythmia detection algorithm research
- ECG signal quality assessment
- Real-time heart rate monitoring devices
- Medical device compliance testing

---

## Project Structure

```
workspace-ecg/
|
├── online.py                    # ★ Online real-time detection (Bluetooth acquisition + PQRST wave detection)
├── offline.py                   # ★ Offline file detection (MIT-BIH database)
├── signal_params.json           # ★ Lead parameter configuration file (JSON format)
├── signal_params.py             # ★ Parameter loading and management module
├── ble_test.py                  # ★ Bluetooth function testing tool
├── unit_test.py                 # ★ Unit testing script
|
├── qrs_detector/                # QRS detector reference implementation
│   ├── QRSDetectorOffline.py    #    Offline detector
│   ├── QRSDetectorOnline.py     #    Online detector
│   └── README.md                #    Module documentation
|
├── tradition/                   # Traditional ECG algorithm collection
│   ├── pan_tomkins_qrs.py       #    Complete Pan-Tomkins algorithm implementation
│   ├── pan_tomkins_qrs_single.py #    Single-lead version
│   ├── hilbert_qrs.py           #    Hilbert transform algorithm
│   ├── comprehensive_ecg_detector.py  #    Comprehensive ECG feature point detection (P-QRS-T)
│   ├── ecg_full_analysis.py     #    Complete ECG analysis system
│   ├── Filter.py                #    Filter tools
│   ├── kalman.py                #    Kalman filtering
│   ├── ArrhythmiaFilter.py      #    Arrhythmia filtering
│   ├── iir.py                   #    IIR filter experiments
│   ├── fir.py                   #    FIR filter experiments
│   ├── transform_ecg.py         #    ECG signal transformation
│   └── *.md                     #    Algorithm analysis documents
|
├── ecg_deepl_method/            # Deep learning methods
│   ├── ecg_cnn_1/               #    CNN implementation
│   ├── ecg-experiment-1/        #    Experiment 1
│   ├── ecg-master/              #    Main experiment code
│   ├── Dataset_Study/           #    Dataset research
│   ├── show_data.py             #    Data visualization
│   └── count_records.py         #    Record statistics
|
├── Information/                 # Technical documentation and materials
│   ├── MIT-BIH.md               #    MIT-BIH database description
│   ├── MIT-BIH数据库.md         #    Database detailed description
│   ├── ECG learn.md             #    ECG learning notes
│   ├── documents.md             #    QRS detection standards
│   ├── connect.md               #    12-lead electrode configuration
│   └── *.pdf                    #    Technical papers and standard documents
|
├── .gitignore                   # Git ignore rules configuration
|
└── README.md                    # This file
```

### File Description

| File | Description |
|:-----|:-----|
| [online.py](online.py) | Online real-time detection main program, contains `RealTimeECGDetector` class and `BlueToothCollector` class |
| [offline.py](offline.py) | Offline file detection program, contains `PanTomkinsQRSDetectorOffline` class |
| [signal_params.json](signal_params.json) | Lead parameter configuration file, defines all lead parameters for online/offline modes |
| [signal_params.py](signal_params.py) | Parameter loading module, provides `get_signal_params()` unified interface |
| [ble_test.py](ble_test.py) | Bluetooth function testing tool, supports device scanning, connection testing, data parsing, etc. |
| [unit_test.py](unit_test.py) | Unit testing script |

---

## Quick Start

### Environment Setup

#### Hardware Requirements

- **Bluetooth Device**: BLE-enabled ECG acquisition device
- **Operating System**: Linux / macOS / Windows
- **Bluetooth Adapter**: BLE 4.0+ support

#### Software Requirements

- **Python**: 3.8+
- **Dependencies**: See installation instructions below

#### Install Dependencies

```bash
pip install numpy scipy matplotlib wfdb asyncio bleak jupyter
```

Or use requirements.txt (if available):

```bash
pip install -r requirements.txt
```

#### Verify Installation

```bash
# Running online detection requires Bluetooth device
python online.py
```

---

### Online Real-time Detection

#### Basic Usage

```bash
python online.py
```

#### Function Description

1. **Automatic Device Scanning**: The program automatically scans for nearby Bluetooth devices on startup
2. **Connect to Target Device**: Automatically matches and connects to the target ECG device based on configuration
3. **Real-time Data Acquisition**: Receives ECG signal data transmitted via Bluetooth
4. **Real-time Detection and Display**: Displays the processing process in 5 subplots in real-time

#### Configure Device

Modify device parameters in [online.py](online.py):

```python
DEVICE_NAME = "YOUR_DEVICE_NAME"

if DEVICE_NAME == "YOUR_DEVICE_NAME":
    device_param = {
        "name": DEVICE_NAME,
        "address": "XX:XX:XX:XX:XX:XX",  # Replace with actual MAC address
        "service_uuid": "YOUR_SERVICE_UUID",
        "rx_uuid": "YOUR_RX_UUID",
        "tx_uuid": "YOUR_TX_UUID",
    }
```

#### Change Lead Type

Modify the initialization parameters of the Bluetooth collector class:

```python
self.qrs_detector = RealTimeECGDetector(signal_name="MLII")  # Can be changed to V1, V2, I, etc.
```

Supported lead types:
- Limb leads: I, MLII, MLIII, aVR, aVL, aVF
- Precordial leads: V1, V2, V3, V4, V5, V6

---

### Offline File Detection

#### Basic Usage

```bash
python offline.py
```

#### Configure Dataset Path

Modify in [offline.py](offline.py):

```python
root = "YOUR_DATABASE_PATH"  # Replace with actual MIT-BIH database path
```

#### Select Detection Records

```python
numberSet = ['100', '101', '103', '105', '106', ...]  # Record numbers to process
```

#### Select Target Lead

```python
target_lead = "MLII"  # Can be changed to other leads
```

---

### Bluetooth Test Tool

[ble_test.py](ble_test.py) provides a complete Bluetooth function testing tool with multiple test modes:

#### Test Functions

Enable different tests by setting flags at the top of the file:

```python
# Test function flag settings
TEST_SCAN_DEVICES = 0          # Test 1: Simple Bluetooth device scan
TEST_CONNECT_AND_VIEW = 0      # Test 2: Connect to device and view services
TEST_DATA_PARSE = 0            # Test 3: Data parsing test
TEST_ECG_COLLECTION = 0        # Test 4: ECG data acquisition and real-time plotting (simple version)
TEST_QINGXUN_COLLECTOR = 0     # Test 5: QingXunBlueToothCollector class complete test
```

#### Usage

1. Set the corresponding test flag to `1`
2. Run the script:

```bash
python ble_test.py
```

#### Main Uses

- Bluetooth device scanning and discovery
- MAC address acquisition
- Device connection testing
- Service and characteristic viewing
- Data packet parsing testing
- ECG data acquisition verification

---

## Core Module Details

### Parameter Configuration System

The project adopts a unified parameter configuration system that centrally manages all lead parameters.

#### signal_params.json

A JSON format parameter configuration file that defines parameters for each lead in online and offline modes:

```json
{
  "online": {
    "MLII": {
      "low": 5,
      "high": 15.0,
      "filter_order": 5,
      "original_weight": 0.2,
      "filtered_weight": 0.8,
      "integration_window_size": 0.100,
      "refractory_period": 0.50,
      "threshold_factor": 1.4,
      ...
    }
  },
  "offline": {
    "MLII": { ... }
  }
}
```

#### signal_params.py

Parameter loading module providing a unified parameter retrieval interface:

```python
from signal_params import get_signal_params

# Get parameters for MLII lead in online mode
params = get_signal_params('online', 'MLII')

# Get parameters for V1 lead in offline mode
params = get_signal_params('offline', 'V1')
```

#### Parameter Description

| Parameter | Description | Applicable Modes |
|:-----|:-----|:---------|
| `low` | Bandpass filter low cutoff frequency (Hz) | online / offline |
| `high` | Bandpass filter high cutoff frequency (Hz) | online / offline |
| `filter_order` | Butterworth filter order | online / offline |
| `original_weight` | Original signal weight | online / offline |
| `filtered_weight` | Filtered signal weight | online / offline |
| `integration_window_size` | Integration window size (seconds) | online / offline |
| `refractory_period` | QRS detection refractory period (seconds) | online / offline |
| `threshold_factor` | Threshold coefficient | online / offline |
| `compensation_ms` | Phase delay compensation time (milliseconds) | online |
| `ema_alpha` | EMA threshold smoothing coefficient | online |
| `q_wave_*` | Q wave detection parameters | online |
| `s_wave_*` | S wave detection parameters | online |
| `p_wave_*` | P wave detection parameters | online |
| `t_wave_*` | T wave detection parameters | online |
| `detection_window_size` | Detection window size (seconds) | offline |
| `overlap_window_size` | Overlap window size (seconds) | offline |

### Online Detector

[online.py](online.py) contains the core implementation of real-time ECG waveform detection.

#### RealTimeECGDetector Class

A real-time ECG waveform detector based on the Pan-Tomkins algorithm.

```python
class RealTimeECGDetector:
    def __init__(self, signal_name="MLII"):
        # Initialize detector, automatically loads parameters from signal_params.json
        ...
```

#### BlueToothCollector Class

Bluetooth data collector responsible for communicating with BLE devices.

```python
class BlueToothCollector:
    def __init__(self, device_param, qrs_detector):
        # Initialize Bluetooth collector
        ...

    async def start_collection(self):
        # Start data acquisition
        ...
```

### Offline Detector

[offline.py](offline.py) contains the implementation of offline file analysis.

#### PanTomkinsQRSDetectorOffline Class

An offline QRS wave detector based on the Pan-Tomkins algorithm.

```python
class PanTomkinsQRSDetectorOffline:
    def __init__(self, signal_name="MLII"):
        # Initialize detector, sampling frequency fixed at 360 Hz
        ...

    def detect_qrs(self, signal_data):
        # Detect QRS waves
        ...
```

---

## Algorithm Principles

### Pan-Tomkins Algorithm Details

This project uses an improved Pan-Tomkins algorithm, a classic method for QRS wave detection.

#### Algorithm Steps

**1. Bandpass Filtering** (1-50 Hz adjustable)
   - Remove baseline drift (< 5 Hz)
   - Filter high-frequency noise (> 15-50 Hz)
   - Use Butterworth filter
   - Weighted combination of original and filtered signals

**2. Differentiation**
   - 5-point central difference formula
   - Highlight high slope characteristics of QRS waves
   - Formula: `f'(x) ≈ (f(x-2h) - 8f(x-h) + 8f(x+h) - f(x+2h)) / (12h)`

**3. Squaring**
   - `y = x²`
   - Make all values positive
   - Amplify high slope points

**4. Moving Window Integration**
   - Window size: 100 ms
   - Smooth signal
   - Extract QRS wave features

**5. Adaptive Threshold Detection**
   - Sliding window: approximately 3 seconds
   - Dynamic threshold: smoothed using Exponential Moving Average (EMA)
   - Refractory period: 200-500ms (avoid duplicate detection)

**6. Phase Delay Compensation**
   - Compensate for delay introduced by filtering and integration
   - Search for true peak near compensation position

#### Key Improvements

- **Lead Adaptive**: Adjust parameters based on different lead characteristics
- **Weighted Combination**: Weighted combination of original and filtered signals
- **Sliding Window**: Dynamic threshold that adapts to signal changes
- **EMA Smoothing**: Avoid sudden threshold changes
- **Refractory Period Protection**: Avoid repeated detection of the same QRS wave
- **Phase Delay Compensation**: Improve positioning accuracy

---

### Key Parameter Optimization

Different leads use different optimized parameters:

| Lead Type | Low Cutoff (Hz) | High Cutoff (Hz) | Integration Window (s) | Refractory Period (s) | Threshold Factor |
|:---------|:-------------|:-------------|:-------------|:-----------|:---------|
| V1       | 1            | 50.0         | 0.100        | 0.20       | 1.2      |
| V2       | 3            | 30.0         | 0.100        | 0.20       | 1.3      |
| V3-V6    | 5            | 15.0         | 0.100        | 0.20       | 1.4      |
| I        | 0.5          | 40.0         | 0.100        | 0.40       | 1.3      |
| MLII     | 5            | 15.0         | 0.100        | 0.50       | 1.4      |
| MLIII    | 5            | 15.0         | 0.100        | 0.20       | 1.4      |
| aVR/aVL/aVF | 5         | 15.0         | 0.100        | 0.20       | 1.4      |
| Others   | 5            | 15.0         | 0.100        | 0.20       | 1.4      |

These parameters are experimentally optimized for different lead characteristics to ensure good detection results under various signal conditions.

---

### Complete PQRST Wave Detection

This system not only detects QRS waves but also supports complete PQRST wave detection:

#### R Peak Detection

- Based on Pan-Tomkins algorithm detection
- Adaptive threshold + EMA smoothing
- Refractory period protection to avoid duplicate detection

#### Q Wave Detection (negative wave before R peak)

- Search window: 10-80ms before R peak
- Detection condition: amplitude significantly lower than R peak (< 70%)
- Minimum amplitude difference: 0.01 mV

#### S Wave Detection (negative wave after R peak)

- Search window: 10-100ms after R peak
- Detection condition: amplitude significantly lower than R peak (< 70%)
- Minimum amplitude difference: 0.01 mV

#### P Wave Detection (atrial depolarization wave)

- Search window: 40-200ms before R peak
- Detection condition: positive small wave, amplitude much smaller than R peak (< 25%)
- Minimum amplitude: 0.02 mV
- Maximum width: 120ms

#### T Wave Detection (ventricular repolarization wave)

- Search window: 150-400ms after R peak
- Detection condition: positive wide wave, amplitude smaller than R peak (< 60%)
- Minimum amplitude: 0.05 mV
- Maximum width: 200ms

---

### Other Algorithms

#### Hilbert Transform Algorithm

- Uses Hilbert transform to extract signal envelope
- More robust to noise
- Suitable for low SNR signals

See [tradition/hilbert_qrs.py](tradition/hilbert_qrs.py)

#### Comprehensive ECG Feature Point Detection

- Complete P, Q, R, S, T wave detection
- Supports wave onset and offset detection
- Suitable for complete ECG analysis

See [tradition/comprehensive_ecg_detector.py](tradition/comprehensive_ecg_detector.py)

#### Deep Learning Methods

- **CNN**: Convolutional neural networks for automatic feature extraction
- **RNN/LSTM**: Recurrent networks for processing time series signals
- **Transformer**: Attention mechanism models

See [ecg_deepl_method/](ecg_deepl_method/)

---

### Device Configuration Guide

#### Adding New Devices

Add new device configuration in [online.py](online.py):

```python
DEVICE_NAME = "YOUR_DEVICE_NAME"
if DEVICE_NAME == "YOUR_DEVICE_NAME":
    device_param = {
        "name": DEVICE_NAME,
        "address": "XX:XX:XX:XX:XX:XX",  # Replace with actual MAC address
        "service_uuid": "YOUR_SERVICE_UUID",
        "rx_uuid": "YOUR_RX_UUID",
        "tx_uuid": "YOUR_TX_UUID",
    }
```

#### Getting Device Information

Use [ble_test.py](ble_test.py) to scan and get device information:

```python
# Set flag in ble_test.py
TEST_SCAN_DEVICES = 1

# Run to get device list
python ble_test.py
```

#### Device Matching Strategy

The program matches devices in the following priority order:

1. **MAC Address Matching** (most reliable)
2. **Device Name Matching**
3. **Service UUID Matching**

---

## Auxiliary Modules

### qrs_detector Reference Implementation

Reference implementation of the QRS detector, providing an alternative implementation approach and code organization.

- **[QRSDetectorOffline.py](qrs_detector/QRSDetectorOffline.py)**: Offline detector implementation
- **[QRSDetectorOnline.py](qrs_detector/QRSDetectorOnline.py)**: Online detector implementation
- **[README.md](qrs_detector/README.md)**: Detailed module documentation

---

### tradition Traditional Algorithm Collection

A collection of traditional ECG algorithms, including various classic algorithm implementations:

#### Core Algorithms

- **[pan_tomkins_qrs.py](tradition/pan_tomkins_qrs.py)**: Complete Pan-Tomkins algorithm implementation
- **[pan_tomkins_qrs_single.py](tradition/pan_tomkins_qrs_single.py)**: Single-lead optimized version
- **[hilbert_qrs.py](tradition/hilbert_qrs.py)**: Hilbert transform-based QRS detection
- **[comprehensive_ecg_detector.py](tradition/comprehensive_ecg_detector.py)**: Comprehensive ECG feature point detection (P-QRS-T)
- **[ecg_full_analysis.py](tradition/ecg_full_analysis.py)**: Complete ECG analysis system

#### Signal Processing Tools

- **[Filter.py](tradition/Filter.py)**: Basic filters (IIR, FIR)
- **[kalman.py](tradition/kalman.py)**: Kalman filter implementation
- **[ArrhythmiaFilter.py](tradition/ArrhythmiaFilter.py)**: Arrhythmia filtering algorithm
- **[iir.py](tradition/iir.py)**: IIR filter experiments
- **[fir.py](tradition/fir.py)**: FIR filter experiments
- **[transform_ecg.py](tradition/transform_ecg.py)**: ECG signal transformation

#### Algorithm Analysis Documents

- `基础Pan-Tomkins QRS检测算法详细分析.md`
- `希尔伯特QRS检测算法详细分析.md`
- `综合ECG特征点检测算法详细分析.md`
- `QRS检测优化方案.md`

#### Running Traditional Algorithms

```bash
# Pan-Tomkins algorithm
cd tradition
python pan_tomkins_qrs.py

# Hilbert transform algorithm
python hilbert_qrs.py

# Comprehensive feature point detection
python comprehensive_ecg_detector.py
```

---

### ecg_deepl_method Deep Learning Methods

Deep learning method exploration, including various experiments and implementations:

- **ecg_cnn_1/**: CNN-based ECG classification
- **ecg-experiment-1/**: Deep learning experiment 1
  - **model.py**: Three CNN models (Model_1: 4-layer CNN, Model_2: Residual network, Model_3: Attention mechanism)
  - **train.py**: Model training
  - **predict.py**: Model prediction
  - **load.py**: Data loading
- **ecg-master/**: Main experiment code
- **Dataset_Study/**: Dataset research and analysis
- **[show_data.py](ecg_deepl_method/show_data.py)**: Data visualization tool
- **[count_records.py](ecg_deepl_method/count_records.py)**: Dataset statistics tool

#### Exploring Deep Learning Methods

```bash
cd ecg_deepl_method

# View dataset
python show_data.py

# Count records
python count_records.py
```

---

### Information Technical Documentation

Technical documentation, standards, and learning materials:

#### Core Documents

- **[connect.md](Information/connect.md)**: 12-lead ECG electrode configuration instructions
  - Wilson central terminal principles
  - Each lead measurement method
  - Electrode position description

- **[documents.md](Information/documents.md)**: QRS wave complex detection standards (YY 9706.247-2021)
  - Detection accuracy requirements
  - Beat-to-beat comparison method
  - Performance metric calculation

- **[MIT-BIH.md](Information/MIT-BIH.md)**: MIT-BIH database description
- **[MIT-BIH数据库.md](Information/MIT-BIH数据库.md)**: Database detailed description
- **[ECG learn.md](Information/ECG%20learn.md)**: ECG learning notes

#### Technical Papers

- `心电信号识别分类算法综述.pdf`: Algorithm review
- `QRS 波群检测算法测试方案.pdf`: Test plan
- `YY 9706.247-2021医用电气设备标准.pdf`: Industry standard
- `1707.01836v1.pdf`: Deep learning related paper
- `applsci-13-04964-v2.pdf`: Applied science paper
- `Classification_of_ECG_signals_using_machine_learning_techniques_A_survey.pdf`: Machine learning classification review

---

## Standards and Testing

### YY 9706.247-2021 Standard

Specific requirements for basic safety and essential performance of dynamic ECG systems, with clear provisions for QRS wave detection:

#### Core Requirements

**1. Detection Accuracy**
   - Sensitivity (Se): Ratio of correctly detected QRS waves to total reference QRS waves
   - Positive Predictive Value (+P): Ratio of correctly detected QRS waves to total detected QRS waves
   - Standard: Total statistical sensitivity/positive predictive value both ≥ 95%

**2. Test Database**
   - **AHA**: 80 records (including ventricular arrhythmias)
   - **MIT-BIH**: 48 records (including common/rare arrhythmias)
   - **NST**: 12 records (including noise suppression tests)

**3. Beat-to-Beat Comparison**
   - Matching window: ≤ 150 ms
   - Beat-by-beat matching verification
   - Both missed detections (FN) and false positives (FP) are counted in statistics

#### Performance Metrics

- QRS Sensitivity (QRS Se): `QTP / (QTP + QFN)`
- QRS Positive Predictive Value (QRS +P): `QTP / (QTP + QFP)`

Where:
- `QTP`: Total number of correctly detected QRS waves
- `QFN`: Number of missed QRS waves
- `QFP`: Number of false positive QRS waves

See: [Information/documents.md](Information/documents.md), `YY 9706.247-2021医用电气设备 第2-47部分：动态心电图系统的基本安全和基本性能专用要求.pdf`

---

### MIT-BIH Database

International standard ECG database containing 48 half-hour two-channel ECG records.

- **Website**: https://physionet.org/content/mitdb/
- **Sampling Frequency**: 360 Hz
- **Leads**: Typically MLII and V1/V2/V5
- **Annotations**: QRS wave positions and types annotated by cardiologists

#### Obtaining MIT-BIH Database

1. Visit https://physionet.org/content/mitdb/
2. Download complete database
3. Modify path in [offline.py](offline.py)

See: [Information/MIT-BIH数据库.md](Information/MIT-BIH数据库.md)

---

## Tech Stack

### Programming Languages

- **Python 3.8+**: Main development language

### Core Libraries

#### Signal Processing

- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (filtering, signal processing)
- **WFDB**: PhysioNet database reading and writing

#### Visualization

- **Matplotlib**: Real-time plotting and data visualization

#### Bluetooth Communication

- **asyncio**: Asynchronous programming
- **bleak**: Cross-platform BLE Bluetooth library

#### Deep Learning (Experimental)

- **TensorFlow / Keras**: Deep learning framework
- **PyTorch**: Deep learning framework

---

## FAQ

### Q1: Cannot find Bluetooth device

**Solutions**:

1. Ensure device is powered on and in discoverable mode
2. Check if device MAC address is correct
3. Increase scan timeout:

```python
all_devices = await BleakScanner.discover(timeout=10.0)  # Increase to 10 seconds
```

4. Check if Bluetooth adapter supports BLE

---

### Q2: Disconnects immediately after connection

**Possible causes**:
- Device is already connected by another program
- Device does not support multiple simultaneous client connections
- Unstable Bluetooth signal

**Solutions**:
- Close other programs that might connect to the device
- Move closer to device to strengthen signal
- Restart Bluetooth adapter

---

### Q3: Too few QRS waves detected

**Adjust parameters**:

1. Lower threshold coefficient:

```python
'threshold_factor': 1.2  # Lower from 1.4 to 1.2
```

2. Adjust bandpass filter range:

```python
'low': 3, 'high': 30.0  # Expand passband range
```

---

### Q4: Too many QRS waves detected (false positives)

**Adjust parameters**:

1. Increase threshold coefficient:

```python
'threshold_factor': 1.6  # Increase from 1.4 to 1.6
```

2. Increase refractory period:

```python
'refractory_period': 0.25  # Increase from 0.20 to 0.25 seconds
```

3. Narrow bandpass filter range:

```python
'low': 5, 'high': 15.0  # Narrow passband range
```

---

### Q5: Real-time display lag

**Optimization**:

1. Reduce drawing refresh frequency:

```python
if len(self.signal) > 500 and sample_show_cnt % 10 == 0:  # Update every 10 samples
    peaks = self.detect_wave()
```

2. Reduce signal buffer size:

```python
self.signal_len = 500  # Reduce from 750 to 500
```

3. Use more efficient plotting library (such as PyQtGraph)

---

### Q6: Data parsing error

**Check items**:

1. Confirm the data format used by the device (little-endian/big-endian)
2. Check if voltage conversion coefficient is correct (single-lead 0.288, 12-lead 0.318)
3. Verify CRC checksum algorithm

---

## References

### Papers and Literature

1. Pan, J., & Tompkins, W. J. (1985). "A real-time QRS detection algorithm." *IEEE Transactions on Biomedical Engineering*. **Original paper of Pan-Tomkins algorithm**

2. MIT-BIH Arrhythmia Database. https://physionet.org/content/mitdb/ **Standard test database**

3. 心电信号识别分类算法综述. [Information/心电信号识别分类算法综述.pdf](Information/心电信号识别分类算法综述.pdf)

4. QRS 波群检测算法测试方案. [Information/QRS 波群检测算法测试方案.pdf](Information/QRS%20波群检测算法测试方案.pdf)

### Technical Documentation

- **12-lead electrode configuration**: [Information/connect.md](Information/connect.md)
- **QRS detection standards**: [Information/documents.md](Information/documents.md)
- **MIT-BIH database description**: [Information/MIT-BIH数据库.md](Information/MIT-BIH数据库.md)
- **ECG learning notes**: [Information/ECG learn.md](Information/ECG%20learn.md)

### Related Projects

- **Core detection modules**: [online.py](online.py), [offline.py](offline.py)
- **Parameter configuration**: [signal_params.json](signal_params.json), [signal_params.py](signal_params.py)
- **Bluetooth testing**: [ble_test.py](ble_test.py)
- **qrs_detector**: Reference implementation [qrs_detector/](qrs_detector/)
- **Traditional algorithms**: [tradition/](tradition/)
- **Deep learning methods**: [ecg_deepl_method/](ecg_deepl_method/)

---

## Development Plan

### Completed ✅

- [x] Pan-Tomkins algorithm implementation (online/offline)
- [x] Bluetooth data acquisition (BLE communication)
- [x] Real-time visualization (5 subplots)
- [x] Multi-lead parameter optimization
- [x] MIT-BIH database support
- [x] Adaptive threshold detection (EMA smoothing)
- [x] Phase delay compensation
- [x] Complete PQRST wave detection
- [x] Traditional algorithm implementation (Pan-Tomkins, Hilbert transform)
- [x] Deep learning method exploration
- [x] Complete project documentation
- [x] Parameter configuration system refactoring (JSON + Python module)
- [x] Bluetooth testing tool (ble_test.py)

### In Progress 🚧

- [ ] Performance optimization (real-time display smoothness)
- [ ] Arrhythmia classification
- [ ] Algorithm performance evaluation and optimization
- [ ] Unit testing improvement

### Planned 📋

- [ ] Deep learning model training and deployment
- [ ] YY 9706.247-2021 standard compliance testing
- [ ] User interface (GUI)
- [ ] Data export and report generation
- [ ] Mobile adaptation
