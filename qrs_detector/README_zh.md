# 基于 Pan-Tomkins 算法的 Python 在线和离线 ECG QRS 检测器

## 作者
* Michał Sznajder (雅盖隆大学) - 技术联系人 (msznajder@gmail.com)
* Marta Łukowska (雅盖隆大学)

## 简介

本仓库中发布的模块是基于 Pan-Tomkins 算法 (Pan J., Tompkins W. J., _A real-time QRS detection algorithm,_ IEEE Transactions on Biomedical Engineering, Vol. BME-32, No. 3, March 1985, pp. 230-236) 的 Python 实现，用于在线和离线检测 ECG 信号中的 QRS 复波。

QRS 复波对应于人体心脏左右心室的去极化。它是 ECG 信号中最直观明显的部分。QRS 复波检测对于时域 ECG 信号分析（特别是心率变异性分析）至关重要。它使得计算连续两个 R 峰值之间的时间间隔（RR 间隔）成为可能。因此，QRS 复波检测器是一个基于 ECG 的心脏收缩检测器。

您可以在[这里](https://en.wikipedia.org/wiki/Cardiac_cycle)和[这里](https://en.wikipedia.org/wiki/QRS_complex)了解更多关于心脏周期和 QRS 复波的信息。

本仓库包含两个版本的 Pan-Tomkins QRS 检测算法实现：
* __QRSDetectorOnline__ - 在线版本，用于实时采集的 ECG 信号中检测 QRS 复波。因此，它需要连接 ECG 设备并实时接收信号。
* __QRSDetectorOffline__ - 离线版本，用于预录制的 ECG 信号数据集（如以 _.csv_ 格式存储）中检测 QRS 复波。

__此 QRS 复波检测器的实现绝非经过认证的医疗工具，不应用于健康监测。它是为心理生理学和心理学实验目的而创建和使用的。__

## 算法

发布的 QRS 检测器模块是著名的 Pan-Tomkins QRS 检测算法的实现，该算法首次由 Jiapu Pan 和 Willis J. Tomkins 在题为_"A Real-Time QRS Detection Algorithm"_ (1985) 的论文中描述。论文的完整版本可在此处访问[这里](http://www.robots.ox.ac.uk/~gari/teaching/cdt/A3/readings/ECG/Pan+Tompkins.pdf)。

算法的直接输入是原始 ECG 信号。检测器分两个阶段处理信号：滤波和阈值处理。

首先，在滤波阶段，每个原始 ECG 测量值都使用低通和高通滤波器的级联进行滤波，这些滤波器共同形成带通滤波器。这种滤波机制确保只有与心脏活动相关的信号部分能够通过。滤波器消除了大部分可能导致误检的测量噪声。然后对带通滤波信号进行微分，以识别具有高信号变化值的信号段。然后对这些变化进行平方和积分，使它们更加明显。在处理阶段的最后一步，积分信号通过峰值检测算法进行筛选，以识别积分信号内的潜在 QRS 复波。

在下一阶段，识别出的 QRS 复波候选者通过动态设置的阈值进行分类，分为 QRS 复波或噪声峰值。阈值是实时调整的：给定时刻的阈值基于先前检测到的 QRS 和噪声峰值的信号值。动态阈值考虑了噪声水平的变化。动态阈值和复杂滤波确保了足够的检测灵敏度，同时相对较少地出现 QRS 复波误检。

重要的是，Pan 和 Tomkins 原始论文中提出的所有特性并未在此模块中实现。具体来说，我们决定不实现论文中提出的非 QRS 检测核心元素的补充机制。因此，我们没有实现以下特性：滤波数据上的基准标记、基于滤波 ECG 的另一组阈值的使用、不规则心率检测和缺失 QRS 复波检测的回溯机制。尽管缺少这些补充特性，实现 Pan 和 Tompkins 提出的核心特性使我们能够达到足够的 QRS 检测水平。

## 依赖项

此处发布的模块包含以下依赖项：
* jupyter
* matplotlib
* numpy
* pyserial
* scipy

所有依赖项都在 _requirements.py_ 文件中。

这些模块是为与 Python 3.x 一起使用而实现的。但是，它们相对容易转换为与 Python 2.x 一起工作：
- 使用以下命令导入除法模块
```
from __future__ import division
```
- 使用以下命令导入打印函数
```
from __future__ import print_function
```
- 在 QRS 检测器在线版本中从 ECG 设备读取数据时，移除 _decode()_ 函数调用，即使用
```
raw_measurement.rstrip().split(',')
```
代替
```
raw_measurement.decode().rstrip().split(',')
```

## 仓库目录结构
```
├── LICENSE
│
├── README.md          				 <- 供使用此项目的开发人员使用的高级 README。
│
├── arduino_ecg_sketch 				 <- E-health ECG 设备 Arduino 草图源代码和库。
│
├── ecg_data           				 <- 预录制的 .csv 格式 ECG 数据集。
│
├── logs               				 <- 在线和离线 QRS 检测器记录的 .csv 格式数据。
│
├── plots          	   			 <- 离线 QRS 检测器生成的图。
│
├── qrs_detector_offline_example.ipynb  	 <- 包含离线 QRS 检测器使用方法的 Jupyter 笔记本。
│
├── QRSDetectorOffline.py   			 <- 离线 QRS 检测器模块。
│
├── QRSDetectorOnline.py    			 <- 在线 QRS 检测器模块。
│
└── requirements.txt  	 			 <- 包含模块依赖项的需求文件。
```

## 安装和使用

QRS 检测器模块以两个独立版本实现：在线和离线。每个版本都有不同的应用程序和使用方法。

### 在线 QRS 检测器

在线版本设计用于与直接连接的 ECG 设备配合工作。它使用实时接收的 ECG 信号作为输入，检测 QRS 复波，并将它们输出供其他脚本使用以触发外部事件。例如，在线 QRS 检测器可用于触发视觉、听觉或触觉刺激。它已成功在 PsychoPy 中实现 (Peirce, J. W. (2009). Generating stimuli for neuroscience using PsychoPy (Peirce, J. (2009). _Generating stimuli for neuroscience using PsychoPy._ Frontiers in Neuroinformatics, 2 (January), 1–8. [http://doi.org/10.3389/neuro.11.010.2008](http://doi.org/10.3389/neuro.11.010.2008)) 并经过测试，用于研究心脏感知（内感受）能力，特别是在 Schandry 的心跳跟踪任务 (Schandry, R. (1981). _Heart beat perception and emotional experience._ Psychophysiology, 18(4), 483–488. [http://doi.org/10.1111/j.1469-8986.1981.tb02486.x](http://doi.org/10.1111/j.1469-8986.1981.tb02486.x)) 和基于 Schandry 和 Weitkunat 提出的心跳检测训练 (Schandry, R., & Weitkunat, R. (1990). _Enhancement of heartbeat-related brain potentials through cardiac awareness training._ The International Journal of Neuroscience, 53(2-4), 243–53. [http://dx.doi.org/10.3109/00207459008986611](http://dx.doi.org/10.3109/00207459008986611)) 中。

QRS 检测器模块的在线版本已实现为与基于 Arduino 的 e-Health Sensor Platform V2.0 ECG 设备配合工作。您可以在此处了解更多有关此设备的信息[这里](https://www.cooking-hacks.com/documentation/tutorials/ehealth-biometric-sensor-platform-arduino-raspberry-pi-medical#step4_2)。

本仓库中还提供了 Arduino e-Health ECG 设备草图。ECG 信号采集的采样率设置为 250 (Hz) 样本每秒。测量值以字符串格式 _"timestamp,measurement"_ 实时发送，必须由 QRS 检测器模块解析。

要使用在线 QRS 检测器模块，加载了 Arduino 草图的 ECG 设备必须连接到 USB 端口。然后，使用连接 ECG 设备的端口名称和设置的测量波特率初始化 QRS 复波检测器对象。无需进一步校准或配置。在线 QRS 检测器在初始化后立即开始检测。

以下是运行在线 QRS 检测器的示例代码：

```
from QRSDetectorOnline import QRSDetectorOnline

qrs_detector = QRSDetectorOnline(port="/dev/cu.usbmodem14311", baud_rate="115200")
```

如果您想在后台使用在线 QRS 检测器，同时在上面层运行其他进程（例如，显示视觉刺激或播放由检测到的 QRS 复波触发的音调），我们建议使用 Python 多进程机制。多进程提供本地和远程并发，通过使用子进程而不是线程有效地避免了全局解释器锁。以下是实现此目的的多种方法之一的示例代码：

```
from QRSDetectorOnline import QRSDetectorOnline

qrs_detector_process = subprocess.Popen(["python", "QRSDetectorOnline.py", "/dev/cu.usbmodem14311"], shell=False)
```

尽管在线 QRS 检测器是为与基于 Arduino 的 e-Health Sensor Platform ECG 设备配合使用而实现的，具有 250 Hz 采样率和指定的数据格式，但可以轻松修改以与任何其他 ECG 设备、采样率或数据格式一起使用。有关更多信息，请查看"自定义"部分。


### 离线 QRS 检测器

检测器的离线版本适用于存储在 _.csv_ 格式中的 ECG 测量数据集。这些可以加载到检测器中以执行 QRS 检测。

离线 QRS 检测器加载数据、分析数据，并以与在线版本相同的方式检测 QRS 复波，但它使用整个现有数据集，而不是直接来自 ECG 设备的实时测量。

此模块旨在在实时基于 QRS 的事件触发不重要但需要更复杂数据分析、可视化和检测信息时，用于 ECG 测量中的离线 QRS 检测。它也可以简单地用作调试工具，以验证在线版本是否按预期工作，或检查 QRS 检测器中间数据处理阶段的行为。

QRS 检测器模块的离线版本实现为与任何可以从文件加载的 ECG 测量数据类型一起工作。与在线版本不同，它不是为与某个特定设备一起工作而设计的。离线 QRS 检测器期望以 _.csv_ 格式存储的 _"timestamp,measurement"_ 数据格式，并且调整为以 250 Hz 采样率获取的测量数据。数据格式和采样率都可以轻松修改。有关更多信息，请查看"自定义"部分。

离线 QRS 检测器需要使用 ECG 测量文件的路径进行初始化。QRS 检测器将加载数据集、分析测量值并检测 QRS 复波。它输出一个标记了检测到的 QRS 复波的检测日志文件。在文件中，检测到的 QRS 复波在 'qrs_detected' 日志数据列中用 '1' 标记标记。此外，离线 QRS 检测器将检测结果内部存储为离线 QRS 检测器对象的 ecg_data_detected 属性。可选地，它会生成包含所有中间信号处理步骤的图并将其保存到 *.csv* 文件。

以下是显示如何运行 QRS 检测器模块离线版本的示例代码：

```
from QRSDetectorOnline import QRSDetectorOnline

qrs_detector = QRSDetectorOffline(ecg_data_path="ecg_data/ecg_data_1.csv", verbose=True, log_data=True, plot_data=True, show_plot=False)
```

检查 _qrs_detector_offline_example.ipynb_ Jupyter 笔记本，了解离线 QRS 检测器的使用示例，包含生成的图和日志。

## 自定义

### QRSDetectorOnline
在线 QRS 检测器模块可以轻松修改以与其他 ECG 设备、不同的采样率和不同的数据格式一起工作：

- QRSDetectorOnline 适用于从 ECG 设备实时接收的信号。数据以 _"timestamp,measurement"_ 格式字符串发送到模块。如果期望不同的 ECG 数据格式，*process_measurement()* 函数需要进行一些更改以启用正确的数据解析。即使只有测量值而没有相应的时间戳，在线 QRS 检测器也能工作。

- QRSDetectorOnline 默认调整为 250 Hz 采样率；但是，这可以通过根据所需的 signal_frequency 更改七个配置属性（在代码中标记）来自定义。 例如，要将信号采样率从 250 更改为 125 样本每秒，只需将所有参数除以 2：将 *signal_frequency* 设置为 125，*number_of_samples_stored* 设置为 100 样本，*integration_window* 设置为 8 样本，*findpeaks_spacing* 设置为 25 样本，*detection_window* 设置为 20 样本，*refractory_period* 设置为 60 样本。

### QRSDetectorOffline
离线 QRS 检测器是硬件独立的，因为它使用 ECG 测量数据集而不是实时获取的 ECG 信号。因此，唯一需要调整的参数是采样率和数据格式：

- QRSDetectorOffline 默认调整为 250 样本每秒的采样率。它可以通过根据所需的 *signal_frequency* 更改 4 个配置属性（在代码中标记）来自定义。例如，要将信号采样率从 250 更改为 125 样本每秒，将所有参数除以 2：将 *signal_frequency* 值设置为 125，*integration_window* 设置为 8 样本，*findpeaks_spacing* 设置为 25 样本，*refractory_period* 设置为 60 样本。

- QRSDetectorOffline 使用加载的 ECG 测量数据集。数据期望为 csv 格式，每行采用 *"timestamp,measurement"* 格式。如果期望不同的 ECG 数据格式，需要在加载数据集的 *load_ecg_data()* 函数或在处理测量值的 *detect_peaks()* 函数中进行更改：
```
ecg_measurements = ecg_data[:, 1]
```
即使只有测量值而没有时间戳，算法也能正常工作。

## 引用信息
如果您在研究项目中使用这些模块，请考虑引用它：

[![DOI](https://zenodo.org/badge/55516257.svg)](https://zenodo.org/badge/latestdoi/55516257)

如果您在任何其他项目中使用这些模块，请参考 MIT 开源许可证。

## 致谢
以下模块和仓库是项目"Relationship between interoceptive awareness and metacognitive abilities in near-threshold visual perception"的一部分创建的，该项目由波兰国家科学中心 PRELUDIUM 7 项目编号 2014/13/N/HS6/02963 支持。特别感谢 Michael Timberlake 的校对工作。