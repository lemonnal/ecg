import wfdb
import numpy as np
import matplotlib.pyplot as plt

folder = 'mit-bih-arrhythmia-database-1.0.0/'
for num in range(100, 234):
    try:
        num = str(num)
        # print("Start Print .hea")
        # records = wfdb.rdheader(folder + num)
        # for record in records.__dict__.keys():
        #     print(record, ":", records.__dict__[record])
        # print()


        # print("Start Print .dat")
        records = wfdb.rdrecord(folder + num)
        # for record in records.__dict__.keys():
        #     print(record, ":", records.__dict__[record])
        # print()
        # 计算显示长度（显示前10秒的数据）
        fs = records.fs  # 采样频率
        display_samples = int(fs * 10)  # 10秒的数据

        # 提取两个导联的完整信号
        mills, v5s = [], []
        for rcs in records.__dict__["p_signal"]:
            mills.append(rcs[0])
            v5s.append(rcs[1])

        # 转换为numpy数组便于处理
        mills = np.array(mills)
        v5s = np.array(v5s)

        # # 创建上下两个子图
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # # 上图：MLII导联（红色）
        # ax1.plot(mills[:display_samples], 'r-', linewidth=1)
        # ax1.set_title(f'MLII Lead - First {display_samples//fs} seconds')
        # ax1.set_ylabel('Amplitude (mV)')
        # ax1.grid(True, alpha=0.3)
        # ax1.tick_params(axis='y')

        # # 下图：V5导联（蓝色）
        # ax2.plot(v5s[:display_samples], 'b-', linewidth=1)
        # ax2.set_title(f'V5 Lead - First {display_samples//fs} seconds')
        # ax2.set_xlabel('Sample Points')
        # ax2.set_ylabel('Amplitude (mV)')
        # ax2.grid(True, alpha=0.3)
        # ax2.tick_params(axis='y')

        # plt.tight_layout()
        # plt.show()

        # print("Start Print .atr")
        annotations = wfdb.rdann(folder + num, 'atr')
        # for annotation in annotations.__dict__.keys():
        #     print(annotation, ":", annotations.__dict__[annotation])
        #     try:
        #         print(len(annotations.__dict__[annotation]))
        #     except:
        #         print("None")
        #         pass

        # 在第78行pass之后添加
        from collections import Counter

        print("=== Symbol 统计 ===")
        symbol_counts = Counter(annotations.symbol)

        # 打印统计结果
        for symbol, count in sorted(symbol_counts.items()):
            print(f"{symbol}: {count}")

        print(f"总注释数: {len(annotations.symbol)}")

        # 在第74行后添加以下代码
        print("=== 保存为C语言格式 ===")

        # 纯文本格式（每行：MLII值 V1值）
        c_format_filename = f'mit-bih-dataset/ecg_{num}.txt'
        with open(c_format_filename, 'w') as f:
            for ml, v1 in zip(mills, v5s):
                f.write(f"{ml:.6f} {v1:.6f}\n")

        print(f"已保存到: {c_format_filename}")
    except Exception as e:
        print(e)
