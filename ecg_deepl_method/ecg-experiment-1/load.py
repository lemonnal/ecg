import wfdb
import pywt
import numpy as np
import torch
from scipy import signal

# MIT-BIH
# 导联方式 ['MLII', 'V5', 'V2', 'V1', 'V4']
root1 = '/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/'

# CT-T
# 导联方式 ['MLIII', 'V1', 'V2', 'MLI', 'D3', 'V3', 'V4', 'V5']
root2 = '/home/yogsothoth/DataSet/european-st-t-database-1.0.0/'

# CU
# 导联方式 ['ECG']
root3 = '/home/yogsothoth/DataSet/cu-ventricular-tachyarrhythmia-database-1.0.0/'

# NST
# 导联方式 ['MLII', 'V1']
root4 = '/home/yogsothoth/DataSet/mit-bih-noise-stress-test-database-1.0.0/'

# 测试集在数据集中所占的比例
# kinds = ['N', 'A', 'V', 'L', 'R']
kinds = ['/', 'j', 'S', 'V', 'R', '~', '+', 'J', 'Q', 'a', 'F', 'x', 'L', 'A', 'E', 'N', 'e', '|', '"', 'f']
pre_sample, past_sample = 150, 200

def bandpass_filter(data, fs=360, lowcut=0.5, highcut=40.0, order=5):
    """
    带通滤波器函数

    Args:
        data: 输入信号
        fs: 采样频率 (Hz)
        lowcut: 低频截止频率 (Hz)
        highcut: 高频截止频率 (Hz)
        order: 滤波器阶数

    Returns:
        滤波后的信号
    """
    # 设计巴特沃斯带通滤波器
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # 创建带通滤波器
    b, a = signal.butter(order, [low, high], btype='band')

    # 应用零相位滤波（forward-backward filtering）
    filtered_data = signal.filtfilt(b, a, data)

    return filtered_data


def wavelet_filter(data, wavelet="db5", level=9):
    """小波去噪函数"""
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet=wavelet, level=level)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换，获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata, coeffs


# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data, dataset_type="MIT-BIH"):
    try:
        if dataset_type == "MIT-BIH":
            ecgClassSet = kinds
            # 读取心电数据记录
            print("正在读取 " + number + " 号心电数据...")
            # 读取MLII导联的数据
            record = wfdb.rdrecord(root1 + number, channel_names=['MLII'])
            data = record.p_signal.flatten()
            rdata, coeffs = wavelet_filter(data=data)
            rdata = bandpass_filter(data=rdata)

            # 获取心电数据记录中R波的位置和对应的标签
            annotation = wfdb.rdann(root1 + number, 'atr')
            Rlocation = annotation.sample
            Rclass = annotation.symbol
            # 去掉前后的不稳定数据
            start = 5
            end = 5
            i = start
            j = len(annotation.symbol) - end
            # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
            # X_data在R波前后截取长度为pre_sample + past_sample的数据点
            # Y_data将NAVLR按顺序转换为01234
            while i < j:
                try:
                    # Rclass[i] 是标签
                    label = ecgClassSet.index(Rclass[i])
                    # 基于经验值，基于R峰向前取pre_sample个点，向后取past_sample个点
                    x_train = rdata[Rlocation[i] - pre_sample:Rlocation[i] + past_sample]
                    X_data.append(x_train)
                    Y_data.append(label)
                    i += 1
                except ValueError:
                    i += 1
            return
        elif dataset_type == "ST-T":
            ecgClassSet = kinds
            # 读取心电数据记录
            print("正在读取 " + number + " 号心电数据...")
            # 读取MLIII导联的数据
            record = wfdb.rdrecord(root2 + number, channel_names=['MLIII'])
            data = record.p_signal.flatten()
            rdata, coeffs = wavelet_filter(data=data)
            rdata = bandpass_filter(data=rdata)

            # 获取心电数据记录中R波的位置和对应的标签
            annotation = wfdb.rdann(root2 + number, 'atr')
            Rlocation = annotation.sample
            Rclass = annotation.symbol
            # 去掉前后的不稳定数据
            start = 10
            end = 5
            i = start
            j = len(annotation.symbol) - end
            # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
            # X_data在R波前后截取长度为pre_sample + past_sample的数据点
            # Y_data将NAVLR按顺序转换为01234
            while i < j:
                try:
                    # Rclass[i] 是标签
                    label = ecgClassSet.index(Rclass[i])
                    # 基于经验值，基于R峰向前取pre_sample个点，向后取past_sample个点
                    x_train = rdata[Rlocation[i] - pre_sample:Rlocation[i] + past_sample]
                    X_data.append(x_train)
                    Y_data.append(label)
                    i += 1
                except ValueError:
                    i += 1
            return
        elif dataset_type == "CU":
            ecgClassSet = ['N', 'A', 'V', 'L', 'R']
            # 读取心电数据记录
            print("正在读取 " + number + " 号心电数据...")
            # 读取 ECG 数据
            record = wfdb.rdrecord(root3 + number, channel_names=['ECG'])
            data = record.p_signal.flatten()
            rdata, coeffs = wavelet_filter(data=data)
            rdata = bandpass_filter(data=rdata)

            # 获取心电数据记录中R波的位置和对应的标签
            annotation = wfdb.rdann(root3 + number, 'atr')
            Rlocation = annotation.sample
            Rclass = annotation.symbol
            # 去掉前后的不稳定数据
            start = 10
            end = 5
            i = start
            j = len(annotation.symbol) - end
            # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
            # X_data在R波前后截取长度为pre_sample + past_sample的数据点
            # Y_data将NAVLR按顺序转换为01234
            while i < j:
                try:
                    # Rclass[i] 是标签
                    label = ecgClassSet.index(Rclass[i])
                    # 基于经验值，基于R峰向前取pre_sample个点，向后取past_sample个点
                    x_train = rdata[Rlocation[i] - pre_sample:Rlocation[i] + past_sample]
                    X_data.append(x_train)
                    Y_data.append(label)
                    i += 1
                except ValueError:
                    i += 1
            return
        elif dataset_type == "NST":
            ecgClassSet = ['N', 'A', 'V', 'L', 'R']
            # 读取心电数据记录
            print("正在读取 " + number + " 号心电数据...")
            # 读取 ECG 数据
            record = wfdb.rdrecord(root4 + number, channel_names=['MLII'])
            data = record.p_signal.flatten()
            rdata, coeffs = wavelet_filter(data=data)
            rdata = bandpass_filter(data=rdata)

            # 获取心电数据记录中R波的位置和对应的标签
            annotation = wfdb.rdann(root4 + number, 'atr')
            Rlocation = annotation.sample
            Rclass = annotation.symbol
            # 去掉前后的不稳定数据
            start = 10
            end = 5
            i = start
            j = len(annotation.symbol) - end
            # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
            # X_data在R波前后截取长度为pre_sample + past_sample的数据点
            # Y_data将NAVLR按顺序转换为01234
            while i < j:
                try:
                    # Rclass[i] 是标签
                    label = ecgClassSet.index(Rclass[i])
                    # 基于经验值，基于R峰向前取pre_sample个点，向后取past_sample个点
                    x_train = rdata[Rlocation[i] - pre_sample:Rlocation[i] + past_sample]
                    X_data.append(x_train)
                    Y_data.append(label)
                    i += 1
                except ValueError:
                    i += 1
            return
    except Exception as e:
        print(e)
        return


def loadData(test_ratio=0.2, dataset_type="MIT-BIH"):
    if dataset_type == "MIT-BIH":
        numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                     '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                     '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                     '231', '232', '233', '234']
    elif dataset_type == "ST-T":
        numberSet = ['e0103', 'e0104', 'e0105', 'e0106', 'e0107', 'e0108', 'e0110', 'e0111', 'e0112', 'e0113', 'e0114', 'e0115',
         'e0116', 'e0118', 'e0119', 'e0121', 'e0122', 'e0123', 'e0124', 'e0125', 'e0126', 'e0127', 'e0129', 'e0133',
         'e0136', 'e0139', 'e0147', 'e0148', 'e0151', 'e0154', 'e0155', 'e0159', 'e0161', 'e0162', 'e0163', 'e0166',
         'e0170', 'e0202', 'e0203', 'e0204', 'e0205', 'e0206', 'e0207', 'e0208', 'e0210', 'e0211', 'e0212', 'e0213',
         'e0302', 'e0303', 'e0304', 'e0305', 'e0306', 'e0403', 'e0404', 'e0405', 'e0406', 'e0408', 'e0409', 'e0410',
         'e0411', 'e0413', 'e0415', 'e0417', 'e0418', 'e0501', 'e0509', 'e0515', 'e0601', 'e0602', 'e0603', 'e0604',
         'e0605', 'e0606', 'e0607', 'e0609', 'e0610', 'e0611', 'e0612', 'e0613', 'e0614', 'e0615', 'e0704', 'e0801',
         'e0808', 'e0817', 'e0818', 'e1301', 'e1302', 'e1304']
    elif dataset_type == "CU":
        numberSet = ['cu01', 'cu02', 'cu03', 'cu04', 'cu05', 'cu06', 'cu07', 'cu08', 'cu09', 'cu10',
                     'cu11', 'cu12', 'cu13', 'cu14', 'cu15', 'cu16', 'cu17', 'cu18', 'cu19', 'cu20',
                     'cu21', 'cu22', 'cu23', 'cu24', 'cu25', 'cu26', 'cu27', 'cu28', 'cu29', 'cu30',
                     'cu31', 'cu32', 'cu33', 'cu34', 'cu35']
    elif dataset_type == "NST":
        numberSet = ['118e00', '118e06', '118e12', '118e18', '118e24', '118e_6',
                     '119e00', '119e06', '119e12', '119e18', '119e24', '119e_6']

    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet, dataset_type=dataset_type)

    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, pre_sample + past_sample)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)
    # 数据集及其标签集
    X = train_ds[:, :pre_sample + past_sample].reshape(-1, pre_sample + past_sample, 1)
    Y = train_ds[:, pre_sample + past_sample].astype(int)  # 确保标签为整数类型
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(test_ratio * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]

    # 转换为PyTorch张量，调整数据维度为 (batch_size, channels, length)
    X_train = torch.FloatTensor(X_train).permute(0, 2, 1)  # (N, 1, pre_sample + past_sample)
    Y_train = torch.LongTensor(Y_train)
    X_test = torch.FloatTensor(X_test).permute(0, 2, 1)   # (N, 1, pre_sample + past_sample)
    Y_test = torch.LongTensor(Y_test)

    return X_train, Y_train, X_test, Y_test
