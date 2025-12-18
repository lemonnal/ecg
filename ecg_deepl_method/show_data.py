import wfdb
import numpy as np

# MIT-BIH
# 导联方式 ['MLII', 'V1', 'V2', 'V4', 'V5']
# 标注 ['/', 'j', 'S', 'V', 'R', '~', '+', 'J', 'Q', 'a', 'F', 'x', 'L', 'A', 'E', 'N', 'e', '|', '"', 'f']
# ['N', 'A', 'V', 'L', 'R']
folder = '/home/yogsothoth/DataSet/mit-bih-arrhythmia-database-1.0.0/'
numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                     '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                     '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                     '231', '232', '233', '234']

# ST-T
# 导联方式 ['MLIII', 'V1', 'V2', 'MLI', 'D3', 'V3', 'V4', 'V5']
# folder = '/home/yogsothoth/DataSet/european-st-t-database-1.0.0/'
# numberSet = ['e0103', 'e0104', 'e0105', 'e0106', 'e0107', 'e0108', 'e0110', 'e0111', 'e0112', 'e0113', 'e0114', 'e0115',
#          'e0116', 'e0118', 'e0119', 'e0121', 'e0122', 'e0123', 'e0124', 'e0125', 'e0126', 'e0127', 'e0129', 'e0133',
#          'e0136', 'e0139', 'e0147', 'e0148', 'e0151', 'e0154', 'e0155', 'e0159', 'e0161', 'e0162', 'e0163', 'e0166',
#          'e0170', 'e0202', 'e0203', 'e0204', 'e0205', 'e0206', 'e0207', 'e0208', 'e0210', 'e0211', 'e0212', 'e0213',
#          'e0302', 'e0303', 'e0304', 'e0305', 'e0306', 'e0403', 'e0404', 'e0405', 'e0406', 'e0408', 'e0409', 'e0410',
#          'e0411', 'e0413', 'e0415', 'e0417', 'e0418', 'e0501', 'e0509', 'e0515', 'e0601', 'e0602', 'e0603', 'e0604',
#          'e0605', 'e0606', 'e0607', 'e0609', 'e0610', 'e0611', 'e0612', 'e0613', 'e0614', 'e0615', 'e0704', 'e0801',
#          'e0808', 'e0817', 'e0818', 'e1301', 'e1302', 'e1304']

# CU
# 导联方式 ['ECG']
# folder = '/home/yogsothoth/DataSet/cu-ventricular-tachyarrhythmia-database-1.0.0/'
# numberSet = ['cu01', 'cu02', 'cu03', 'cu04', 'cu05', 'cu06', 'cu07', 'cu08', 'cu09', 'cu10',
#              'cu11', 'cu12', 'cu13', 'cu14', 'cu15', 'cu16', 'cu17', 'cu18', 'cu19', 'cu20',
#              'cu21', 'cu22', 'cu23', 'cu24', 'cu25', 'cu26', 'cu27', 'cu28', 'cu29', 'cu30',
#              'cu31', 'cu32', 'cu33', 'cu34', 'cu35']

# NST
# 导联方式 ['MLII', 'V1']
# 不完整记录: 3
#   bw: 缺失 .atr
#   em: 缺失 .atr
#   ma: 缺失 .atr
# folder = '/home/yogsothoth/DataSet/mit-bih-noise-stress-test-database-1.0.0/'
# numberSet = ['118e00', '118e06', '118e12', '118e18', '118e24', '118e_6',
#              '119e00', '119e06', '119e12', '119e18', '119e24', '119e_6']

# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, data):
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据文件...")
    record = wfdb.rdrecord(folder + number)
    for key in record.__dict__:
        print(key, ':', record.__dict__[key])

    # print("正在读取 " + number + " 号心电头文件...")
    # head = wfdb.rdheader(folder + number)
    # for key in head.__dict__:
    #     print(key, ":", head.__dict__[key])
    #
    # print("正在读取 " + number + " 号心电标注文件...")
    # annotation = wfdb.rdann(folder + number, 'atr')
    # for key in annotation.__dict__:
    #     print(key, ":", annotation.__dict__[key])
    #
    # for i in annotation.__dict__["symbol"]:
    #     data.append(i)
    # return

# 加载数据集并进行预处理
def loadData():
    data= []
    for n in numberSet:
        getDataSet(n, data)
    return data

def main():
    data = loadData()
    data = list(set(data))
    print(data)

if __name__ == '__main__':
    main()