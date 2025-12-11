import wfdb
import pywt
import seaborn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

root = '../mit-bih-arrhythmia-database-1.0.0/'
# 测试集在数据集中所占的比例
RATIO = 0.2
kinds = ['N', 'A', 'V', 'L', 'R']

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)
    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata

# 读取心电数据和对应标签,并对数据进行小波去噪
def getDataSet(number, X_data, Y_data):
    ecgClassSet = ['N', 'A', 'V', 'L', 'R']
    # 读取心电数据记录
    print("正在读取 " + number + " 号心电数据...")
    # 读取MLII导联的数据
    record = wfdb.rdrecord(root + number, channel_names=['MLII'])
    data = record.p_signal.flatten()
    rdata = denoise(data=data)
    # 获取心电数据记录中R波的位置和对应的标签
    annotation = wfdb.rdann(root + number, 'atr')
    Rlocation = annotation.sample
    Rclass = annotation.symbol
    # 去掉前后的不稳定数据
    start = 10
    end = 5
    i = start
    j = len(annotation.symbol) - end
    # 因为只选择NAVLR五种心电类型,所以要选出该条记录中所需要的那些带有特定标签的数据,舍弃其余标签的点
    # X_data在R波前后截取长度为300的数据点
    # Y_data将NAVLR按顺序转换为01234
    while i < j:
        try:
            # Rclass[i] 是标签
            label = ecgClassSet.index(Rclass[i])
            # 基于经验值，基于R峰向前取100个点，向后取200个点
            x_train = rdata[Rlocation[i] - 100:Rlocation[i] + 200]
            X_data.append(x_train)
            Y_data.append(label)
            i += 1
        except ValueError:
            i += 1
    return

# 加载数据集并进行预处理
def loadData():
    numberSet = ['100', '101', '103', '105', '106', '107', '108', '109', '111', '112', '113', '114', '115',
                 '116', '117', '119', '121', '122', '123', '124', '200', '201', '202', '203', '205', '208',
                 '210', '212', '213', '214', '215', '217', '219', '220', '221', '222', '223', '228', '230',
                 '231', '232', '233', '234']
    dataSet = []
    lableSet = []
    for n in numberSet:
        getDataSet(n, dataSet, lableSet)
    # 转numpy数组,打乱顺序
    dataSet = np.array(dataSet).reshape(-1, 300)
    lableSet = np.array(lableSet).reshape(-1, 1)
    train_ds = np.hstack((dataSet, lableSet))
    np.random.shuffle(train_ds)
    # 数据集及其标签集
    X = train_ds[:, :300].reshape(-1, 300, 1)
    Y = train_ds[:, 300].astype(int)  # 确保标签为整数类型
    # 测试集及其标签集
    shuffle_index = np.random.permutation(len(X))
    # 设定测试集的大小 RATIO是测试集在数据集中所占的比例
    test_length = int(RATIO * len(shuffle_index))
    # 测试集的长度
    test_index = shuffle_index[:test_length]
    # 训练集的长度
    train_index = shuffle_index[test_length:]
    X_test, Y_test = X[test_index], Y[test_index]
    X_train, Y_train = X[train_index], Y[train_index]
    return X_train, Y_train, X_test, Y_test


# 构建CNN模型
def buildModel_1():
    newModel = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(300, 1)),
        # 第一个卷积层, 4 个 21x1 卷积核
        tf.keras.layers.Conv1D(filters=4, kernel_size=21, strides=1, padding='same', activation='tanh'),
        # 第一个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # 第二个卷积层, 16 个 23x1 卷积核
        tf.keras.layers.Conv1D(filters=16, kernel_size=23, strides=1, padding='same', activation='relu'),
        # 第二个池化层, 最大池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.MaxPool1D(pool_size=3, strides=2, padding='same'),
        # 第三个卷积层, 32 个 25x1 卷积核
        tf.keras.layers.Conv1D(filters=32, kernel_size=25, strides=1, padding='same', activation='tanh'),
        # 第三个池化层, 平均池化,4 个 3x1 卷积核, 步长为 2
        tf.keras.layers.AvgPool1D(pool_size=3, strides=2, padding='same'),
        # 第四个卷积层, 64 个 27x1 卷积核
        tf.keras.layers.Conv1D(filters=64, kernel_size=27, strides=1, padding='same', activation='relu'),
        # 打平层,方便全连接层处理'
        tf.keras.layers.Flatten(),
        # 全连接层,128 个节点 转换成128个节点
        tf.keras.layers.Dense(128, activation='relu'),
        # Dropout层,dropout = 0.2
        tf.keras.layers.Dropout(rate=0.2),
        # 全连接层,5 个节点
        tf.keras.layers.Dense(5, activation='softmax')
    ])
    return newModel

def buildModel_2():
    # 使用函数式API构建残差网络
    inputs = tf.keras.Input(shape=(300, 1))

    # 第一部分：input->conv->bn_relu
    x = tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=2, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)

    # 第二部分：进行一个残差连接（conv->bn->relu-dropout->conv）+(maxpool）
    # 残差块的shortcut
    shortcut = x
    # 主路径
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(x)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(0.2)(y)
    y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(y)
    # 残差连接
    x = tf.keras.layers.add([shortcut, y])
    x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    # 第三部分：进行残差连接((bn->relu->dropout->conv->bn->relu->dropout->conv)+(maxpool))*15次
    for i in range(15):
        shortcut = x
        # 第一个卷积块
        y = tf.keras.layers.BatchNormalization()(x)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(y)

        # 第二个卷积块
        y = tf.keras.layers.BatchNormalization()(y)
        y = tf.keras.layers.ReLU()(y)
        y = tf.keras.layers.Dropout(0.2)(y)
        y = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='same')(y)

        # 残差连接
        x = tf.keras.layers.add([shortcut, y])
        x = tf.keras.layers.MaxPool1D(pool_size=2, strides=2, padding='same')(x)

    # 第四部分：bn->relu->dense->softmax
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(5, activation='softmax')(x)

    # 创建模型
    newModel = tf.keras.Model(inputs=inputs, outputs=outputs)
    return newModel

# def plotHeatMap(Y_test, Y_pred):
#     con_mat = confusion_matrix(Y_test, Y_pred)
#     # 绘图
#     plt.figure(figsize=(4, 5))
#     seaborn.heatmap(con_mat, annot=True, fmt='.20g', cmap='Blues')
#     plt.ylim(0, 5)
#     plt.xlabel('Predicted labels')
#     plt.ylabel('True labels')
#     plt.show()
#
# def loadModel(model_path='ecg_cnn_model.h5'):
#     """
#     加载已保存的模型
#     Args:
#         model_path: 模型文件路径
#     Returns:
#         加载的Keras模型
#     """
#     try:
#         model = tf.keras.models.load_model(model_path)
#         print(f"成功加载模型: {model_path}")
#         return model
#     except Exception as e:
#         print(f"加载模型失败: {e}")
#         return None
#
# def loadTrainingHistory(history_path='training_history.json'):
#     """
#     加载训练历史
#     Args:
#         history_path: 训练历史文件路径
#     Returns:
#         训练历史字典
#     """
#     try:
#         import json
#         with open(history_path, 'r') as f:
#             history = json.load(f)
#         print(f"成功加载训练历史: {history_path}")
#         return history
#     except Exception as e:
#         print(f"加载训练历史失败: {e}")
#         return None


def main():
    # X_train,Y_train为所有的数据集和标签集
    # X_test,Y_test为拆分的测试集和标签集
    X_train, Y_train, X_test, Y_test = loadData()
    print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    model = buildModel_1()
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy']
                  # metrics: 列表，包含评估模型在训练和测试时的性能的指标，典型用法是metrics=[‘accuracy’]。
                  )
    model.summary()

    # 训练与验证
    history = model.fit(X_train, Y_train, epochs=30, batch_size=128, validation_split=RATIO)

    # 保存模型
    model.save('ecg_wu_model.h5')
    print("模型已保存为 'ecg_wu_model.h5'")

    # 保存训练历史
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    print("训练历史已保存为 'training_history.json'")

    print(f"\n开始预测后20%的测试数据 ({len(X_test)} 个样本)...")
    print("=" * 60)

    # 批量预测并输出详细结果
    correct_predictions = 0
    total_predictions = 0

    for i in tqdm(range(len(X_test))):
        # 获取单个样本进行预测
        ecg_data = X_test[i:i+1]
        predictions = model.predict(ecg_data, verbose=0)

        # 获取预测结果
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        class_probabilities = predictions[0]
        true_class = int(Y_test[i])

        # 判断预测是否正确
        is_correct = predicted_class == true_class
        if is_correct:
            correct_predictions += 1
        total_predictions += 1

        # # 输出预测结果
        # print(f"样本 {i+1}: 真实类别={kinds[true_class]}, "
        #       f"预测类别={kinds[predicted_class]}, "
        #       f"置信度={confidence:.2%}, "
        #       f"预测{'正确' if is_correct else '错误'}")
        #
        # # 输出各类别概率
        # print(f"  各类别概率:")
        # for j, (kind, prob) in enumerate(zip(kinds, class_probabilities)):
        #     print(f"    {kind}: {prob:.2%}")
        # print()

    # 计算并输出最终准确率
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        print("=" * 60)
        print(f"预测完成!")
        print(f"预测准确率: {accuracy:.2%} ({correct_predictions}/{total_predictions})")

if __name__ == '__main__':
    main()