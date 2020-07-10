import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt


def loadTrainData(trainDataFile):
    """
    加载训练数据集，并且处理成需要的格式
    """
    trainData = pd.read_csv(trainDataFile, encoding='big5')
    # 去除日期等误用数据
    trainData = trainData.iloc[:, 2:]
    # 将NR替换为0,并将其转回为numpy
    trainData[trainData == 'NR'] = 0
    # 原始的数据大小为4320*25
    trainDataArray = trainData.to_numpy().astype('float64')
    # 重组数据
    rows, columns = trainDataArray.shape
    Data = []
    for day in range(rows // 18):
        rowLow = day * 18
        rowHigh = rowLow + 18
        rowPM25 = rowLow + 9
        for columnLeft in range(15):
            columnRight = columnLeft + 9
            OneTrainData = trainDataArray[rowLow:rowHigh, columnLeft:columnRight].reshape(-1)
            label = trainDataArray[rowPM25, columnRight]
            oneSampleData = np.append(OneTrainData, label)
            Data.append(oneSampleData)
    Data = np.array(Data)
    return Data[:, :162], Data[:, 162]


def loadTestData(testDataFile):
    """
    加载测试数据
    """
    testData = pd.read_csv(testDataFile, encoding='big5', header=None)
    testData = testData.iloc[:, 2:]
    testData[testData == 'NR'] = 0
    testDataArray = testData.to_numpy().astype('float64')
    rows, column = testDataArray.shape
    Data = []
    for day in range(rows // 18):
        rowLow = day * 18
        rowHigh = rowLow + 18
        OneTestData = testDataArray[rowLow:rowHigh, :].reshape(-1)
        Data.append(OneTestData)
    Data = np.array(Data)
    return Data


def dataIter(batchSize, feature, label, shuffle=True):
    """
    将训练数据进行shuffle，按照一个batch_size的数据量返回
    """
    numExample = len(feature)
    indexs = list(range(numExample))
    if shuffle:
        random.shuffle(indexs)
    for i in range(0, numExample, batchSize):
        index = indexs[i:i + batchSize]
        yield feature[index], label[index]


def regression(feature, param):
    """
    线性回归模型
    """
    predict = np.dot(feature, param[0]) + param[1]
    return predict


def Loss(predict, label):
    """
    损失函数
    """
    return 0.5 * np.sum(np.square(predict - label)) / (len(label))


def MBGD(param, lr, features, labels):
    """
    使用SGD优化器
    """
    w = param[0]
    b = param[1]
    N = len(labels)
    h = np.dot(features, w) + b
    labels = labels.reshape(h.shape)
    dw = features.T.dot((h - labels)) / N
    db = np.sum((h - labels)) / N
    new_w = w - lr * dw
    new_b = b - lr * db
    new_param = (new_w, new_b)
    return new_param


def normalization(data):
    """
    对数据进行归一化
    """
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    result = (data - mu) / sigma
    result = np.nan_to_num(result)
    return result


def SGD(param, lr, features, y):
    """
    使用SGD优化器
    """
    w = param[0]
    b = param[1]
    w_gradient = np.array([0.0] * len(w)).reshape(w.shape)
    b_gradient = np.array([0.0])
    new_w = [0] * len(w)
    for i in range(len(features)):
        x = features[i]
        features = features.reshape(-1, 1)
        b_gradient[0] += (np.sum(w * features) + b - y)
        w_gradient[i] += x * (np.sum(w * features) + b - y)
    for j in range(len(w)):
        new_w[j] = w[j] - lr * w_gradient[j]
    new_b = b - lr * b_gradient
    return (np.array(new_w), np.array(new_b))


def save2csv(predict, path='../data/sample_submission.csv'):
    testData = pd.read_csv(path,index_col=None)
    testData['value'] = predict
    print(testData.columns)
    testData.to_csv(path, index=0)


if __name__ == '__main__':
    # 定义一些超参数
    learningRate = 1e-4
    batchSize = 128
    epoch = 1000

    # 加载数据集
    Data, Label = loadTrainData('../data/train.csv')
    TestData = loadTestData('../data/test.csv')
    rows, column = Data.shape
    # 拆分训练集和验证集
    TrainNum = rows // 5 * 4
    TrainData = Data[:TrainNum, :]
    TrainLabel = Label[:TrainNum]
    VaildData = Data[TrainNum:, :]
    VaildLabel = Label[:TrainNum]

    # 创建参数矩阵
    W = np.random.random_sample(column).reshape(column, 1)
    b = np.random.random_sample(1)
    param = (W, b)

    # 开始训练
    loss = 0
    losslist = []
    LossListVaild = []
    for e in range(epoch):
        loss = 0
        lossVaild = 0
        for i, (train, label) in enumerate(
                dataIter(batchSize=batchSize, feature=TrainData, label=TrainLabel, shuffle=True)):
            train = normalization(train)
            predict = regression(train, param)
            loss += Loss(predict, label)
            param = MBGD(param, learningRate, train, label)
            # for j, item in enumerate(train):
            #     param = SGD(param, learningRate, item, label[j])
        print("epoch: {}, train loss: {}".format(e + 1, loss / (len(TrainLabel) * 4)), end='')
        losslist.append(loss / (4 * len(TrainLabel)))
        for i, (vaild, label_vaild) in enumerate(
                dataIter(batchSize=batchSize, feature=VaildData, label=VaildLabel, shuffle=True)):
            vaild = normalization(vaild)
            predictVaild = regression(vaild, param)
            lossVaild += Loss(predictVaild, label_vaild)
        print(" vaild loss: {}".format(lossVaild / len(VaildLabel)))
        LossListVaild.append(lossVaild / len(VaildLabel))

    # print(predict)
    plt.plot(list(range(len(losslist))), losslist, label='train')
    plt.plot(list(range(len(losslist))), LossListVaild, label='vaild')
    plt.legend(loc='upper right')
    plt.show()

    # 在训练集上进行预测
    predictTest = regression(TestData, param)
    # save2csv(predictTest)
