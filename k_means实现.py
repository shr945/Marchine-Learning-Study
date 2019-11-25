import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.datasets.samples_generator import make_blobs


# 加载数据
def loadDataSet():
    # X为样本特征，Y为样本簇类别， 共100个样本，每个样本3个特征，共3个簇，簇中心在 [0,0],[1,1], [2,2]， 簇方差分别为[0.3, 0.3, 0.3]
    dataSet, y = make_blobs(n_samples=100, n_features=2,
                            centers=[[0, 0], [1, 1], [2, 2]],
                            cluster_std=[0.3, 0.3, 0.3],
                            random_state=9)
    return dataSet, y


# 欧氏距离计算
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))  # 计算欧氏距离


# 为给定数据集构建一个包含K个随机质心的集合
def randCent(dataSet, k):
    m, n = dataSet.shape
    # 质心大小k*2
    centroids = np.zeros((k, n))
    # 生成k个在0-m范围内不重复的整数，保证各质心不同
    rl = random.sample(range(0, m), k)
    for i in range(k):
        index = rl[i]  #
        centroids[i, :] = dataSet[index, :]
    print(centroids)
    return centroids


# k均值聚类
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # 行的数目
    # 第一列存样本属于哪一簇
    # 第二列存样本的到簇的中心点的误差
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterChange = True # 用来记录质心是否改变

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterChange:
        clusterChange = False

        # 遍历所有的样本（行数）
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # 遍历所有的质心
            # 第2步 找出最近的质心
            for j in range(k):
                # 计算该样本到质心的欧式距离
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterChange = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("KMeans实现完成!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment, y):
    m, n = dataSet.shape
    if n != 2:
        print("数据不是二维的")
        return 1

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k值太大了")
        return 1

    # 绘制样本，属于不同簇的样本属于不同颜色
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=y)
    plt.show()
    # 绘制所有的样本 样本是原点
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心 质心是方块
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()


if '__main__' == __name__:
    k = 3
    dataSet, y = loadDataSet()
    centroids, clusterAssment = KMeans(dataSet, k)
    showCluster(dataSet, k, centroids, clusterAssment, y)
