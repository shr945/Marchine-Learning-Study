import numpy as np
import math

from sklearn import datasets
import operator
# 加载数据
def loadDataSet():
    datas = datasets.load_breast_cancer()
    X = datas.data
    y = datas.target
    return X, y

#计算多维欧式距离
def distEclud(x, y):
    distance = 0
    for i in range(len(x)):
        distance += pow((x[i] - y[i]),2)
    return math.sqrt(distance)

#得到相邻点的label
def getNeighbors(trainingSet, labelSet, testInstance, k):
    distance = []
    for i in range(len(trainingSet)):
        distance.append((labelSet[i], distEclud(trainingSet[i], testInstance)))
    distance.sort(key=operator.itemgetter(1))
    neighbors = []
    # 返回k个最近邻
    for x in range(k):
        neighbors.append(distance[x][0])

    return neighbors
#得到最多的label
def getMax(neighbors):
    a = np.zeros((2,2))
    a[1,0] = 1
    for i in neighbors:
        a[i, 1] = a[i, 1] + 1
    if(a[0, 1] > a[1, 1]):
        return a[0,0]
    else:
        return a[1,0]
#计算准确率
def getAccuracy(testSet,predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x] == predictions[x]:
            correct+=1
    return (correct/float(len(testSet)))


if __name__ == "__main__":
    X, y = loadDataSet()
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    k=6
    labels = []
    for i in range(len(X_test)):
        neighbors = getNeighbors(X_train, y_train, X_test[i],k)
        label = getMax(neighbors)
        labels.append(label)

    print(getAccuracy(y_test, labels))

