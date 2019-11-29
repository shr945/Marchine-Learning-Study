# coding=utf-8
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
#有本地数据




def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))
#数据提取
def load():
    x= pd.read_csv("thyroid.txt",header=-1,usecols=[i for i in range(21)])
    y= pd.read_csv("thyroid.txt",header=-1,usecols = [21])
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    x = ss.fit_transform(x)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    return x_train, y_train, x_test, y_test


def train(digits, labels, maxIter = 100, alpha = 0.1):
    weights = np.random.randn(3, 21) * 0.01
    for iter in range(maxIter):
        for i in range(len(digits)):
            x = digits[i].reshape(-1, 1)
            y = np.zeros((3, 1))
            y[labels[i]-1] = 1
            y_ = softmax(np.dot(weights, x))
            weights -= alpha * (np.dot((y_ - y), x.T))
    return weights

def predict(digit,w):   #预测函数
    return np.argmax(np.dot(w, digit))+1   #返回softmax中概率最大的值


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = load();
    w = train(x_train, y_train, maxIter=150) #训练
    accuracy = 0
    N = len(x_test) #总共多少测试样本
    for i in range(N):
        digit = x_test[i]   #每个测试样本
        label = y_test[i][0]    #每个测试标签
        predict1 = predict(digit, w)  #测试结果
        if (predict1 == label):
            accuracy += 1
        print("predict:%d, actual:%d"% (predict1, label))
    print("accuracy:%.1f%%" %(accuracy / N * 100))


