import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

from sklearn.datasets import make_multilabel_classification

def LDA(X, y):
    # 根据y标签的0,1类别分个类
    X1 = np.array([X[i] for i in range(len(X)) if y[i] == 0])
    X2 = np.array([X[i] for i in range(len(X)) if y[i] == 1])
    # 求中心点
    mju1 = np.mean(X1, axis=0)
    mju2 = np.mean(X2, axis=0)
    # 类内离散度矩阵
    cov1 = np.dot((X1 - mju1).T, (X1 - mju1))
    cov2 = np.dot((X2 - mju2).T, (X2 - mju2))
    # 总类离散度矩阵
    Sw = cov1 + cov2
    print(Sw)
    print((mju1 - mju2).reshape((len(mju1), 1)))
    # 计算w
    w = np.dot(np.mat(Sw).I, (mju1 - mju2).reshape((len(mju1), 1)))  # 计算w
    print("w::",w)
    return w, X1, X2


def plot_LDA(w, c_1, c_2):
    plt.scatter(c_1[:, 0], c_1[:, 1], c='yellow')
    plt.scatter(c_2[:, 0], c_2[:, 1], c='green')
    plt.show()

    plt.scatter(c_1[:, 0], c_1[:, 1], c='yellow')
    plt.scatter(c_2[:, 0], c_2[:, 1], c='green')

    line_x = np.arange(min(np.min(c_1[:, 0]), np.min(c_2[:, 0])),
                       max(np.max(c_1[:, 0]), np.max(c_2[:, 0])),
                       step=1)
    w = np.array(w)
    line_y = - (w[1] / w[0]) * line_x
    plt.plot(line_x, line_y)
    plt.show()

    X1_new = func(c_1, w)
    X2_new = func(c_2, w)
    y1_new = [1 for i in range(len(c_1))]
    y2_new = [2 for i in range(len(c_2))]
    plt.plot(X1_new, y1_new, 'b*')
    plt.plot(X2_new, y2_new, 'ro')
    plt.show()



def func(x, w):

    return np.dot((x), w)


if '__main__' == __name__:
    X, y = make_multilabel_classification(n_samples=200, n_features=2,
                                          n_labels=1, n_classes=1,
                                          random_state=2)  # 设置随机数种子，保证每次产生相同的数据。
    # 调用函数获取w
    w, c_1, c_2 = LDA(X, y)
    # 画出fisher分类图像
    plot_LDA(w, c_1,c_2)
