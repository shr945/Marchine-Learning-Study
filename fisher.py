import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification

X1, Y1 = make_classification(n_samples=500, n_features=2, n_redundant=0, n_classes=2,
                               n_informative=1, n_clusters_per_class=1, class_sep=0.5, random_state=10)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X1, Y1, test_size = 0.25, random_state = 0)

plt.scatter(X1[:, 0], X1[:, 1], marker='o', c=Y1)
plt.show()
from sklearn import datasets, model_selection,discriminant_analysis

lda = discriminant_analysis.LinearDiscriminantAnalysis()
lda.fit(X_train, Y_train)

print('Score: %.2f' % lda.score(X_test, Y_test))  # 测试集
print('Score: %.2f' % lda.score(X_train, Y_train))  # 训练集
