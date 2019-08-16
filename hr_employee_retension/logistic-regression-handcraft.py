#!/usr/bin/env python
# coding: utf-8

# # 手写Logistic Regression模型优化，过拟合实验，梯度检查
# In[ ]:
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

# In[ ]:
data = pd.read_csv("hr_employee_retension//HR_comma_sep.csv")

# In[ ]:
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company+Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
print(X)
print(y)
X = np.asmatrix(X)
y = np.ravel(y)
print(X)
print(y)

# In[ ]:
# 将所有列的值归一化到[0,1]区间
# for every columns
for i in range(1, X.shape[1]):
    xmin = X[:,i].min() # min value of a column
    xmax = X[:,i].max() # max value of a column
    X[:, i] = (X[:, i] - xmin) / (xmax - xmin)   # value / range

# In[ ]:
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # random initialize parameters, number of features.
error_rates = []
for T in range(200):
    prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率, this is sigmoid
    prob_y = list(zip(prob, y))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(y) # 计算损失函数的值 # cross entropy

    # calculate error rate to see if it's going down
    error_rate = 0
    for i in range(len(y)):
        if ((prob[i] > 0.5 and y[i] == 0) or (prob[i] <= 0.5 and y[i] == 1)):
            error_rate += 1;

    error_rate /= len(y)
    error_rates.append(error_rate)

    if T % 5 == 0:
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate))

    # calculate derivative of cross entropy 计算损失函数关于beta每个分量的导数
    deriv = np.zeros(X.shape[1]) # shape of beta
    for i in range(len(y)): # for every row
        deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i]) # xi * (pi - yi)
    deriv /= len(y)

    # 沿导数相反方向修改beta
    beta -= alpha * deriv


# In[ ]:
# # 过拟合实验
Xtrain, Xvali, ytrain, yvali=train_test_split(X, y, test_size=0.2, random_state=3)

# In[ ]:
np.random.seed(1)
alpha = 5 # learning rate
beta = np.random.randn(Xtrain.shape[1]) # 随机初始化参数beta
error_rates_train=[]
error_rates_vali=[]
for T in range(200):
    # train set
    prob = np.array(1. / (1 + np.exp(-np.matmul(Xtrain, beta)))).ravel()  # 根据当前beta预测离职的概率
    prob_y = list(zip(prob, ytrain))
    loss = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y]) / len(ytrain) # 计算损失函数的值

    # error rate
    error_rate = 0
    for i in range(len(ytrain)):
        if ((prob[i] > 0.5 and ytrain[i] == 0) or (prob[i] <= 0.5 and ytrain[i] == 1)):
            error_rate += 1;
    error_rate /= len(ytrain)
    error_rates_train.append(error_rate)

    # validation set
    prob_vali = np.array(1. / (1 + np.exp(-np.matmul(Xvali, beta)))).ravel()  # 根据当前beta预测离职的概率
    # prob_y_vali = list(zip(prob_vali, yvali))
    # loss_vali = -sum([np.log(p) if y == 1 else np.log(1 - p) for p, y in prob_y_vali]) / len(yvali) # 计算损失函数的值

    # error rate
    error_rate_vali = 0
    for i in range(len(yvali)):
        if ((prob_vali[i] > 0.5 and yvali[i] == 0) or (prob_vali[i] <= 0.5 and yvali[i] == 1)):
            error_rate_vali += 1
    error_rate_vali /= len(yvali)
    error_rates_vali.append(error_rate_vali)

    # print error rates
    if T % 5 ==0 :
        print('T=' + str(T) + ' loss=' + str(loss) + ' error=' + str(error_rate)+ ' error_vali=' + str(error_rate_vali))

    # 计算损失函数关于beta每个分量的导数
    deriv = np.zeros(Xtrain.shape[1])
    for i in range(len(ytrain)):
        deriv += np.asarray(Xtrain[i,:]).ravel() * (prob[i] - ytrain[i])
    deriv /= len(ytrain)
    # 沿导数相反方向修改beta
    beta -= alpha * deriv


# In[ ]:
# # 梯度检查 to check overfit
plt.plot(range(50,200), error_rates_train[50:], 'r^', range(50, 200), error_rates_vali[50:], 'bs')
plt.xlabel("Traning Round")
plt.ylabel("error rate")
plt.show()

# In[]:
np.random.seed(1)
alpha = 1  # learning rate
beta = np.random.randn(X.shape[1]) # 随机初始化参数beta

#dF/dbeta0
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率
prob_y = list(zip(prob, y))
loss = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # 计算损失函数的值
deriv = np.zeros(X.shape[1])
for i in range(len(y)):
    deriv += np.asarray(X[i,:]).ravel() * (prob[i] - y[i])
deriv /= len(y)
print('We calculated ' + str(deriv[0]))

delta = 0.0001
beta[0] += delta
prob = np.array(1. / (1 + np.exp(-np.matmul(X, beta)))).ravel()  # 根据当前beta预测离职的概率
prob_y = list(zip(prob, y))
loss2 = -sum([np.log(p) if y == 1 else np.log(1. - p) for p, y in prob_y]) / len(y) # 计算损失函数的值
shouldbe = (loss2 - loss) / delta # (F(b0+delta,b1,...,bn) - F(b0,...bn)) / delta
print('According to definition of gradient, it is ' + str(shouldbe))


# In[ ]:
print(cross_val_score(Lo))
