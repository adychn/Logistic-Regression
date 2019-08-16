#!/usr/bin/env python
# coding: utf-8
# HR Employee Retension Rate, predicting an employee likely to leave or not.
# In[ ]:
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库

# In[ ]:
# 从../input/HR_comma_sep.csv文件中读入数据
data = pd.read_csv("hr_employee_retension/HR_comma_sep.csv")
data

# In[ ]:
data.describe()
data.dtypes
# 观察离职人数与工资分布的关系

# In[ ]:
# 观察离职比例与工资分布的关系
pd.crosstab(data.salary, data.left)
pd.crosstab(data.salary, data.left).plot(kind='bar')
plt.show()

# In[ ]:
q = pd.crosstab(data.salary, data.left)
print(q)
print(q.sum(1))
print(q.div(q.sum(1), axis=0)) # axis=0 is to get which value in q.sum(1) to divide to.
q.div(q.sum(1), axis = 0).plot(kind='bar', stacked = True)
plt.show()

# In[ ]:
# 观察员工满意度的分布图(histogram)
data[data.left==0].satisfaction_level.hist()
plt.show()

# In[ ]:
data[data.left==1].satisfaction_level.hist()
plt.show()


# In[ ]:
# dmatrices将数据中的离散变量变成哑变量，并指明用satisfaction_level, last_evaluation, ... 来预测left
# 然后重命名列的名字
# C(xxx) categorical feature
# 'y~X' like R language
y, X = dmatrices('left~satisfaction_level+last_evaluation+number_project+average_montly_hours+time_spend_company\
                  +Work_accident+promotion_last_5years+C(sales)+C(salary)', data, return_type='dataframe')
X = X.rename(columns = {
    'C(sales)[T.RandD]': 'Department: Random',
    'C(sales)[T.accounting]': 'Department: Accounting',
    'C(sales)[T.hr]': 'Department: HR',
    'C(sales)[T.management]': 'Department: Management',
    'C(sales)[T.marketing]': 'Department: Marketing',
    'C(sales)[T.product_mng]': 'Department: Product_Management',
    'C(sales)[T.sales]': 'Department: Sales',
    'C(sales)[T.support]': 'Department: Support',
    'C(sales)[T.technical]': 'Department: Technical',
    'C(salary)[T.low]': 'Salary: Low',
    'C(salary)[T.medium]': 'Salary: Medium'})
y = np.ravel(y) #将y变成np的一维数组

# In[ ]:
# zip(a,b)可将a的每一个元素和b里对应位置的元素组成一对
# 用X和y训练模型，然后输出X中每一项自变量对于y的影响
model = LogisticRegression()
model.fit(X, y)
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

# In[ ]:
# model.score为准确率(0到1之间)
print(model.score(X,y))

# In[ ]:
#一个高工资HR，对公司满意度0.5, 上次评审0.7分，做过4个项目，每月平均工作160小时，在公司呆了3年，过去5年没有被晋升，没有工伤
model.predict_proba([[1,0,0,1,0,0,0,0,0,0,0,0, 0.5, 0.7, 4.0, 160, 3.0, 0, 0]])

# In[ ]:
pred = model.predict(X)
(abs(pred-y)).sum() / len(y) # 錯誤率 不相等才是1

# In[ ]:
# 生成7:3的训练测试集
Xtrain, Xtest, ytrain, ytest=train_test_split(X, y, test_size=0.3, random_state=0)

model2 = LogisticRegression(C=10000)
model2.fit(Xtrain, ytrain)
pred = model2.predict(Xtest)

# In[ ]:
# 观察实际离职/未离职被预测成为离职/未离职的数目
print(metrics.accuracy_score(ytest, pred))
print(metrics.confusion_matrix(ytest, pred))
#        prediction
#
#
#actual
#
#
#
# classification_report会输出每一类对应的precision, recall
print(metrics.classification_report(ytest, pred))

# In[ ]:
# 10份的交叉验证Cross Validation
print(cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10).mean())
print(cross_val_score(LogisticRegression(C=10), X, y, scoring='accuracy', cv=10).mean())
print(cross_val_score(LogisticRegression(C=1000), X, y, scoring='accuracy', cv=10).mean()) # best
print(cross_val_score(LogisticRegression(C=10000), X, y, scoring='accuracy', cv=10).mean())
