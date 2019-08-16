#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# 预测Titanic乘客逃生
import numpy as np # 数组常用库
import pandas as pd # 读入csv常用库
from patsy import dmatrices # 可根据离散变量自动生成哑变量
from sklearn.linear_model import LogisticRegression # sk-learn库Logistic Regression模型
from sklearn.model_selection import train_test_split, cross_val_score # sk-learn库训练与测试
from sklearn import metrics # 生成各项测试指标库
import matplotlib.pyplot as plt # 画图常用库

# In[ ]:
# > 从../input/train.csv读入数据
data = pd.read_csv('titanic/train.csv')
data.shape

# In[ ]:
# 删除不需要的列以及含有NA值的行
data = data.drop(['Ticket', 'Cabin'], axis = 1)
data = data.dropna()
data.shape


# In[ ]:
# 观察逃生人数与未逃生人数





# 观察女性逃生人数

# In[ ]:





# 观察男性逃生人数

# In[ ]:





# 观察非低等舱逃生情况

# In[ ]:


highclass = data.Survived[data.Pclass != 3].value_counts().sort_index()
highclass.plot(kind='bar',label='Highclass', color='red', alpha=0.6)
plt.show()


# 观察低等舱逃生情况

# In[ ]:





# In[ ]:
data.head(5)
# dmatrices将数据中的离散变量变成哑变量，并指明用Pclass, Sex, Embarked来预测Survived
y, X = dmatrices('Survived~C(Pclass)+C(Sex)+Age+C(Embarked)', data=data, return_type='dataframe')

# In[ ]:
model = LogisticRegression()
model.fit(X, y)
model.score(X, y)

# In[ ]:
# 输出空模型的正确率：空模型预测所有人都未逃生
# null model, our model score need to be better than null model.
1 - y.mean()

# In[ ]:
# 观察模型系数，即每种因素对于预测逃生的重要性
pd.DataFrame(list(zip(X.columns, np.transpose(model.coef_))))

# %%
# 对测试数据../input/test.csv生成预测，将结果写入./my_prediction.csv
test_data = pd.read_csv('titanic/test.csv')
# data preprocessing for dmatrices
test_data['Survived'] = 1
test_data.loc[np.isnan(test_data['Age']), 'Age'] = np.mean(test_data['Age'])
test_data.isna().any()

# %%
ytest, Xtest = dmatrices('Survived~ C(Pclass) + C(Sex) + Age + C(Embarked)', data=data, return_type='dataframe')

# %%
pred = model.predict(Xtest).astype(int)
solution = pd.DataFrame(list(zip(test_data['PassengerId'], pred)), columns=['PassengerId', 'Survivied'])
solution

# %%
solution.to_csv('titanic/my_prediciton.csv', index=False)
