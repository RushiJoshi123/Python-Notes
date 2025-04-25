#!/usr/bin/env python
# coding: utf-8

# # Ch-4 Model Training and machine learning 

# In[ ]:


# machine learning
# polynomail regression
# k nearest neighbours 
# decision tree
# np.where
# pd.getdummies 


# In[ ]:


# types of machine learning : 
#     1. supervised 
#     2. unsupervised 
#     3. reinforcement 
# regresssion: numerical data


# In[ ]:


# Flow of the making machine learning model : 
# problem destination
# data
# pre-processing data
# eature engine
# data - train - test 
# one mla 
# make a model 
# evaluate 
# deploy 


# In[ ]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[103]:


df = pd.read_csv('iris.csv')
df.head(5)


# In[ ]:


df.isna().sum()


# In[ ]:


df.duplicated()


# In[ ]:


df.drop('Id',axis=1,inplace=True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


df.drop_duplicates(inplace=True)


# In[ ]:


df.duplicated().sum()


# In[ ]:


corr_m = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(corr_m,annot=True)
plt.tight_layout()


# In[ ]:


sns.pairplot(df)


# In[83]:


from sklearn.model_selection import train_test_split
x= df[['PetalLengthCm']]
y = df[['PetalWidthCm']]


# In[85]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_test.shape)
print(y_train.shape)


# In[ ]:


print(x_train)
print(y_train)


# In[86]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[87]:


model.coef_


# In[ ]:


model.intercept()


# In[88]:


2*model.coef_+model.intercept_


# In[90]:


y_pred = model.predict(x_test)
y_pred


# In[94]:


plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred)
plt.show()


# In[113]:


y1 = model.predict(np.array([[2]]))
y1


# In[118]:


# MSE, MAE, RMSE, R2_Score : 
from sklearn.metrics import mean_squared_error, r2_score
print((mean_squared_error(y_test,y_pred))*100 , '%')


# In[ ]:




