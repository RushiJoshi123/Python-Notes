#!/usr/bin/env python
# coding: utf-8

# # Ch-4 Model Training and machine learning 

# In[104]:


# machine learning
# polynomail regression
# k nearest neighbours 
# decision tree
# np.where
# pd.getdummies 


# In[105]:


# types of machine learning : 
#     1. supervised 
#     2. unsupervised 
#     3. reinforcement 
# regresssion: numerical data


# In[106]:


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


# In[107]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[108]:


df = pd.read_csv('iris.csv')
df.head(5)


# In[109]:


df.isna().sum()


# In[110]:


df.duplicated()


# In[111]:


df.drop('Id',axis=1,inplace=True)


# In[112]:


df.duplicated().sum()


# In[113]:


df.drop_duplicates(inplace=True)


# In[114]:


df.duplicated().sum()


# In[115]:


corr_m = df.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(corr_m,annot=True)
plt.tight_layout()


# In[116]:


sns.pairplot(df)


# In[117]:


from sklearn.model_selection import train_test_split
x= df[['PetalLengthCm','SepalLengthCm','SepalWidthCm']]
y = df[['PetalWidthCm']]


# In[118]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_test.shape)
print(y_train.shape)


# In[119]:


print(x_train)
print(y_train)


# In[120]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[121]:


model.coef_


# In[122]:


model.intercept()


# In[ ]:


2*model.coef_+model.intercept_


# In[ ]:


y_pred = model.predict(x_test)
y_pred


# In[ ]:


y1 = model.predict(np.array([[1.5,1.8,2]]))
y1


# In[ ]:





# In[ ]:


# MSE, MAE, RMSE, R2_Score : 
from sklearn.metrics import mean_squared_error, r2_score
print((mean_squared_error(y_test,y_pred))*100 , '%')


# In[ ]:


df = pd.read_csv('auto-mpg.csv')
df


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=3)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_test.shape)
print(y_train.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
x= df[['horsepower']]
y = df[['weight']]


# In[ ]:


# MSE, MAE, RMSE, R2_Score : 
from sklearn.metrics import mean_squared_error, r2_score
print((mean_squared_error(y_test,y_pred))*100 , '%')


# In[ ]:


df.horsepower.unique()


# In[124]:


df['horsepower'] = df[df['horsepower']!='?']
df['horsepower']= df['horsepower'].astype('float64')
df.horsepower.unique()


# In[ ]:


df['car name'].unique()
df= df.drop(['car name'],axis=1)


# In[ ]:


x = df.drop('horsepower',axis=1)


# In[125]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=5)
print(x.shape)
print(x_train.shape)
print(x_test.shape)
print(y.shape)
print(y_test.shape)
print(y_train.shape)


# In[126]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)


# In[127]:


y_pred = model.predict(x_test)
y_pred


# In[128]:


from sklearn.metrics import mean_squared_error, r2_score
print((mean_squared_error(y_test,y_pred)))


# In[ ]:


model.intercept_


# In[129]:


2*model.coef_+model.intercept_


# In[ ]:


y_pred


# In[ ]:




