#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df = pd.read_csv('car data.csv')
df


# In[3]:


df.isna().sum()


# In[4]:


df.duplicated().sum()


# In[5]:


df.drop_duplicates(inplace=True)


# In[6]:


df.duplicated().sum()


# In[10]:





# In[ ]:





# In[7]:


df.drop('Car_Name',axis=1,inplace=True)


# In[8]:


df=pd.get_dummies(df,drop_first=True)
df


# In[ ]:





# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = df['Selling_Price']
x = df.drop('Selling_Price',axis=1)
mse = []
r2=[]

for i in range(1,51):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=i)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    mse.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))

plt.plot(range(1,51),mse)
plt.show()
plt.plot(range(1,51),r2)
plt.show()
print(min(mse),mse.index(min(mse)))
print(max(r2),r2.index(max(r2)))


# In[53]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

y = df['Selling_Price']
x = df.drop('Selling_Price',axis=1)
mse = []
r2=[]

for i in [0.5,0.4,0.3,0.2,0.1]:
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=i,random_state=30)

    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    
    mse.append(mean_squared_error(y_test,y_pred))
    r2.append(r2_score(y_test,y_pred))

plt.plot(range(1,6),mse)
plt.show()
plt.plot(range(1,6),r2)
plt.show()
print(min(mse),mse.index(min(mse)))
print(max(r2),r2.index(max(r2)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




