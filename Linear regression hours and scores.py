#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import seaborn as sns


# In[2]:


ds=pd.read_csv('http://bit.ly/w-data')


# In[4]:


ds.head()


# In[5]:


sns.jointplot(x='Hours',y='Scores',data=ds)


# In[26]:


X=ds.iloc[:, :-1].values
y=ds.iloc[:, 1].values  
print(X,y)


# In[27]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 


# In[29]:


lr= LinearRegression()  
lr.fit(X_train, y_train) 


# In[32]:


print(lr.intercept_,lr.coef_)


# In[33]:


lr.predict(X)


# In[34]:


sns.jointplot(x='Hours',y='Scores',data=ds,kind='reg')


# In[35]:


print(X_test)


# In[36]:


y_pred=lr.predict(X_test)


# In[37]:


ds = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
ds 


# In[46]:


hours = np.array(9.25)
d=hours.reshape(-1, 1)
print(d)


# In[47]:


own_pred=lr.predict(d)
print("Hours={}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[48]:


from sklearn import metrics  
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test, y_pred)) 


# In[ ]:





# In[ ]:




