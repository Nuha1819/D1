#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


df=pd.read_csv(r'C:\Users\User\Desktop\D1\007 heart-disease.csv')


# In[6]:


df.shape


# In[7]:


df.sample(5)


# In[8]:


x=df.drop(columns=['target'])
y=df['target']


# In[9]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)


# In[22]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)
model.score(x_test,y_test)


# In[17]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler=StandardScaler()
x_train_trf=scaler.fit_transform(x_train)
x_test_trf=scaler.transform(x_test)
accuracy_score(y_test,
              LogisticRegression()
              .fit(x_train_trf,y_train)
              .predict(x_test_trf))


# In[ ]:


#from sklearn.ensemble import RandomForestClassifier
#accuracy_score(y_test,RandomForestClassifier().fit(x_train,y_train).predict(x_test))


# In[23]:


import pickle
pickle.dump(model,open('model.pk1','wb'))


# In[ ]:




