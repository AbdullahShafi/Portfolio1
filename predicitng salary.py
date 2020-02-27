#!/usr/bin/env python
# coding: utf-8

# In[17]:



get_ipython().system('pip install word2number')
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn import linear_model
from word2number import w2n


# In[ ]:





# In[18]:


df= pd.read_csv(r"C:\Users\shafi\Downloads\hiring.csv")
df


# In[19]:


df.experience=df.experience.fillna("zero")
df


# In[20]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[21]:


import math
test_score = math.floor(df['test_score(out of 10)'].mean())
test_score


# In[22]:


df['test_score(out of 10)'] = df['test_score(out of 10)'].fillna(test_score)
df


# In[25]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[26]:


reg.predict([[2,6,9]])


# In[27]:


reg.predict([[13,10,10]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




