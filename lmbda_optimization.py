#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pytan
import pandas as pd
import numpy as np
import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import MEM
from tqdm import tqdm_notebook as tqdm

X_train, y_train, X_test, y_test = datasets.get_dataset('adults')


lmbda_range = [2**x for x in range(-10, 10)]
lmbda_score = np.zeros(len(lmbda_range))
j=0
for x in tqdm(lmbda_range):
    for i in range(5):
        mem = MEM.MEM(lmbda=x)
        X_train_lmbda, X_test_lmbda, y_train_lmbda, y_test_lmbda = train_test_split(X_train, y_train, 
                                                                                    test_size=0.3, shuffle=True)
        mem.fit(X_train_lmbda, y_train_lmbda)
        lmbda_score[j] += accuracy_score(y_test_lmbda, mem.predict(X_test_lmbda))
    j+=1
    print(j)
lmbda_score = lmbda_score/5

lmbda_opt = lmbda_range[np.argmax(lmbda_score)]


# In[ ]:




