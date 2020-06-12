#!/usr/bin/env python
# coding: utf-8

# In[50]:


import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

import plotSVMBoundaries as py


# In[51]:


#deal with data
feature_data = np.genfromtxt("train_xx.csv", delimiter = ",")
print(np.shape(feature_data))

label_train = np.genfromtxt("train_yy.csv", delimiter = ",")
print(np.shape(label_train))


# In[52]:


model = SVC(C=50, gamma = "auto", kernel='rbf')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
# print("support vector for c=50 is ")
# print(sv)
py.plotSVMBoundaries(feature_data, label_data, model, sv)


# In[53]:


model = SVC(C=5000, gamma = "auto", kernel='rbf')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
# print("support vector for c=5000 is ")
# print(sv)

py.plotSVMBoundaries(feature_data, label_data, model, sv)


# In[54]:


#(E)
#gamma=10
model = SVC(gamma=10, kernel='rbf')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
# print("support vector for gamma=10 is ")
# print(sv)
py.plotSVMBoundaries(feature_data, label_data, model, sv)


# In[55]:


#gamma=50
model = SVC(gamma=50, kernel='rbf')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
# print("support vector for gamma=50 is ")
# print(sv)
py.plotSVMBoundaries(feature_data, label_data, model, sv)


# In[56]:


#gamma=500
model = SVC(gamma=500, kernel='rbf')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
py.plotSVMBoundaries(feature_data, label_data, model, sv)

