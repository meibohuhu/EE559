#!/usr/bin/env python
# coding: utf-8

# In[387]:


import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


import plotSVMBoundaries as py
#from sklearn.utils.validation import column_or_1d


# In[388]:


feature_data = np.genfromtxt("train_x.csv", delimiter = ",")
print(np.shape(feature_data))

label_train = np.genfromtxt("train_y.csv", delimiter = ",")
print(np.shape(label_train))

# with open('train_x.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# train_x = np.asarray(data, dtype=np.float32)
# feature_data = copy.deepcopy(train_x)
# print(np.shape(feature_data))

# with open('train_y.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)
# # print(np.shape(data))
# train_y = np.asarray(data, dtype=np.float32)
# label_data = copy.deepcopy(train_y)
# #print(np.shape(label_data))
# label_data = column_or_1d(label_data, warn=False)    #change to (20,)
# print(np.shape(label_data))


# In[389]:


def g_x(feature_data, label_train, model):
    plt.scatter(feature_data[:,0], feature_data[:,1], c=label_train, cmap='winter');
    ax = plt.gca()
    xlim = ax.get_xlim()
    w = model.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(xlim[0], xlim[1])
    
    # w.x + b = 0
    yy = a * xx - model.intercept_[0] / w[1]
    plt.plot(xx, yy)
    # w.x + b = 1
    yy = a * xx - (model.intercept_[0] - 1) / w[1]
    plt.plot(xx, yy, 'k--')
    # w.x + b = -1
    yy = a * xx - (model.intercept_[0] + 1) / w[1]
    plt.plot(xx, yy, 'k--')
    return w


# In[390]:


model = SVC(C=1, kernel='linear')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
print("support vector for c=1 is ")
print(sv)
py.plotSVMBoundaries(feature_data, label_data, model, sv)

#g(x)
w = g_x(feature_data, label_train, model)
print(w)


# In[391]:


model = SVC(C=100, kernel='linear')
model.fit(feature_data, label_data)
s = model.predict(feature_data)      #predict  

acc = accuracy_score(label_data, s)   
print("accuracy is " + str(acc))

sv = model.support_vectors_
print("support vector for c=100 is ")
print(sv)


py.plotSVMBoundaries(feature_data, label_data, model, sv)
#g(x)
wv = g_x(feature_data, label_train, model)
print("weight vector is " + str(wv))
w0 = model.intercept_[0]
print("w0 is " + str(w0))


# In[392]:


list = []
list.append((1,2))
list.append((2,3))
print(list)


# In[393]:


#(c)
x1 = np.array(sv[0])
wwv = np.array(wv)
y1 = w0 + np.dot(wv,x1)
print("g(x1) is " + str(y1))

x2 = np.array(sv[1])
y2 = w0 + np.dot(wv,x2)
print("g(x2) is " + str(y2))

x3 = np.array(sv[2])
y3 = w0 + np.dot(wv,x3)
print("g(x3) is " + str(y3))


# #### 
