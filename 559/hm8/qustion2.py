#!/usr/bin/env python
# coding: utf-8

# In[7]:


import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pandas as pd
import matplotlib.pyplot as plt

import plotSVMBoundaries as py
#from sklearn.utils.validation import column_or_1d


# In[8]:


#training model
def train_model(model, feature_set, label_set):
    skf = StratifiedKFold(n_splits=5,shuffle=True)
    cnt = []
    for train_index, dev_index in skf.split(feature_set, label_set):
    #   print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_dev = feature2_train[train_index], feature2_train[dev_index]
        y_train, y_dev = label_train[train_index], label_train[dev_index]
        model.fit(X_train, y_train)
        acc = model.score(X_dev, y_dev)
        cnt.append(acc)

    return cnt


# In[9]:


#Q2(a)
feature_train = np.genfromtxt("feature_train.csv", delimiter = ",")
label_train = np.genfromtxt("label_train.csv", delimiter = ",")
feature_test  = np.genfromtxt("feature_test.csv",  delimiter = ",")
label_test = np.genfromtxt("label_test.csv", delimiter = ",")

feature2_train = feature_train[:,0:2]
#train_data = np.column_stack((feature2_train,label_train))
# print(np.shape(feature2_train))


# s = model.predict(feature2_train)      #predict
model = SVC(C=1, gamma=1, kernel='rbf')
# skf = StratifiedKFold(n_splits=5,shuffle=True)

cnt = train_model(model, feature2_train, label_train)
print(np.shape(cnt))
print("accuracy is ")
print(np.mean(cnt, axis=0))


# In[10]:


#Q2(b)
##model selection
# skf = StratifiedKFold(n_splits=5, shuffle=True)
para_gamma = np.logspace(-3, 3, num=50)
para_C = np.logspace(-3, 3, num=50)

ACC = np.zeros((50,50))  #store average accuracies p on the validation set
DEV = np.zeros((50,50))     #store the estimated standard deviation of accuracies

res = 1
for i,gamma in enumerate(para_gamma):
    for j,c in enumerate(para_C):
        model2 = SVC(C = c, gamma = gamma, kernel = 'rbf')
        acc = train_model(model2, feature2_train, label_train) 
        ACC[i,j] = np.mean(acc)
        DEV[i,j] = np.std(acc)
                            
plt.imshow(ACC, interpolation = 'nearest', cmap=plt.cm.Blues)
plt.colorbar()  

i, j = np.argwhere(ACC == np.max(ACC))[0]
print("the best gamma is " + str(para_gamma[i]))
print("the best C is " + str(para_C[j]))
print("the best accuracy is ")
print(ACC[i,j])               


# In[11]:


##(C)
ACC = []
DEV = []
res = []
for k in range(9):
    list = []
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    para_gamma = np.logspace(-3, 3, num=50)
    para_C = np.logspace(-3, 3, num=50)

    acc = np.zeros((50,50))  #store average accuracies p on the validation set
    dev = np.zeros((50,50))     #store the estimated standard deviation of accuracies

    for i,gamma in enumerate(para_gamma):
        for j,c in enumerate(para_C):
            model2 = SVC(C = c, gamma = gamma, kernel = 'rbf')
            accuracy = train_model(model2, feature2_train, label_train) 
            acc[i,j] = np.mean(accuracy)
            dev[i,j] = np.std(accuracy)

    #plt.imshow(ACC, interpolation = 'nearest', cmap=plt.cm.Blues)
    #plt.colorbar()  

    i, j = np.argwhere(acc == np.max(acc))[0]
    res.append((para_gamma[i], para_C[j]))
    
    print(acc[i,j])
    ACC.append(acc[i,j])
    DEV.append(dev[i,j])

print()
print(res)
print()
i = np.argwhere(ACC == np.max(ACC))[0]    
print("accuracy is " + str(ACC[int(i)]))
print("standard deviation is " + str(DEV[int(i)]))
print("i,j pair is " + str(res[int(i)]))


##(d) on all training set
gam_best = res[int(i)][0]
c_best = res[int(i)][1]
model = SVC(C=c_best, gamma=gam_best, kernel='rbf')
model.fit(feature_train, label_train)
s = model.predict(feature_test)      #predict  

acc = accuracy_score(label_test, s)   
print("training on test set the accuracy is " + str(acc))


# 

# In[12]:


print(np.array(res))


# In[ ]:




