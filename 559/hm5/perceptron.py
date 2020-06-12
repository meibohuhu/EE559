#!/usr/bin/env python
# coding: utf-8

# In[248]:


import sklearn 
import csv
import numpy as np 
import math
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score


# In[249]:


with open('wine_train.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

#print(np.shape(data))

data_array = np.asarray(data, dtype=np.float32)
data_unnorm = copy.deepcopy(data_array)

data_fea_unnorm = data_unnorm[:,0:13]
print(np.shape(data_fea_unnorm))
mean_unnorm = np.mean(data_fea_unnorm, axis = 0)
std_unnorm = np.std(data_fea_unnorm, axis = 0)
print(np.shape(mean_unnorm))
print("mean of unnormalized data points is: ") 
print(mean_unnorm)
print("standard deviation of unnormalized data points is: ") 
print(std_unnorm)

data_fea_norm = copy.deepcopy(data_fea_unnorm)
sc_X = StandardScaler()
sc_X.fit(data_fea_norm)  #the normalizing factors should be calculated from the training data only
data_fea_norm = sc_X.transform(data_fea_norm)     #	Fit to data, then transform it.
print('After standardization:') 
print(data_fea_norm)
#print(np.shape(data_fea_norm))


# In[250]:


#take first tow columns
feature_2 = data_fea_norm[:,0:2]
print(np.shape(feature_2))
class_labels = data_unnorm[:,13]
print(np.shape(class_labels))
#print(class_labels)


# In[251]:


def calcul_accuracy(real_label, pre_label):
    cnt = 0
    for i in range(len(real_label)):
        if real_label[i] == pre_label[i]:
            cnt += 1
    return cnt/len(real_label)

#perceptron on the first two features
perceptron = Perceptron(max_iter=1000, tol=0.0001, random_state = None) 
perceptron.fit(feature_2, class_labels) #fit(X, y[, coef_init, intercept_init, ...])	Fit linear model with Stochastic Gradient Descent. 
print("feature_2's final Weight:")
#print(perceptron.intercept_)             #不懂
final_wei1 = perceptron.coef_
print(final_wei1)

label_train_pred1 = perceptron.predict(feature_2)      #Predict class labels for samples in X.
#print(label_train_pred1)
mean_ar = perceptron.score(feature_2, class_labels)
#print(mean_ar)
mean_accuracy1 = calcul_accuracy(class_labels, label_train_pred1)    #Returns the mean accuracy on the given test data and labels.
print("mean accuracy is " + str(mean_accuracy1))

#print(calcul_accuracy(label_train_pred1, class_labels))


# In[252]:


###all features
perceptron.fit(data_fea_norm, class_labels)
print("feature_all's final Weight:")

final_wei2 = perceptron.coef_
print(final_wei2)
label_train_pred2 = perceptron.predict(data_fea_norm)      #Predict class labels for samples in X.
#print(label_train_pred2)
mean_accuracy2 = calcul_accuracy(label_train_pred2, class_labels)
print("mean accuracy is " + str(mean_accuracy2))
#mean_ar2 = perceptron.score(data_fea_norm, class_labels)
print(mean_ar2)


# In[253]:


###test set
with open('wine_test.csv', newline='') as f:
    reader_t = csv.reader(f)
    data_t = list(reader_t)

data_array_t = np.asarray(data_t, dtype=np.float32)
data_unnorm_t = copy.deepcopy(data_array_t)

data_fea_unnorm_t = data_unnorm_t[:,0:13]
print(np.shape(data_fea_unnorm_t))

data_fea_norm_t = copy.deepcopy(data_fea_unnorm_t)

data_fea_norm_t = sc_X.transform(data_fea_norm_t)     #	Fit to data, then transform it using 
print('After standardization:') 
print(data_fea_norm_t)
print(np.shape(data_fea_norm_t))


# In[254]:


#take first tow columns
feature_2_t = data_fea_norm_t[:,0:2]
print(np.shape(feature_2_t))
class_labels_t = data_unnorm_t[:,13]
print(np.shape(class_labels_t))
#print(class_labels_t)

perceptron_t = Perceptron(max_iter=1000, tol=0.0001, random_state = None) 
#use weight in training data as starting weight
perceptron_t.fit(feature_2_t, class_labels_t, coef_init = final_wei1)      
print("feature_2_t's final Weight:")
print(perceptron_t.coef_)

label_train_pred1_t = perceptron_t.predict(feature_2_t)      #Predict class labels for samples in X.
#print(label_train_pred1_t)

mean_accuracy1_t = perceptron_t.score(feature_2_t, class_labels_t)    #Returns the mean accuracy on the given test data and labels.
print("mean accuracy is " + str(mean_accuracy1_t))
mean_ac3 = calcul_accuracy(label_train_pred1_t, class_labels_t)
#print(calcul_accuracy(label_train_pred1_t, class_labels_t))
#print(mean_ac3)


# In[255]:


#all_features_test = data_fea_norm_t[:,0:13]

###all features for test
perceptron_t.fit(data_fea_norm_t, class_labels_t, coef_init = final_wei2)
print("feature_all's final Weight:")
print(perceptron_t.coef_)
label_train_pred2_t = perceptron_t.predict(data_fea_norm_t)      #Predict class labels for samples in X.
#print(label_train_pred2_t)
mean_accuracy2_t = perceptron_t.score(data_fea_norm_t, class_labels_t)
print("mean accuracy is " + str(mean_accuracy2_t))
mean_ac3 = calcul_accuracy(label_train_pred2_t, class_labels_t)
print(mean_ac3)
print('interation times: ')
print(perceptron_t.n_iter_)


# In[256]:


#d
epoch = 100
Weight1 = np.zeros((3,2)) 
accu_tmp1 = 0
#perceptron1 = Perceptron(max_iter=1000, tol=0.0001, random_state = None)   
for i in range(epoch):   
    perceptron1 = Perceptron(max_iter=1000, tol=0.0001, random_state = None)   
    random_w1 = np.random.randn(3,2)
    perceptron1.fit(feature_2, class_labels, coef_init = random_w1)
    label1_train_pred = perceptron1.predict(feature_2) 
    acc_train1 = perceptron1.score(feature_2, class_labels) 
    if(acc_train1 > accu_tmp1):
        accu_tmp1 = acc_train1 
        Weight1 = perceptron1.coef_ 
        #ini_weight1 = random_w1      
"""
    random_w2 = np.random.randn(3,13)
    perceptron1.fit(data_fea_norm, class_labels, coef_init = random_w2)
    label2_train_pred = perceptron1.predict(data_fea_norm) 
    acc_train2 = perceptron1.score(data_fea_norm, class_labels) 
    if(acc_train2 > accu_tmp2):
        accu_tmp2 = acc_train2 
        Weight2 = perceptron1.coef_
        ini_weight2 = random_w2          
"""                       
print(Weight1)
print(accu_tmp1)
print(perceptron1.n_iter_)
#print(Weight2)       #two are the same
#print(accu_tmp2)
""" 
    perceptron1.fit(feature_2, class_labels, coef_init = random_w)
    label1_train_pred = perceptron1.predict(feature_2) 
    acc_train1 = perceptron1.score(feature_2, class_labels) 
    if(acc_train1 > accu_tmp1):
        accu_tmp1 = acc_train1 
        Weight1 = perceptron1.coef_ 
        ini_weight1 = random_w
"""      
    


# In[257]:


perceptron1.fit(feature_2_t, class_labels_t, coef_init=Weight1)    #use best training initiate weight

test_label1 = perceptron1.predict(feature_2_t) 
acc_test1 = perceptron1.score(feature_2_t, class_labels_t) 
print(acc_test1)

print(Weight1)
print(perceptron1.coef_)
#print(perceptron1.n_iter_)

"""
perceptron1.fit(data_fea_norm_t, class_labels_t, coef_init = ini_weight2)    #use best training initiate weight
test_label2 = perceptron1.predict(data_fea_norm_t) 
acc_test2 = perceptron1.score(data_fea_norm_t, class_labels_t) 
print(acc_test2)
print(perceptron1.coef_)
"""


# In[258]:


Weight2 = np.zeros((3,13))
#ini_weight2 = np.zeros((3,13))
accu_tmp2 = 0
perceptron2 = Perceptron(max_iter=1000, tol=0.0001, random_state = None)     ##????为什么放进循环  
for i in range(100): 
    
    random_w2 = np.random.randn(3,13)
    perceptron2.fit(data_fea_norm, class_labels, coef_init = random_w2)
    label2_train_pred = perceptron2.predict(data_fea_norm) 
    acc_train2 = perceptron2.score(data_fea_norm, class_labels) 
    if(acc_train2 > accu_tmp2):
        accu_tmp2 = acc_train2 
        Weight2 = perceptron2.coef_
    
print('all features weight is ' + str(Weight2))
print('accuracy for training data is ' + str(accu_tmp2))

perceptron2.fit(all_features_test, class_labels_t, coef_init = Weight2)    #use best training initiate weight
test_label2 = perceptron2.predict(data_fea_norm_t) 
acc_test2 = perceptron2.score(data_fea_norm_t, class_labels_t) 
print(acc_test2)
print(perceptron2.coef_)


# 
