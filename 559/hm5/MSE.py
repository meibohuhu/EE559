#!/usr/bin/env python
# coding: utf-8

# In[76]:


import sklearn 
import csv
import numpy as np 
import math
import matplotlib.pyplot as plt
import copy
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LinearRegression 


# In[77]:


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
#standard
data_fea_norm = copy.deepcopy(data_fea_unnorm)
sc_X = StandardScaler()
sc_X.fit(data_fea_norm)
data_fea_norm = sc_X.transform(data_fea_norm)     #	Fit to data, then transform it.
print('After standardization:') 
print(data_fea_norm)
print(np.shape(data_fea_norm))

##take first two columns without normalization
feature_2_unnorm = data_unnorm[:,0:2]

#take first tow columns with normalization
feature_2 = data_fea_norm[:,0:2]
print(np.shape(feature_2))
class_labels = data_unnorm[:,13]
print(np.shape(class_labels))

#print(class_labels)


# In[78]:


###test set
with open('wine_test.csv', newline='') as f:
    reader_t = csv.reader(f)
    data_t = list(reader_t)

data_array_t = np.asarray(data_t, dtype=np.float32)
data_unnorm_t = copy.deepcopy(data_array_t)

data_fea_unnorm_t = data_unnorm_t[:,0:13]
print(np.shape(data_fea_unnorm_t))
mean_unnorm_t = np.mean(data_fea_unnorm_t, axis = 0)
std_unnorm_t = np.std(data_fea_unnorm_t, axis = 0)
print(np.shape(mean_unnorm_t))
print("mean of unnormalized data points is: ") 
print(mean_unnorm_t)
print("standard deviation of unnormalized data points is: ") 
print(std_unnorm_t)

data_fea_norm_t = copy.deepcopy(data_fea_unnorm_t)

data_fea_norm_t = sc_X.transform(data_fea_norm_t)     #	Fit to data, then transform it.
print('After standardization:') 
print(data_fea_norm_t)
print(np.shape(data_fea_norm_t))


feature_2_unnorm_t = data_unnorm_t[:,0:2]
#take first tow columns
feature_2_t = data_fea_norm_t[:,0:2]
print(np.shape(feature_2_t))
class_labels_t = data_unnorm_t[:,13]
print(np.shape(class_labels_t))
feature_all_t = data_fea_norm_t[:,0:13]
#print(class_labels_t)


# In[79]:


#Define Class MSE Binary
class MSE_binary(LinearRegression):
    def __init__(self):
        print("Calling newly created MSE_binary function...") 
        super(MSE_binary, self).__init__()
    def predict(self, X): 
        thr = 0.5
        y = self._decision_function(X) 
        y_binary = np.zeros(len(X))
        for i in range(len(X)):
            if(y[i] <= thr): 
                y_binary[i] = 0
            else:
                y_binary[i] = 1 
        return y_binary


# In[80]:


#Apply to training data without normalization 
binary_model = MSE_binary()
MSE = OneVsRestClassifier(binary_model) 
###
MSE.fit(feature_2_unnorm, class_labels)
print('Final Weights of training set on 2 features:')
weight_train_2 = MSE.coef_
print(weight_train_2)
train_label_2 = MSE.predict(feature_2_unnorm) 
#print(train_label_2)
print(MSE.score(feature_2_unnorm, class_labels))

###testing set
train_label_2t = MSE.predict(feature_2_unnorm_t) 
#print(train_label_2t)
print()
print('two features without normalization is ' + str(MSE.score(feature_2_unnorm_t, class_labels_t)))


# In[81]:


#Apply to training data for all features without normalization 
binary_model = MSE_binary()
MSE1 = OneVsRestClassifier(binary_model) 
MSE1.fit(data_fea_unnorm, class_labels)
print('Final Weights of training set on all features:')
weight_train_all = MSE1.coef_
print(weight_train_all)
train_label_all = MSE1.predict(data_fea_unnorm) 
#print(train_label_all)
print('accuracy is ' + str(MSE1.score(data_fea_unnorm, class_labels)))
###for testing 
print()
train_label_allt = MSE1.predict(data_fea_unnorm_t) 
#print(train_label_allt)
print('all features without normalization accuracy is ' + str(MSE1.score(data_fea_unnorm_t, class_labels_t)))


# In[82]:


#Apply to training data with normalization 
binary_model = MSE_binary()
MSE2 = OneVsRestClassifier(binary_model) 
###
MSE2.fit(feature_2, class_labels)
print('Final Weights of training set on 2 features:')
weight_train_norm2 = MSE2.coef_
print(weight_train_norm2)
train_label_2 = MSE2.predict(feature_2) 
#print(train_label_2)
print(MSE2.score(feature_2, class_labels))

###testing set
train_label_norm2t = MSE2.predict(feature_2_t) 
#print(train_label_norm2t)
print('test accuracy with normalization is ' + str(MSE2.score(feature_2_t, class_labels_t)))


# In[83]:


#Apply to training data with normalization 
binary_model = MSE_binary()
MSE3 = OneVsRestClassifier(binary_model) 
###
MSE3.fit(data_fea_norm, class_labels)
print('Final Weights of training set on 2 features:')
weight_train_all_normt = MSE3.coef_
print(weight_train_all_normt)
train_label_all_normt = MSE3.predict(data_fea_norm) 
#print(train_label_all_normt)
print(MSE3.score(data_fea_norm, class_labels))


###testing set
train_label_all_normt = MSE3.predict(data_fea_norm_t) 
#print(train_label_all_normt)
print()
print('test accuracy with normalization is ' + str(MSE3.score(data_fea_norm_t, class_labels_t)))

