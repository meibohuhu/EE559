#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
import copy


# In[19]:


train_set = np.genfromtxt("D_train.csv", delimiter = ",")
test_set  = np.genfromtxt("D_test.csv",  delimiter = ",")
print("The col is " + str(len(train_set[0,:])))
# print(len(test_set))

train_list = train_set[:, 3:]
class_list = train_set[:, 1]
id_list = train_set[:, 2]

train_data = np.column_stack((train_list, class_list))

print(np.shape(train_data))

train_data = train_data.tolist()
del(train_data[0])
print("the row of train_data is " + str(len(train_data)))
test_data = test_set.tolist()


# In[20]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    return math.sqrt(variance)


# In[21]:


class_list1 = class_list.tolist()
del(class_list1[0])
# print(len(class_list1))

id_list1 = id_list.tolist()
del(id_list1[0])
print(len(id_list1))

#index classification
total_index = []
i_0 = id_list1[0]
tmp_list = []
tmp_list.append(0)
for i in range(1,len(id_list1)):
    if id_list1[i] != i_0:
        total_index.append(tmp_list)
        tmp_list = []
    tmp_list.append(i)
    i_0 = id_list1[i]
    
total_index.append(tmp_list)

print(np.shape(total_index))
print(total_index[0])


# In[22]:


def feature_extraction(numbers):

    x_fea = []
    y_fea = []
    z_fea = []

    #delete class number
    mynewlist = [s for s in numbers if not math.isnan(s)]
    del(mynewlist[-1])
#     print(mynewlist)

    # get rid of null data points
    for i in range(len(mynewlist)):  
        if i%3 == 0:
            x_fea.append(mynewlist[i])
        if i%3 == 1:
            y_fea.append(mynewlist[i])
        if i%3 == 2:
            z_fea.append(mynewlist[i])

    length = (int)((len(mynewlist))/3)
    x_mean, y_mean, z_mean = mean(x_fea), mean(y_fea), mean(z_fea)

    x_std, y_std, z_std = stdev(x_fea), stdev(y_fea), stdev(z_fea)

    min_x, min_y, min_z = min(x_fea), min(y_fea), min(z_fea)

    max_x, max_y, max_z = max(x_fea), max(y_fea), max(z_fea)

    # print(length)
    # print(x_fea)
    # print(y_fea)

    return x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, length

    
# x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, length = feature_extraction(train_data[0])
# print(x_mean)

# num_mark = []
# x_mean_list = []
# x_std_list = []
# x_min_list = []
# x_max_list = []
# y_mean_list = []
# y_std_list = []
# y_min_list = []
# y_max_list = []
# z_mean_list = []
# z_std_list = []
# z_min_list = []
# z_max_list = []


#length, x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, class_index
features_data_points = []
index = 0


for li in train_data:
    tmp_list = []
    x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, length = feature_extraction(li)
    tmp_list.append(length)
    tmp_list.append(x_mean)
    tmp_list.append(y_mean)
    tmp_list.append(z_mean)
    tmp_list.append(x_std)
    tmp_list.append(y_std)
    tmp_list.append(z_std)
    tmp_list.append(min_x)
    tmp_list.append(min_y)
    tmp_list.append(min_z)
    tmp_list.append(max_x)
    tmp_list.append(max_y)
    tmp_list.append(max_z)
        
    tmp_list.append((int)(class_list1[index]))
    index += 1
    features_data_points.append(tmp_list)
    
print(features_data_points[1])
#13 features + class index
print(np.shape(features_data_points))
    
#     x_mean_list.append(x_mean)
#     x_std_list.append(x_std)
#     x_min_list.append(min_x)
#     x_max_list.append(max_x)
#     y_mean_list.append(y_mean)
#     y_std_list.append(y_std)
#     y_min_list.append(min_y)
#     y_max_list.append(max_y)
#     z_mean_list.append(z_mean)
#     z_std_list.append(z_std)
#     z_min_list.append(min_z)
#     z_max_list.append(max_z)
#     num_mark.append(length)


# In[23]:


# class_cnt = [0] * 5
# for features_data in features_data_points:
#     if features_data[-1] == 1:
#         class_cnt[0] += 1
#     if features_data[-1] == 2:
#         class_cnt[1] += 1
#     if features_data[-1] == 3:
#         class_cnt[2] += 1
#     if features_data[-1] == 4:
#         class_cnt[3] += 1
#     if features_data[-1] == 5:
#         class_cnt[4] += 1

# print(class_cnt)
    
        
        


# In[24]:





# In[ ]:




