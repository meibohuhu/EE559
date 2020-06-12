#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing

import pandas as pd
from sklearn.decomposition import PCA


def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers) - 1)
    return math.sqrt(variance)


#deal with data set
def load(csv):
    train_set = np.genfromtxt(csv, delimiter=",")
    # print("The col is " + str(len(train_set[0,:])))
    # print(len(test_set))
    train_list = train_set[:, 3:]
    class_list = train_set[:, 1]
    id_list = train_set[:, 2]

    train_data = np.column_stack((train_list, class_list))
    # print(np.shape(train_data))
    train_data = train_data.tolist()
    del (train_data[0])

    class_data = class_list.tolist()
    del (class_data[0])
    # print(len(class_data))

    id_data = id_list.tolist()
    del (id_data[0])
    # print(len(id_data))

    return train_data, class_data, id_data


#process one data point into 13 features and class index
def feature_extraction(train_list):
    x_fea = []
    y_fea = []
    z_fea = []

    #delete class number
    mynewlist = [s for s in train_list if not math.isnan(s)]
    del (mynewlist[-1])
    #     print(mynewlist)

    # get rid of null data points
    for i in range(len(mynewlist)):
        if i % 3 == 0:
            x_fea.append(mynewlist[i])
        if i % 3 == 1:
            y_fea.append(mynewlist[i])
        if i % 3 == 2:
            z_fea.append(mynewlist[i])

    #one data point's 13 features
    length = (int)((len(mynewlist)) / 3)
    x_mean, y_mean, z_mean = mean(x_fea), mean(y_fea), mean(z_fea)
    x_std, y_std, z_std = stdev(x_fea), stdev(y_fea), stdev(z_fea)
    min_x, min_y, min_z = min(x_fea), min(y_fea), min(z_fea)
    max_x, max_y, max_z = max(x_fea), max(y_fea), max(z_fea)

    return x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, length


#index classification
def idx_clf(id_list):
    total_index = []
    i_0 = id_list[0]
    tmp_list = []
    tmp_list.append(0)
    for i in range(1, len(id_list)):
        if id_list[i] != i_0:
            total_index.append(tmp_list)
            tmp_list = []
        tmp_list.append(i)
        i_0 = id_list[i]

    total_index.append(tmp_list)
    return total_index
    # print(len(total_index))


def getResult(csv):
    train_data, class_data, id_data = load(csv)

    #length, x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, class_index
    features_data_points = []

    for li in train_data:
        tmp_list = []
        x_mean, y_mean, z_mean, x_std, y_std, z_std, min_x, min_y, min_z, max_x, max_y, max_z, length = feature_extraction(
            li)
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
        #         tmp_list.append((int)(class_data[index]))
        features_data_points.append(tmp_list)

    # M1_standardrazation
    features_data_points1 = preprocessing.scale(features_data_points)

    # pca = PCA(n_components=12)
    # X_train = pca.fit_transform(features_data_points1)

    # # M2_Normalization
    # features_data_points1 = preprocessing.normalize(features_data_points)
    final_arr = features_data_points1.tolist()

    total_index = idx_clf(id_data)

    index = 0
    for li in final_arr:
        li.append((int)(class_data[index]))
        index += 1

    return final_arr, total_index

