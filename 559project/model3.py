import sklearn
from sklearn.preprocessing import StandardScaler
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import preprocessing
import numpy as np
import data
from utils import Config
from math import pi
from math import exp
from math import sqrt
import pandas as pd
import seaborn as sn
from sklearn import metrics

from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn import tree
# Split the dataset by class values, returns a dictionary
from random import randrange


def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if class_value not in separated:
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg)**2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column))
                 for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    exponent = exp(-((x - mean)**2 / (2 * stdev**2)))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(
            total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, count = class_summaries[i]
            probabilities[class_value] *= calculate_probability(
                row[i], mean, stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Calculate accuracy percentage
def accuracy_metric(test_set, predicted):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predicted[i]:
            correct += 1
    return correct / float(len(test_set)) * 100.0


def feature_selection(train_set, test_set):
    train_feature = [i[0:13] for i in train_set]
    train_label = [i[-1] for i in train_set]
    test_feature = [i[0:13] for i in test_set]
    test_label = [i[-1] for i in test_set]

    ##PCA
    #     pca = PCA(n_components=13)
    #     pca.fit(train_feature)
    #     train_feature = pca.transform(train_feature)
    #     test_feature = pca.transform(test_feature)
    ## selectkbest
    model = SelectKBest(k=8).fit(train_feature, train_label)
    train_feature = model.transform(train_feature)
    test_feature = model.transform(test_feature)

    return train_feature, train_label, test_feature, test_label


# Naive Bayes Algorithm
def naive_bayes(train, test):
    summarize = summarize_by_class(train)
    predictions = list()
    for row in test:
        output = predict(summarize, row)
        predictions.append(output)
    accuracy = accuracy_metric(test, predictions)
    # return predictions
    return accuracy


#SVC
def SVC_model(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    model = SVC(C=1, gamma='auto', kernel='rbf')
    # model.fit(train_feature, train_label)
    y_pred = model.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = model.score(test_feature, test_label)
    return accuracy, y_pred, test_label


def KNN_modell(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    classifier = KNeighborsClassifier(n_neighbors=4)
    y_pred = classifier.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = classifier.score(test_feature, test_label)
    return accuracy, y_pred, test_label


def log_regression(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    logreg = LogisticRegression()
    y_pred = logreg.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = logreg.score(test_feature, test_label)
    return accuracy, y_pred, test_label


def linear_discriminant(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    lda = LinearDiscriminantAnalysis()
    y_pred = lda.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = lda.score(test_feature, test_label)
    return accuracy, y_pred, test_label


def perceptron(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    clf = Perceptron(tol=1e-3, random_state=0)
    y_pred = clf.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = clf.score(test_feature, test_label)
    return accuracy, y_pred, test_label


def decisionTree(train_set, test_set):
    train_feature, train_label, test_feature, test_label = feature_selection(
        train_set, test_set)
    clf = tree.DecisionTreeClassifier()
    y_pred = clf.fit(train_feature, train_label).predict(test_feature)
    #     gnb.fit(train_feature, train_label).
    accuracy = clf.score(test_feature, test_label)
    return accuracy, y_pred, test_label




#split into k
def cross_split(dataset, folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    for i in range(len(folds)):
        fold = list()
        for index in folds[i]:
            fold.append(dataset_copy[index])
        dataset_split.append(fold)
    return dataset_split


def evaluate_algorithm(dataset, algorithm, folds, *args):
    scores = []
    datadset_split = cross_split(dataset, folds)
    for fold in datadset_split:
        train_set = list(datadset_split)
        test_set = fold
        train_set.remove(fold)
        train_set = sum(train_set, [])
        accuracy, y_pred, test_label = algorithm(train_set, test_set)
        scores.append(accuracy)
    return scores


def test(train, test, best_one):
    accuracy, y_pred, test_label = best_one(train, test)
    return accuracy, y_pred, test_label


dataset, folds = data.getResult(Config['train'])

# n_validation = Config['n_validation']
# epoches = Config['epoches']
# scores = evaluate_algorithm(dataset, naive_bayes, folds)
# # scores = evaluate_algorithm(dataset, naive_bayes, folds, n_validation, epoches)
# print('Scores: %s' % scores)
# print('Mean Accuracy: %.3f%%' % (np.mean(scores, axis=0) * 100))

algorithms = [
    KNN_modell, log_regression, SVC_model, linear_discriminant, naive_bayes,
    perceptron, decisionTree
]
accuracies = []
for algorithm in algorithms:
    scores = evaluate_algorithm(dataset, algorithm, folds)
    accuracy = np.mean(scores, axis=0) * 100
    accuracies.append(accuracy)
    #  print('Mean Accuracy: %.3f%%' % (np.mean(scores2, axis=0) * 100))
print(accuracies)

# graph
name_list = ['KNN', 'logic', 'SVM', 'linear', 'naive', 'perceptron', 'Tree']
plt.bar(range(len(accuracies)), accuracies, color='rgb', tick_label=name_list)
plt.show()

index = accuracies.index(max(accuracies))
print("the best index is " + str(index))

# scores2 = evaluate_algorithm(dataset, svm_model, folds)
# print('Mean Accuracy: %.3f%%' % (np.mean(scores2, axis=0) * 100))
best_one = algorithms[index]
t, e = data.getResult(Config['test'])
accuracy, y_pred, test_label = test(dataset, t, best_one)
print("the best test accuracy is " + str(accuracy))

# confusion matrix for test
# list1 = [2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0]
# list2 = [1, 1, 0, 1, 2, 1, 1, 0, 1, 0, 0, 0]
data = {'y_Actual': test_label, 'y_Predicted': y_pred}
df = pd.DataFrame(data, columns=['y_Actual', 'y_Predicted'])
confusion_matrix = pd.crosstab(df['y_Actual'],
                               df['y_Predicted'],
                               rownames=['Actual'],
                               colnames=['Predicted'])

sn.heatmap(confusion_matrix, annot=True)
plt.show()