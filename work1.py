# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:19:14 2019

@author: rajar
"""

import scipy
import numpy
import matplotlib
import pandas
import sklearn
##
#print ('python: {}'.format(scipy.__version__))

from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#load dataset
url= "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length','sepal-width','petal-length','petal-width','class']
dataset = pandas.read_csv(url,names=names)

#shape
print(dataset.shape)

#head
print(dataset.head(20))

#describe
print(dataset.describe())

#distribution
print(dataset.groupby('class').size())

#histograms
dataset.hist()
plt.show()

#scatterplot matrix
scatter_matrix(dataset)
plt.show()