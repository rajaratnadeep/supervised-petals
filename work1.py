# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:19:14 2019

@author: rajar
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

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

#split out validation dataset
array = dataset.values
X = array[:,0:4] #first 4 columns
Y = array[:,4] #last column
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train , Y_validation = model_selection.train_test_split(X,Y,test_size = validation_size,random_state = seed)

#test option and evaluation metric
scoring = 'accuracy'

models = []
models.append(('LR', LogisticRegression(solver='lbfgs')))
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVM',SVC(gamma = 'auto')))

#print(models)

#evaluate each model
results = []
names = []

for name,model in models:
    kfold = model_selection.KFold(n_splits =10, random_state = seed)
    cv_results = model_selection.cross_val_score(model,X_train,Y_train,cv = kfold, scoring = scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#make predictions on validation dataset
    
    for name,model in models:
        model.fit(X_train,Y_train)
        predictions = model.predict(X_validation)
        print(name)
        print(accuracy_score(Y_validation, predictions))
        print(classification_report(Y_validation,predictions))
        






























