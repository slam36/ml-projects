import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import KFold
from sklearn.metrics import *
import time


def explore_hyperparameter(X, y, X_train, y_train, X_test, y_test, clf, param, param_name):
   train_scores = []
   test_scores = []
   cv_scores = []
   for i in range(len(clf)):
      clf[i].fit(X_train, y_train)
      train_pred = clf[i].predict(X_train)
      score = accuracy_score(y_train, train_pred)
      train_scores.append(score)
      y_pred = clf[i].predict(X_test)
      score = accuracy_score(y_test, y_pred)
      test_scores.append(score)
      kf = KFold(n_splits=5)
      kfscores = []
      for train_index, test_index in kf.split(X):
         clf[i].fit(X[train_index], y[train_index])
         y_pred = clf[i].predict(X[test_index])
         kfscores.append(accuracy_score(y[test_index], y_pred))
      cv_scores.append(sum(kfscores) / len(kfscores))


   plt.plot(param, train_scores)
   plt.plot(param, test_scores)
   plt.plot(param, cv_scores)
   plt.xlabel(param_name)
   plt.ylabel('AUC Score')
   plt.legend(['Training', 'Testing', 'Cross Validation'])
   plt.show()
