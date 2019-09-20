import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.metrics import *
import time

def show_auc(y_true, y_score):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Plot')
    plt.legend(loc="lower right")
    plt.show()

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=5,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


#This method I got from BD4H which I am also taking this semester
def classification_metrics(Y_pred, Y_true):
    #NOTE: It is important to provide the output in the same order
    accuracy = accuracy_score(Y_true, Y_pred)
    auc = roc_auc_score(Y_true, Y_pred)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1score = f1_score(Y_pred, Y_true)
    return accuracy, auc, precision, recall, f1score

#This method I got from BD4H which I am also taking this semester
def display_metrics(classifierName,Y_pred,Y_true):
    print("______________________________________________")
    acc, auc_, precision, recall, f1score = classification_metrics(Y_pred,Y_true)
    print(("Accuracy: "+str(acc)))
    print(("AUC: "+str(auc_)))
    print(("Precision: "+str(precision)))
    print(("Recall: "+str(recall)))
    print(("F1-score: "+str(f1score)))
    print("______________________________________________")
    print("")

def general_analysis(classifier, classifier_name, X_train, y_train, X_test, y_test, y_pred):
    print("performing analysis for " + classifier_name)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    y_predict = y_predict.astype(int)
    y_predict = np.array(y_predict)
    print(classifier)
    show_auc(y_test, y_pred)

    array = confusion_matrix(y_test, y_pred)
    df_cm = pd.DataFrame(array, index = [i for i in ["Not Spam", "Spam"]], columns = [i for i in ["Not Spam", "Spam"]])
    ax = sn.heatmap(df_cm, annot=True, cmap=sn.cm.rocket_r)
    ax.set_title(str(classifier_name) + " " + "Confusion Matrix", fontsize=20)
    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)

    start = time.time()
    plot_learning_curve(classifier, classifier_name + "Learning Curve", X_train, y_train, cv=5, train_sizes = np.linspace(0.1, 1, 10))
    elapsed_time = time.time() - start
    print(str(elapsed_time) + " sec")
    display_metrics(str(classifier_name), y_pred, y_test)
    accuracy, auc, precision, recall, f1score= classification_metrics(y_pred, y_test)
    return [accuracy, auc]







