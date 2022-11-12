
import pandas as pd
import numpy as np
import numpy.matlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import metrics

def conf_mat(true_target, pred, name='', precision_conf=False):
    non_nan_ind = np.where(np.logical_and(np.isnan(true_target) == False, np.isnan(pred) == False))[0]

    plt.figure(figsize=(8, 6))
    # Print the normalized confusion matrix
    ax1 = plt.subplot(1, 1, 1)
    real_conf = metrics.confusion_matrix(np.array(true_target)[non_nan_ind], np.array(pred)[non_nan_ind])
    if precision_conf is True:
        a = real_conf.sum(axis=0)
        pred_label_size_mat = np.matlib.repmat(a, len(a), 1)
        b = real_conf/pred_label_size_mat
    else:
        a = real_conf.sum(axis=1)
        true_label_size_mat = np.transpose(np.matlib.repmat(a, len(a), 1))
        b = real_conf/true_label_size_mat
    b = np.round(b, decimals=3)

    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            ax1.text(j, i, b[i, j], ha="center", va="center", color="r")

    ax1.imshow(b, cmap='Blues', interpolation='nearest')

    # plotting
    plt.title(name)
    plt.show()
    
def precision_recal_report(true_target, pred, name=''):
    # nan remove
    non_nan_ind = np.where(np.logical_and(np.isnan(true_target) == False, np.isnan(pred) == False))[0]
    # Print the precision and recall, among other metrics
    print(metrics.classification_report(np.array(true_target)[non_nan_ind], np.array(pred)[non_nan_ind], digits=4))

def conf_execute(true_target, pred, name=''):
    # nan remove
    non_nan_ind = np.where(np.logical_and(np.isnan(true_target) == False, np.isnan(pred) == False))[0]
    # Print the precision and recall, among other metrics
    return metrics.confusion_matrix(np.array(true_target)[non_nan_ind], np.array(pred)[non_nan_ind])
