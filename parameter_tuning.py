# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 21:48:59 2019

@author: DhruvUpadhyay
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
random.seed(1234)
data = pd.read_csv("pancreatic_cancer_smokers.csv")
target = data['case (1: case, 0: control)']
data.drop('case (1: case, 0: control)', axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5)
np.random.seed(1234)

def n_estimators_tuning():
    n_estimators = [1, 2, 4, 8, 16, 32, 64 ,100, 200]
    validation_results = []
    
    for i in n_estimators:
        clf = RandomForestClassifier(n_estimators=i, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_validate)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation_results.append(roc_auc) 
    plt.plot(n_estimators, validation_results)
    plt.ylabel('AUC Score')
    plt.xlabel('Number of Trees')
    plt.title("Number of Trees vs Performance in Random Forest Algorithm")
    plt.savefig("n_estimators.pdf")
    plt.show()

def max_depth_tuning():
    max_depth = np.linspace(1, 32, 32, endpoint=True)
    validation_results = []
    
    for i in max_depth:
        clf = RandomForestClassifier(max_depth=i, n_jobs=-1)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_validate)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation_results.append(roc_auc)
    plt.plot(max_depth, validation_results)
    plt.ylabel("AUC Score")
    plt.xlabel("Maximum Tree Depth")
    plt.title("Max Tree Depth vs Performance in Random Forest Algorithm")
    plt.savefig("max_depth.pdf")
    plt.show()

def min_sample_split():
    min_sample_split = np.linspace(0.1, 1.0, 10, endpoint=True)
    validation_results = []
    
    for i in min_sample_split:
        clf = RandomForestClassifier(min_samples_split=i)
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_validate)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        validation_results.append(roc_auc)
    plt.plot(min_sample_split, validation_results)
    plt.ylabel("AUC Score")
    plt.xlabel("Minimum Sample Split")
    plt.title("Min Sample Split vs Random Forest Performance")
    plt.savefig('min_sample_split.pdf')
    plt.show()
