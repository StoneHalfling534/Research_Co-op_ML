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
x_validate, y_validate, x_test, y_test = train_test_split(x_test, y_test, test_size=0.5)
np.random.seed(1234)

n_estimators = [1, 2, 4, 8, 16, 32, 64 ,100, 200]
validation_results = []

for i in n_estimators:
    clf = RandomForestClassifier(n_estimators=i)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_validate)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    validation_results.append(roc_auc)
line1 = plt.plot(n_estimators, validation_results, label="Test AUC")    
plt.ylabel('AUC Score')
plt.xlabel('Number of Trees')
plt.title("Number of Trees vs Performance in Random Forest Algorithm")
plt.savefig("n_estimators.pdf")
plt.show()