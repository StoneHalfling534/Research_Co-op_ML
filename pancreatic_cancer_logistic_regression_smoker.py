# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:43:04 2019
@author: s201100013
"""
import pandas as pd
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import random
import numpy as np
from matplotlib.colors import ListedColorMap


random.seed(1234)
np.random.seed(1234)
data = pd.read_csv('pancreatic_cancer_smokers.csv')
print(data)
x=data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
target = data['case (1: case, 0: control)']
data = data.drop('case (1: case, 0: control)', axis=1)
print(data)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
sc = StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
x_validate_std = sc.transform(x_validate)

logreg = LogisticRegression(C=1000.0, random_state=42)

logreg.fit(x_train_std, y_train)
y_pred = logreg.predict(x_validate)
accuracy = accuracy_score(y_validate, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
print(accuracy)
print(cnf_matrix)
