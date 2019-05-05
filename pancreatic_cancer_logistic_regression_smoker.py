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
import matplotlib.pyplot as plt

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
odds_ratios = np.exp(logreg.coef_)
odds_ratios = odds_ratios[0]
odd_ratios = np.resize(odds_ratios, ((odds_ratios.size), 1))
print(odds_ratios)
contributions_dataframe = pd.DataFrame({'sex (0:female, 1:male)': odds_ratios[1], 'age':odds_ratios[2], 'smoker': odds_ratios[3], 'family': odds_ratios[4],'rs13303010_G': odds_ratios[5],'rs12615966_T': odds_ratios[6], 'rs657152_A': odds_ratios[7],'rs9564966_A': odds_ratios[8],'rs16986825_T': odds_ratios[9]}, index=0)
data_headings = ['ID', 'sex (0:female, 1:male)', 'age', 'smoker (0: no, 1: yes)', 
                 'family (0: no, 1: yes)', 'rs13303010_G', 'rs12615966_T', 'rs657152_A'
                 'rs9564966_A', 'rs16986825_T']

y_pred = logreg.predict(x_validate)
accuracy = accuracy_score(y_validate, y_pred)
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.rcParams['figure.figsize'] = (8.0, 10.0)
contributions_dataframe.plot(kind='barh')
plt.title('Odds Ratios for Each Feature')
plt.xlabel('Features')
plt.ylabel('Odds Ratio')
plt.show()
