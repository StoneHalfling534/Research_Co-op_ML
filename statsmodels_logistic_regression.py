# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:42:39 2019

@author: DhruvUpadhyay
"""

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import random
import seaborn as sns

np.random.seed(42)
random.seed(42)
data = pd.read_csv('pancreatic_cancer_smokers.csv')
features = list(data.columns)
features.remove('case (1: case, 0: control)')
features.reverse()
x=data.values
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)
target = data['case (1: case, 0: control)']
data = data.drop('case (1: case, 0: control)', axis=1)
print(data)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)
model = sm.Logit(y_validate, x_validate)
result = model.fit()
params = result.params
conf = result.conf_int()
conf['OR'] = params
conf.columns = ['2.5%', "97.5%", "OR"]


conf = np.exp(conf)

odds_ratios_95 = conf['OR']
odds_ratios_low = conf['2.5%']
odds_ratios_high = conf['97.5%']
low_diff = []
high_diff = []

for i in range(len(odds_ratios_low)):
    low_diff.append(odds_ratios_95[i]-odds_ratios_low[i])
    high_diff.append(odds_ratios_high[i]-odds_ratios_95[i])

asymmetric_error = [low_diff, high_diff]

    
def plot_hist():
    plt.rcParams['figure.figsize'] = (8.0, 10.0)   
    conf['OR'].plot(kind='barh')
    plt.title('Odds Ratios for Pancreatic Cancer Data')
    plt.errorbar(y=features, x=odds_ratios_95, xerr = asymmetric_error, ls = 'none')
    plt.xlabel('Odds Ratios')
    plt.ylabel('Features')
    plt.savefig('odds_ratios.pdf')
    plt.show()
   
def violin_plot():
    plot = sns.violinplot(data=params)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha="right")
    plot.set(title='Odds Ratios', xlabels='features', ylabel='Odds Ratios')
    plt.tight_layout
    plt.savefig('violinplot_OR.pdf')
    plt.show()

plot_hist()
