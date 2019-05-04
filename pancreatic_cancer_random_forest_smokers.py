# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:39:19 2019
@author: DhruvUpadhyay
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier 
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt
import numpy as np
import random
import seaborn as sns

random.seed(1234)
np.random.seed(1234)
data = pd.read_csv("pancreatic_cancer_smokers.csv")
target = data['case (1: case, 0: control)']
data.drop('case (1: case, 0: control)', axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5)
clf = RandomForestClassifier(n_estimators=18, max_depth=4, min_samples_split=0.01)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

clf_accuracy = accuracy_score(y_test, y_pred)
print(clf_accuracy)

clf_pred, clf_bias, contributions = ti.predict(clf, x_test)
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

#The code below was taken from DataDive's treeinterpreter tutorial. 
#--------------------------------------------
feature_dict = dict((k, []) for k in list(x_test))
for i in range(len(x_test)):
    for c, feature in sorted(zip(contributions[i, : ,0], data.columns), 
                             key=lambda x: -abs(x[0])):
        feature_dict[feature].append(round(c, 2))
feature_mean_dict = {}
#--------------------------------------------
for feature in data.columns:
    feature_mean_dict[feature] = np.mean(feature_dict[feature])
sorted(feature_mean_dict.items(), key=lambda x: x[1], reverse=True)
feature = list(feature_mean_dict.keys())
values = list(feature_mean_dict.values())
contributions_dataframe = pd.Series(values, index=feature)
contributions_dataframe = contributions_dataframe.sort_values()
contributions_dataframe = contributions_dataframe.drop('ID', axis=0)
all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in clf], columns=data.columns)

def contributions_histogram():
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    contributions_dataframe.plot(kind = 'barh')
    plt.title('Feature Contributions for Pancreatic Cancer Risk Model')
    plt.savefig('contributions.png' , bbox_inches='tight')
    plt.savefig('contributions.pdf', bbox_inches='tight')

def box_plot_feature_importance():
    plot = sns.boxplot(data=all_feat_imp_df)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha="right")
    plot.set(title='Feature Importance Distributions', ylabel='Importance')
    plt.tight_layout
    plt.show()
    
def violin_plot_feature_importance():
    plot = sns.violinplot(data=all_feat_imp_df)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha="right")
    plot.set(title='Feature Importance Distributions', ylabel='Importance')
    plt.tight_layout
    plt.show()
#I still need to find a way to remove the ID from the graphs, as that skews the results
    
