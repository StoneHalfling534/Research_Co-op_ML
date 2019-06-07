# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 13:39:19 2019
@author: Dhruv Upadhyay
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier 
#from treeinterpreter import treeinterpreter as ti, utils
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase
from pdpbox import pdp
from plotnine import *
#from mpl_toolkit.axes_grid1 import make_axes_locatable
import numpy as np
import random
import seaborn as sns
from sklearn.utils.multiclass import unique_labels

random.seed(1234)
np.random.seed(1234)
data = pd.read_csv("pancreatic_cancer_smokers_good.csv")
target = data['case (1: case, 0: control)']
data.drop('case (1: case, 0: control)', axis=1, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size = 0.2)
x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5)
clf = RandomForestClassifier(n_estimators=60, max_depth=4, min_samples_split=0.01)
clf.fit(x_train, y_train)
#print(x_train.columns)
y_pred = clf.predict(x_test)

clf_accuracy = accuracy_score(y_validate, y_pred)
print(clf_accuracy)

#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)

features=[
        'ID',
        'sex (0:female, 1: male)',
        'age',
        'smoker (0: no, 1: yes)',
        'family (0: no, 1: yes)',
        'rs13303010_G',
        'rs12615966_T',
        'rs657152_A',
        'rs9564966_A',
        'rs16986825_T'
        ]
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index = [i for i in "AB"],
                  columns = [i for i in "AB"])
plt.figure(figsize=(10,7))
ax = plt.axes()
sns.heatmap(df_cm, annot=True)
ax.set_title("Confusion Matrix for Random Forest Model")
plt.show()

def individual_contributions():
    clf_pred, clf_bias, contributions = ti.predict(clf, x_test)
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
    return contributions_dataframe

def contributions_histogram(contributions_dataframe):
    plt.rcParams['figure.figsize'] = (8.0, 10.0)
    contributions_dataframe.plot(kind = 'barh', color=(0.2, 0.4, 0.6, 0.6))
    plt.title('Feature Contributions for Pancreatic Cancer Risk Model')
    #plt.xscale('log')
    plt.savefig('contributions.png' , bbox_inches='tight')
    plt.savefig('contributions.pdf', bbox_inches='tight')
    plt.show()

def box_plot_feature_importance():
    plot = sns.boxplot(data=individual_contributions()[1])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha="right")
    plot.set(title='Feature Importance Distributions', ylabel='Importance')
    plt.tight_layout
    plt.savefig('boxplot_randomforest.pdf')
    plt.show()
    
def violin_plot_feature_importance():
    plot = sns.violinplot(data=individual_contributions()[1])
    plot.set_xticklabels(plot.get_xticklabels(), rotation=40, ha="right")
    plot.set(title='Feature Importance Distributions', ylabel='Importance')
    plt.tight_layout
    plt.savefig('violinplot_randomforest.pdf')
    plt.show()
    
def two_way_plot_pdp(feats, clusters=None, feat_name=None):
    #feats = ['family (0: no, 1: yes)', 'smoker (0: no, 1: yes)']
    feat_name = feat_name or feats
    p = pdp.pdp_interact(clf, x_train, features=feats, model_features=x_train.columns)
    graph = pdp.pdp_interact_plot(p, feats)
    pdp.plot.savefig('pdp.png', dpi=200)
    return graph

contributions_histogram(individual_contributions())
