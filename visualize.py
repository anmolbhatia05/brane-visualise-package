#!/usr/bin/env python3

import os
import sys
import yaml

# libraries imported for plotting
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import scikit classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

# this method plots bar plots and saves files in jpeg format
def barplots(df):   
    sns.countplot(df.Survived)
    plt.savefig('/data/img/survived.jpeg')

    sns.barplot(x='Sex', y='Survived', data=df)
    plt.savefig('/data/img/sex_vs_survived.jpeg')

    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.savefig('/data/img/bar_pclass_vs_survived.jpeg')
    get_null_values(df)


# this method plots histograms plots and saves files in jpeg format
def hist(df):
    graph = sns.FacetGrid(df, col='Survived')
    graph.map(plt.hist, 'Age', bins=20)
    graph.savefig('/data/img/hist_age_vs_survived.jpeg')

    grid = sns.FacetGrid(df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.savefig('/data/img/hist_pclass_vs_survived.jpeg')

# this method plots graph with features null values and saves in jpeg format
def get_null_values(df):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
    na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio %' :na_df})
    missing_data.plot(kind = "barh")
    plt.savefig('/data/img/missing_values.jpeg')

''' Based on the name input, it first fetches the dataframe and calls the internal methods 
    to plot graphs used for understanding the data (EDA).  
'''
def visualize_EDA(name: str) -> int:
    try:
        df = pd.read_csv(name)
        # Barplots
        barplots(df)
        # Histograms
        hist(df)
        return 0
    except IOError as e:
        return e.errno

# this function draws the feature importance graph and takes dataframe and model type in input
# mode possible values can be 'dtc' for DecisionTreeClassifier and 'rfc' is for RandomForestClassifier
def draw_feature_importance(df, mode):
    y_train = df['Survived']
    x_train = df.drop('Survived', axis='columns')
    model = get_model(mode)
    model.fit(x_train, y_train)
    importance = pd.DataFrame({'feature': x_train.columns, 'importance': np.round(model.feature_importances_,3)})
    importance = importance.sort_values('importance', ascending=False).set_index('feature')
    importance.plot(kind='bar', rot=0)
    plt.savefig('/data/img/feature_importance'+str(mode)+'.jpeg')

# this method returns the machine learning model using sklearn library functions
def get_model(name):
    if(name=='dtc'):
        model = DecisionTreeClassifier()
    elif(name=='rfc'):
        model = RandomForestClassifier()
    elif(name=='bnb'):
        model = BernoulliNB()
    return model

# this method plots heat map and shows correlation of features and saves heat map as jpeg format
# input is dataframe
def draw_heat_map(df):
    y_train = df['Survived']
    x_train = df.drop('Survived', axis=1)
    temp = pd.concat([x_train.iloc[:891], y_train], axis = 1)
    corr = temp.corr()
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(corr, cbar = True, annot = True, linewidths = 0.4)
    
    plt.savefig('/data/img/heatmap.jpeg')

''' Based on the name input, it first fetches the dataframe and calls the internal methods 
    to plot heat and feature important graphs 
'''
def visualize_model(name: str, mode: str) -> int:
    try:
        df = pd.read_csv(name, index_col=0)
        draw_heat_map(df)
        draw_feature_importance(df, mode)
        return 0
    except IOError as e:
        return e.errno

''' Based on the function called from branscript this main function calls the functions
    with business logic present in this python file, provides them suitable input data
'''
if __name__ == "__main__":
    if len(sys.argv) != 2 or (sys.argv[1] != "visualize_EDA" and sys.argv[1] != "read" and sys.argv[1] != "visualize_results"):
        exit(1)

    command = sys.argv[1]
    if command == "visualize_EDA":
        print(yaml.dump({"status": visualize_EDA(os.environ["NAME"])}))

    elif command == "visualize_results":
        print(yaml.dump({"status": visualize_model(os.environ["NAME"], os.environ["MODE"])}))