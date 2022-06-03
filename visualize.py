#!/usr/bin/env python3

# importing python modules
import os
import sys
import yaml

# libraries imported for plotting and data analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# import scikit classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

# The functions are listed now.
# There are two type of functions - 1) brane/external functions that will be run in the containers 2) helper functions which are
# just python functions that will be called in the bran/external functions to do some tasks
# If a function has type annotation, then it is a brane/external function or else it's just a helper function


def barplots(df):
    # this method plots bar plots and saves files in jpeg format
    sns.countplot(df.Survived)
    plt.savefig('/data/img/survived.jpeg')

    sns.barplot(x='Sex', y='Survived', data=df)
    plt.savefig('/data/img/sex_vs_survived.jpeg')

    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.savefig('/data/img/bar_pclass_vs_survived.jpeg')
    # get_null_values(df)


def hist(df):
    # this method plots histograms plots and saves files in jpeg format
    graph = sns.FacetGrid(df, col='Survived')
    graph.map(plt.hist, 'Age', bins=20)
    graph.savefig('/data/img/hist_age_vs_survived.jpeg')

    grid = sns.FacetGrid(df, col='Survived', row='Pclass',
                         size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.savefig('/data/img/hist_pclass_vs_survived.jpeg')


def get_null_values(df):
    # this method plots graph with features null values and saves in jpeg format
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
    na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio %': na_df})
    missing_data.plot(kind="barh")
    plt.savefig('/data/img/missing_values.jpeg')


def visualize_EDA(name: str) -> int:
    """
        Creates visualisation graphs for null values, data skewness, feature correlation
    """
    try:
        df = pd.read_csv(name)
        # Barplots
        barplots(df)
        # Histograms
        # hist(df)
        return 0
    except IOError as e:
        return e.errno


def draw_feature_importance(df, mode):
    # this function draws the feature importance graph and takes dataframe and model type in input
    # mode possible values can be 'dtc' for DecisionTreeClassifier and 'rfc' is for RandomForestClassifier
    y_train = df['Survived']
    x_train = df.drop('Survived', axis='columns')
    model = get_model(mode)
    model.fit(x_train, y_train)
    importance = pd.DataFrame(
        {'feature': x_train.columns, 'importance': np.round(model.feature_importances_, 3)})
    importance = importance.sort_values(
        'importance', ascending=False).set_index('feature')
    importance.plot(kind='bar', rot=0)
    plt.savefig('/data/img/feature_importance'+str(mode)+'.jpeg')


def get_model(name):
    # this method returns the machine learning model using sklearn library functions
    if(name == 'dtc'):
        model = DecisionTreeClassifier()
    elif(name == 'rfc'):
        model = RandomForestClassifier()
    elif(name == 'bnb'):
        model = BernoulliNB()
    return model


def draw_heat_map(df):
    # this method plots heat map and shows correlation of features and saves heat map as jpeg format
    # input is dataframe
    y_train = df['Survived']
    x_train = df.drop('Survived', axis=1)
    temp = pd.concat([x_train.iloc[:891], y_train], axis=1)
    corr = temp.corr()
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(corr, cbar=True, annot=True, linewidths=0.4)

    plt.savefig('/data/img/heatmap.jpeg')


def visualize_model(name: str, mode: str) -> int:
    """
        Based on the name input, it first fetches the dataframe and calls the internal methods 
        to plot heat and feature important graphs 
    """
    try:
        df = pd.read_csv(name, index_col=0)
        draw_heat_map(df)
        # draw_feature_importance(df, mode)
        return 0
    except IOError as e:
        return e.errno


def create_image_folder() -> str:
    """
        Function to store the visualisation (histogram and barplots) graphs in the persistent folder /data
    """
    try:
        os.mkdir("/data/img")
        return "created"
    except:
        return "error"


# The entrypoint of the script
# Based on the function called from branscript this main function calls the functions
# with business logic present in this python file, provides them suitable input data
if __name__ == "__main__":
    # Make sure that at least one argument is given, that is either - 'visualize_EDA' or 'visualize_results'
    if len(sys.argv) != 2 or (sys.argv[1] != "visualize_EDA" and sys.argv[1] != "create_img" and sys.argv[1] != "visualize_results"):
        exit(1)

    # If it checks out, call the appropriate function
    command = sys.argv[1]

    if command == "visualize_EDA":
        # Print the result with the YAML package
        print(yaml.dump({"status": visualize_EDA(os.environ["NAME"])}))

    elif command == "visualize_results":
        print(yaml.dump({"status": visualize_model(
            os.environ["NAME"], os.environ["MODE"])}))

    elif command == "create_img":
        print(yaml.dump({"status": create_image_folder()}))
