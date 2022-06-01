#!/usr/bin/env python3

import os
import sys
import yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The functions


# def visualize():
#     try:
#         df = pd.read_csv('/data/train.csv')
#         sns.countplot(df['Survived'])
#         plt.savefig('/data/first.jpeg')
#         return 0
# sns.countplot(train['Pclass'])
# plt.bar(list(train['Sex'].value_counts().keys()),list(train['Sex'].value_counts()))
# graph = sns.FacetGrid(train, col='Survived')
# graph.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
# grid.map(plt.hist, 'Age', alpha=.5, bins=20)
# grid.add_legend();
# plt.figure(figsize=(10,5))
# sns.countplot(x='Pclass', data=train, hue='Survived').set(title='Survived Passengers based on Pclass')
# plt.figure(figsize=(10,5))
# sns.countplot(x='Sex', data=train, hue='Survived').set(title='Survived Passengers based on Sex')
# plt.figure(figsize=(10, 5))
# sns.countplot(x='Embarked', data=train, hue='Survived').set(title='Survived Passengers based on port of embarkation')
# plt.figure(figsize=(10, 5))
# sns.countplot(x='Pclass', data=train, hue='Survived').set(title='Survived Passengers based on Pclass')
# except:
#     return 1


def barplots(df):
    sns.countplot(df.Survived)
    plt.savefig('/data/img/survived.jpeg')

    sns.barplot(x='Sex', y='Survived', data=df)
    plt.savefig('/data/img/sex_vs_survived.jpeg')

    sns.barplot(x='Pclass', y='Survived', data=df)
    plt.savefig('/data/img/bar_pclass_vs_survived.jpeg')


def hist(df):
    graph = sns.FacetGrid(df, col='Survived')
    graph.map(plt.hist, 'Age', bins=20)
    graph.savefig('/data/img/hist_age_vs_survived.jpeg')

    grid = sns.FacetGrid(df, col='Survived', row='Pclass',
                         size=2.2, aspect=1.6)
    grid.map(plt.hist, 'Age', alpha=.5, bins=20)
    grid.savefig('/data/img/hist_pclass_vs_survived.jpeg')


def plot_null_values(df):
    if df.isnull().sum().sum() != 0:
        na_df = (df.isnull().sum() / len(df)) * 100
    na_df = na_df.drop(na_df[na_df == 0].index).sort_values(ascending=False)
    missing_data = pd.DataFrame({'Missing Ratio %': na_df})
    missing_data.plot(kind="barh")
    plt.savefig('/data/img/missing_values.jpeg')


def visualize_EDA(name: str) -> int:
    try:
        df = pd.read_csv(name)
        # missing_values
        plot_null_values(df)
        # Barplots
        barplots(df)
        # Histograms
        hist(df)
        return 0
    except IOError as e:
        return e.errno


def visualize_model(name: str) -> int:
    try:
        df = pd.read_csv(name)
        y_train = df['Survived']
        x_train = df.drop('Survived', axis=1)
        temp = pd.concat([x_train.iloc[:891], y_train], axis=1)
        corr = temp.corr()
        plt.figure(figsize=(10, 5))
        heatmap = sns.heatmap(corr, cbar=True, annot=True, linewidths=0.5)
        heatmap.savefig('/data/img/heatmap.jpeg')
        return 0
    except IOError as e:
        return e.errno


# The entrypoint of the script
if __name__ == "__main__":
    if len(sys.argv) != 2 or (sys.argv[1] != "visualize_EDA" and sys.argv[1] != "visualize_EDA" and sys.argv[1] != "visualize_results"):
        exit(1)

    command = sys.argv[1]
    if command == "visualize_EDA":
        print(yaml.dump({"status": visualize_EDA(os.environ["NAME"])}))

    elif command == "visualize_results":
        print(yaml.dump({"status": visualize_model(os.environ["NAME"])}))
