#!/usr/bin/env python3

import os
import sys
import yaml

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# The functions


def visualize():
    try:
        df = pd.read_csv('/data/train.csv')
        sns.countplot(df['Survived'])
        plt.savefig('/data/first.jpeg')
        return 0
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
    except:
        return 1


# The entrypoint of the script
if __name__ == "__main__":
    if len(sys.argv) != 2 or (sys.argv[1] != "visualize" and sys.argv[1] != "read"):
        exit(1)

    command = sys.argv[1]
    if command == "visualize":
        # Print the result with the YAML package
        print(yaml.dump({"status": visualize()}))
