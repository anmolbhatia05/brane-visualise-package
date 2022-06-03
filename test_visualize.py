
import pytest
import pandas as pd

import visualize


# import scikit classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

def test_get_model():
    model_name = 'dtc'
    result = visualize.get_model(model_name)
    expect_model = DecisionTreeClassifier()
    assert result == expect_model

def test_visualize_eda():
    test_input_path = './data/train.csv'
    result = visualize.visualize_EDA(test_input_path)
    expected_output = 0
    assert  expected_output == result


