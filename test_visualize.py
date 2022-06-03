from unittest.mock import patch
import pytest
import pandas as pd

import visualize


# import scikit classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB

def test_get_model():
    # this tests if the correct model is return by get_model function
    model_name = 'dtc'
    result = visualize.get_model(model_name)
    expect_model = DecisionTreeClassifier()
    assert type(result) == type(expect_model)

@patch("matplotlib.pyplot.savefig")
def test_visualize_eda(mock_show):
    # this tests if the visualize_EDA function creates and saves plots correctly, 
    # visualize_EDA should return 0 if all plots created
    test_input_path = 'data/train.csv'
    result = visualize.visualize_EDA(test_input_path)
    expected_output = 0
    assert  expected_output == result

@patch("matplotlib.pyplot.savefig")
def test_visualize_model(mock_show):
    # this tests if the visualize_model function creates and saves plots correctly, 
    # visualize_model should return 0 if all plots created
    # here rfc i.e. random forest model is used
    test_input_path = "./data/prep_data1.csv"
    result = visualize.visualize_model(test_input_path, "rfc")
    expected_output = 0
    # df = pd.read_csv("data/prep_data1.csv", index_col=0)
    assert result == expected_output 


