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
    # here for testing DecisionTreeClassifier() is used
    model_name = 'dtc'
    result = visualize.get_model(model_name)
    expect_model = DecisionTreeClassifier()
    assert type(result) == type(expect_model)

@patch("matplotlib.pyplot.savefig")
def test_visualize_eda(mock_show):
    # this tests if the visualize_EDA function creates plots correctly, 
    # visualize_EDA should return 0 if all plots created
    test_input_path = 'data/train.csv'
    result = visualize.visualize_EDA(test_input_path)
    expected_output = 0
    assert  expected_output == result

@patch("matplotlib.pyplot.savefig")
def test_visualize_model(mock_show):
    # this tests if the visualize_model function creates plots correctly, 
    # visualize_model should return 0 if all plots created
    # here rfc i.e. random forest model is used
    test_input_path = "./data/prep_data1.csv"
    result = visualize.visualize_model(test_input_path, "rfc")
    expected_output = 0
    assert result == expected_output 

@patch("matplotlib.pyplot.savefig")
def test_visualize_eda_error(mock_show):
    # this tests if the visualize_EDA function returns correct error number, 
    # visualize_EDA should return 2 if provided file name is wrong
    test_input_path = 'data/train123.csv'
    result = visualize.visualize_EDA(test_input_path)
    expected_output = 2
    assert  expected_output == result

@patch("matplotlib.pyplot.savefig")
def test_get_null_values(mock_show):
    # this tests if the null value count plot is created or not, 
    train_df = pd.read_csv("data/train.csv", index_col=0)
    result = visualize.get_null_values(train_df)

