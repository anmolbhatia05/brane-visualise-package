
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
    assert type(result) == type(expect_model)


