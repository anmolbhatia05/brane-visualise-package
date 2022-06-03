
import pytest
import pandas as pd

import visualize

def test_visualize_eda():
    test_input_path = '/data/train.csv'
    result = visualize.visualize_EDA(test_input_path)
    expected_output = 0
    assert  expected_output == result


