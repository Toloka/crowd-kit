import pandas as pd
import pytest

from pandas.testing import assert_series_equal
from crowdkit.aggregation import ROVER


@pytest.fixture
def data_toy():
    return pd.DataFrame(
        [
            ['w1', 't1', 'a b c d'],
            ['w2', 't1', 'b z d e'],
            ['w3', 't1', 'b c d e f'],
        ],
        columns=['performer', 'task', 'text']
    )


@pytest.fixture
def rover_toy_result():
    return pd.Series(['b c d e'], index=['t1'])


def test_rover_aggregation(rover_toy_result, data_toy):
    rover = ROVER(tokenizer=lambda x: x.split(' '), detokenizer=lambda x: ' '.join(x))
    assert_series_equal(rover_toy_result, rover.fit_predict(data_toy))


@pytest.fixture
def rover_single_overlap_data():
    return pd.DataFrame(
        [
            ['w1', 't1', 'a b c d'],
        ],
        columns=['performer', 'task', 'text']
    )


@pytest.fixture
def rover_single_overlap_result():
    return pd.Series(['a b c d'], index=['t1'])


def test_rover_single_overlap(rover_single_overlap_data, rover_single_overlap_result):
    rover = ROVER(tokenizer=lambda x: x.split(' '), detokenizer=lambda x: ' '.join(x))
    assert_series_equal(rover_single_overlap_result, rover.fit_predict(rover_single_overlap_data))
