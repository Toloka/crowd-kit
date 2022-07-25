import pandas as pd
import pytest

from pandas.testing import assert_series_equal
from crowdkit.aggregation import ROVER
from .data_rover import simple_text_result_rover  # noqa: F401


@pytest.fixture
def data_toy() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', 'a b c d'],
            ['w2', 't1', 'b z d e'],
            ['w3', 't1', 'b c d e f'],
        ],
        columns=['worker', 'task', 'text']
    )


@pytest.fixture
def rover_toy_result() -> pd.Series:
    result = pd.Series(['b c d e'], index=['t1'], name='agg_text')
    result.index.name = 'task'
    return result


def test_rover_aggregation(rover_toy_result: pd.Series, data_toy: pd.DataFrame) -> None:
    rover = ROVER(tokenizer=lambda x: x.split(' '), detokenizer=lambda x: ' '.join(x))
    assert_series_equal(rover_toy_result, rover.fit_predict(data_toy))


@pytest.fixture
def rover_single_overlap_data() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', 'a b c d'],
        ],
        columns=['worker', 'task', 'text']
    )


@pytest.fixture
def rover_single_overlap_result() -> pd.Series:
    result = pd.Series(['a b c d'], index=['t1'], name='agg_text')
    result.index.name = 'task'
    return result


def test_rover_single_overlap(rover_single_overlap_data: pd.DataFrame,
                              rover_single_overlap_result: pd.Series) -> None:
    rover = ROVER(tokenizer=lambda x: x.split(' '), detokenizer=lambda x: ' '.join(x))
    assert_series_equal(rover_single_overlap_result, rover.fit_predict(rover_single_overlap_data))


def test_rover_simple_text(simple_text_df: pd.DataFrame,
                           simple_text_result_rover: pd.Series) -> None:  # noqa F811
    rover = ROVER(tokenizer=lambda x: x.split(' '), detokenizer=lambda x: ' '.join(x))
    predicted = rover.fit_predict(simple_text_df.rename(columns={'output': 'text'}))
    assert_series_equal(predicted, simple_text_result_rover)
