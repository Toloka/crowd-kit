import pandas as pd
import pytest

from crowdkit.aggregation import BradleyTerry
from pandas.testing import assert_series_equal


@pytest.fixture
def data_empty():
    return pd.DataFrame([], columns=['performer', 'task', 'left', 'right', 'label'])


@pytest.fixture
def data_abc():
    return pd.DataFrame(
        [
            ['w1', 't1', 'a', 'b', 'a'],
            ['w1', 't2', 'b', 'c', 'b'],
            ['w1', 't3', 'c', 'a', 'a'],
        ],
        columns=['performer', 'task', 'left', 'right', 'label']
    )


@pytest.fixture
def data_equal():
    return pd.DataFrame(
        [
            ['w1', 't1', 'a', 'b', 'a'],
            ['w1', 't2', 'b', 'c', 'b'],
            ['w1', 't3', 'c', 'a', 'c'],
        ],
        columns=['performer', 'task', 'left', 'right', 'label']
    )


@pytest.fixture
def result_empty():
    return pd.Series([])


@pytest.fixture
def result_equal():
    return pd.Series([.333, .333, .333], index=['a', 'b', 'c'])


@pytest.fixture
def result_iter_0():
    return pd.Series([.333, .333, .333], index=['a', 'b', 'c'])


@pytest.fixture
def result_iter_10():
    return pd.Series([.934, .065, 0.], index=['a', 'b', 'c'])


def test_bradley_terry_empty(result_empty, data_empty):
    bt = BradleyTerry(n_iter=1).fit(data_empty)
    assert_series_equal(result_empty, bt.result_)


def test_bradley_terry_equal(result_equal, data_equal):
    bt = BradleyTerry(n_iter=10).fit(data_equal)
    assert_series_equal(result_equal, bt.result_, atol=0.005)


@pytest.mark.parametrize('n_iter', [0, 10])
def test_bradley_terry_step_by_step(request, data_abc, n_iter):
    result = request.getfixturevalue(f'result_iter_{n_iter}')
    bt = BradleyTerry(n_iter=n_iter).fit(data_abc)
    assert_series_equal(result, bt.result_, atol=0.005)
