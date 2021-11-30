import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal
from crowdkit.aggregation import BradleyTerry, NoisyBradleyTerry


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
    return pd.Series([], dtype=np.float64, name='agg_score')


@pytest.fixture
def result_equal():
    return pd.Series([1/3, 1/3, 1/3], index=['a', 'b', 'c'], name='agg_score')


@pytest.fixture
def noisy_bt_result():
    return pd.Series([0.999993, 0.653991, 0.000025], index=pd.Index(['a', 'b', 'c'], name='label'), name='agg_score')


@pytest.fixture
def noisy_bt_result_equal():
    return pd.Series([0.672879, 0.043679, 0.985681], index=pd.Index(['a', 'b', 'c'], name='label'), name='agg_score')


@pytest.fixture
def result_iter_0():
    return pd.Series([1/3, 1/3, 1/3], index=['a', 'b', 'c'], name='agg_score')


@pytest.fixture
def result_iter_10():
    return pd.Series([.934, .065, 0.], index=['a', 'b', 'c'], name='agg_score')


def test_bradley_terry_empty(result_empty, data_empty):
    bt = BradleyTerry(n_iter=1).fit(data_empty)
    assert_series_equal(result_empty, bt.scores_)


def test_bradley_terry_equal(result_equal, data_equal):
    bt = BradleyTerry(n_iter=10).fit(data_equal)
    assert_series_equal(result_equal, bt.scores_, atol=0.005)


@pytest.mark.parametrize('n_iter', [0, 10])
def test_bradley_terry_step_by_step(request, data_abc, n_iter):
    result = request.getfixturevalue(f'result_iter_{n_iter}')
    bt = BradleyTerry(n_iter=n_iter).fit(data_abc)
    assert_series_equal(result, bt.scores_, atol=0.005)


def test_noisy_bradley_terry(data_abc, noisy_bt_result):
    noisy_bt = NoisyBradleyTerry().fit(data_abc)
    assert_series_equal(noisy_bt.scores_, noisy_bt_result, atol=0.005)
    assert noisy_bt.skills_.name == 'skill'
    assert noisy_bt.biases_.name == 'bias'


def test_noisy_bradley_terry_equal(data_equal, noisy_bt_result_equal):
    noisy_bt = NoisyBradleyTerry().fit(data_equal)
    assert_series_equal(noisy_bt.scores_, noisy_bt_result_equal, atol=0.005)
