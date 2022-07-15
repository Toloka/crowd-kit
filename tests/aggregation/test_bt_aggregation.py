from typing import Any

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_series_equal
from crowdkit.aggregation import BradleyTerry, NoisyBradleyTerry


@pytest.fixture
def data_empty() -> pd.DataFrame:
    return pd.DataFrame([], columns=['worker', 'task', 'left', 'right', 'label'])


@pytest.fixture
def data_abc() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', 'a', 'b', 'a'],
            ['w1', 't2', 'b', 'c', 'b'],
            ['w1', 't3', 'c', 'a', 'a'],
        ],
        columns=['worker', 'task', 'left', 'right', 'label']
    )


@pytest.fixture
def data_equal() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ['w1', 't1', 'a', 'b', 'a'],
            ['w1', 't2', 'b', 'c', 'b'],
            ['w1', 't3', 'c', 'a', 'c'],
        ],
        columns=['worker', 'task', 'left', 'right', 'label']
    )


@pytest.fixture
def result_empty() -> pd.Series:
    return pd.Series([], dtype=np.float64, name='agg_score')


@pytest.fixture
def result_equal() -> pd.Series:
    return pd.Series([1/3, 1/3, 1/3], index=['a', 'b', 'c'], name='agg_score')


@pytest.fixture
def noisy_bt_result() -> pd.Series:
    return pd.Series([1.0, 1.0, 1.497123058531228e-45], index=pd.Index(['a', 'b', 'c'], name='label'), name='agg_score')


@pytest.fixture
def noisy_bt_result_equal() -> pd.Series:
    return pd.Series([0.6715468044437242, 0.6462882683525435, 0.632947637600415], index=pd.Index(['a', 'b', 'c'], name='label'), name='agg_score')


@pytest.fixture
def result_iter_0() -> pd.Series:
    return pd.Series([1/3, 1/3, 1/3], index=['a', 'b', 'c'], name='agg_score')


@pytest.fixture
def result_iter_10() -> pd.Series:
    return pd.Series([.934, .065, 0.], index=['a', 'b', 'c'], name='agg_score')


def test_bradley_terry_empty(result_empty: pd.Series, data_empty: pd.DataFrame) -> None:
    bt = BradleyTerry(n_iter=1).fit(data_empty)
    assert_series_equal(result_empty, bt.scores_)


@pytest.mark.parametrize(
    'n_iter, tol', [(10, 0), (100500, 1e-5)]
)
def test_bradley_terry_equal(n_iter: int, tol: float,
                             result_equal: pd.Series, data_equal: pd.DataFrame) -> None:
    bt = BradleyTerry(n_iter=n_iter, tol=tol).fit(data_equal)
    assert_series_equal(result_equal, bt.scores_, atol=0.005)


@pytest.mark.parametrize('n_iter', [0, 10])
def test_bradley_terry_step_by_step(request: Any, data_abc: pd.DataFrame, n_iter: int) -> None:
    result = request.getfixturevalue(f'result_iter_{n_iter}')
    bt = BradleyTerry(n_iter=n_iter, tol=0).fit(data_abc)
    assert_series_equal(result, bt.scores_, atol=0.005)


@pytest.mark.parametrize(
    'n_iter, tol', [(10, 0), (100500, 1e-5)]
)
def test_noisy_bradley_terry(n_iter: int, tol: float,
                             data_abc: pd.DataFrame, noisy_bt_result: pd.Series) -> None:
    with pytest.warns(RuntimeWarning):
        noisy_bt = NoisyBradleyTerry(n_iter=n_iter, tol=tol).fit(data_abc)
    assert_series_equal(noisy_bt.scores_, noisy_bt_result, atol=0.005)
    assert noisy_bt.skills_.name == 'skill'
    assert noisy_bt.biases_.name == 'bias'


def test_noisy_bradley_terry_equal(data_equal: pd.DataFrame, noisy_bt_result_equal: pd.Series) -> None:
    noisy_bt = NoisyBradleyTerry().fit(data_equal)
    assert_series_equal(noisy_bt.scores_, noisy_bt_result_equal, atol=0.005)


@pytest.mark.parametrize('agg_class', [BradleyTerry, NoisyBradleyTerry])
def test_zero_iter(agg_class: Any, data_equal: pd.DataFrame, result_equal: pd.Series) -> None:
    aggregator = agg_class(n_iter=0)
    answers = aggregator.fit_predict(data_equal)
    assert len(answers.index.difference(result_equal.index)) == 0
