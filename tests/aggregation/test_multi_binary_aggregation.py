"""
Simple aggregation tests.
"""
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from crowdkit.aggregation import MultiBinary, MajorityVote, DawidSkene, Wawa, GLAD


@pytest.fixture
def data_toy():
    return pd.DataFrame(
        [
            ['t1', 'w1', ['house', 'tree']],
            ['t1', 'w2', ['house']],
            ['t1', 'w3', ['house', 'tree', 'grass']],
            ['t2', 'w1', ['car']],
            ['t2', 'w2', ['car', 'human']],
            ['t2', 'w3', ['train']],
        ],
        columns=['task', 'worker', 'label']
    )


@pytest.fixture
def multi_binary_toy_result():
    result = pd.Series([['house', 'tree'], ['car']], index=['t1', 't2'], name='agg_label')
    result.index.name = 'task'
    return result


@pytest.mark.parametrize(
    'aggregator, args', [(MajorityVote, {}), (DawidSkene, {'n_iter': 10}), (Wawa, {}), (GLAD, {})]
)
def test_multi_binary_aggregation_on_toy_data(aggregator, args, multi_binary_toy_result, data_toy):
    mb = MultiBinary(aggregator=aggregator, args=args)
    assert_series_equal(multi_binary_toy_result, mb.fit_predict(data_toy))


@pytest.mark.parametrize(
    'aggregator, args', [(MajorityVote, {}), (DawidSkene, {'n_iter': 10}), (Wawa, {}), (GLAD, {})]
)
def test_multi_binary_aggregation_on_empty(aggregator, args):
    mb = MultiBinary(aggregator=aggregator, args=args)
    result = mb.fit_predict(pd.DataFrame([], columns=['task', 'worker', 'label']))
    assert_series_equal(pd.Series(dtype=float, name='agg_label'), result)