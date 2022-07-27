"""
Simple aggregation tests.
"""
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
import typing as tp

from crowdkit.aggregation import BinaryRelevance, MajorityVote, DawidSkene, Wawa, GLAD


@pytest.fixture
def data_toy_binary_relevance() -> pd.DataFrame:
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
def binary_relevance_toy_result() -> pd.Series:
    result = pd.Series([['house', 'tree'], ['car']], index=['t1', 't2'], name='agg_label')
    result.index.name = 'task'
    return result


@pytest.mark.parametrize(
    'aggregator, args', [(MajorityVote, {}), (DawidSkene, {'n_iter': 10}), (Wawa, {}), (GLAD, {})]
)
def test_binary_relevance_aggregation_on_toy_data(aggregator: type, args: tp.Dict[str, tp.Any],
                                                  binary_relevance_toy_result: pd.Series,
                                                  data_toy_binary_relevance: pd.DataFrame) -> None:
    mb = BinaryRelevance(aggregator, args)
    assert_series_equal(binary_relevance_toy_result, mb.fit_predict(data_toy_binary_relevance))


@pytest.mark.parametrize(
    'aggregator', [MajorityVote, DawidSkene, Wawa, GLAD]
)
def test_binary_relevance_aggregation_without_args(aggregator: type,
                                                   binary_relevance_toy_result: pd.Series,
                                                   data_toy_binary_relevance: pd.DataFrame) -> None:
    mb = BinaryRelevance(aggregator)
    assert_series_equal(binary_relevance_toy_result, mb.fit_predict(data_toy_binary_relevance))


@pytest.mark.parametrize(
    'aggregator, args', [(MajorityVote, {}), (DawidSkene, {'n_iter': 10}), (Wawa, {}), (GLAD, {})]
)
def test_binary_relevance_aggregation_on_empty(aggregator: type, args: tp.Dict[str, tp.Any]) -> None:
    mb = BinaryRelevance(aggregator=aggregator, args=args)
    result = mb.fit_predict(pd.DataFrame([], columns=['task', 'worker', 'label']))
    assert_series_equal(pd.Series(dtype=float, name='agg_label'), result)
