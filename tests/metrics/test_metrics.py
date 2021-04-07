import pandas as pd
from crowdkit.aggregation.utils import get_accuracy
from crowdkit.metrics.data import consistency
from crowdkit.metrics.performers import accuracy_on_aggregates
from pandas.testing import assert_series_equal


def test_consistency(toy_answers_df):
    assert consistency(toy_answers_df) == 0.9384615384615385


def test_golden_set_accuracy(toy_answers_df, toy_gold_df):
    assert get_accuracy(toy_answers_df, toy_gold_df) == 5 / 9
    assert get_accuracy(toy_answers_df, toy_gold_df, by='performer').equals(pd.Series(
        [0.5, 1.0, 1.0, 0.5, 0.0],
        index=['w1', 'w2', 'w3', 'w4', 'w5'],
    ))


def test_accuracy_on_aggregates(toy_answers_df):
    expected_performers_accuracy = pd.Series(
        [0.6, 0.8, 1.0,  0.4, 0.8],
        index=pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='performer'),
    )
    assert_series_equal(accuracy_on_aggregates(toy_answers_df, by='performer'), expected_performers_accuracy)
    assert accuracy_on_aggregates(toy_answers_df) == 0.7083333333333334
