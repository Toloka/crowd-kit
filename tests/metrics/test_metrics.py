import pandas as pd
from nltk.metrics.distance import masi_distance
from pandas.testing import assert_series_equal

from crowdkit.aggregation.utils import get_accuracy
from crowdkit.metrics.data import alpha_krippendorff, consistency, uncertainty
from crowdkit.metrics.performers import accuracy_on_aggregates


def test_consistency(toy_answers_df):
    assert consistency(toy_answers_df) == 0.9384615384615385


def test_uncertainty(toy_answers_df):
    performers_skills = pd.Series(
        [0.6, 0.8, 1.0,  0.4, 0.8],
        index=pd.Index(['w1', 'w2', 'w3', 'w4', 'w5'], name='performer'),
    )
    assert uncertainty(toy_answers_df, performers_skills) == 0.12344835394606832


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


def test_alpha_krippendorff(toy_answers_df):
    assert alpha_krippendorff(pd.DataFrame.from_records([
        {'task': 'X', 'performer': 'A', 'label': 'Yes'},
        {'task': 'X', 'performer': 'B', 'label': 'Yes'},
        {'task': 'Y', 'performer': 'A', 'label': 'No'},
        {'task': 'Y', 'performer': 'B', 'label': 'No'},
    ])) == 1.0

    assert alpha_krippendorff(pd.DataFrame.from_records([
        {'task': 'X', 'performer': 'A', 'label': 'Yes'},
        {'task': 'X', 'performer': 'B', 'label': 'Yes'},
        {'task': 'Y', 'performer': 'A', 'label': 'No'},
        {'task': 'Y', 'performer': 'B', 'label': 'No'},
        {'task': 'Z', 'performer': 'A', 'label': 'Yes'},
        {'task': 'Z', 'performer': 'B', 'label': 'No'},
    ])) == 0.4444444444444444

    assert alpha_krippendorff(toy_answers_df) == 0.14219114219114215


def test_alpha_krippendorff_with_distance():
    whos_on_the_picture = pd.DataFrame.from_records([
        {'task': 'X', 'performer': 'A', 'label': frozenset(['dog'])},
        {'task': 'X', 'performer': 'B', 'label': frozenset(['dog'])},
        {'task': 'Y', 'performer': 'A', 'label': frozenset(['cat'])},
        {'task': 'Y', 'performer': 'B', 'label': frozenset(['cat'])},
        {'task': 'Z', 'performer': 'A', 'label': frozenset(['cat'])},
        {'task': 'Z', 'performer': 'B', 'label': frozenset(['cat', 'mouse'])},
    ])

    assert alpha_krippendorff(whos_on_the_picture) == 0.5454545454545454
    assert alpha_krippendorff(whos_on_the_picture, masi_distance) == 0.6673336668334168
