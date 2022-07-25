import pandas as pd

from crowdkit.aggregation.utils import get_most_probable_labels, get_accuracy, normalize_rows
from pandas.testing import assert_frame_equal, assert_series_equal


def test_get_most_probable_labels() -> None:
    proba = pd.DataFrame(
        [
            [0.9, 0.1, 0.0],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
            [0.5, 0.4, 0.1],
        ],
        columns=['a', 'b', 'c'],
        index=['w', 'x', 'y', 'z'],
    )

    most_probable_labels = pd.Series(
        ['a', 'b', 'c', 'a'],
        index=['w', 'x', 'y', 'z'],
    )

    assert_series_equal(get_most_probable_labels(proba), most_probable_labels)


def test_normalize_rows() -> None:
    scores = pd.DataFrame(
        [
            [4.5, 0.5, 0.0],
            [0.6, 1.0, 0.4],
            [0.3, 0.6, 2.1],
            [3.5, 2.8, 0.7],
        ],
        columns=['a', 'b', 'c'],
        index=['w', 'x', 'y', 'z'],
    )

    proba = pd.DataFrame(
        [
            [0.9, 0.1, 0.0],
            [0.3, 0.5, 0.2],
            [0.1, 0.2, 0.7],
            [0.5, 0.4, 0.1],
        ],
        columns=['a', 'b', 'c'],
        index=['w', 'x', 'y', 'z'],
    )

    assert_frame_equal(normalize_rows(scores), proba)


def test_get_accuracy() -> None:
    true_labels = pd.Series({'t1': 'c', 't2': 'b'})
    data = pd.DataFrame(
        [
            ['t1', 'p1', 'a', 1],
            ['t1', 'p2', 'a', 1],
            ['t1', 'p3', 'c', 3],
            ['t2', 'p1', 'b', 2],
            ['t2', 'p4', 'b', 1],
            ['t2', 'p3', 'c', 1],
        ],
        columns=['task', 'worker', 'label', 'weight']
    )

    # With weights
    assert get_accuracy(data, true_labels) == 6 / 9

    skills = pd.Series([3/5, 3/4], index=pd.Index(['t1', 't2'], name='task'))
    assert_series_equal(get_accuracy(data, true_labels, by='task'), skills)

    skills = pd.Series([2/3, 0, 3/4, 1], index=pd.Index(['p1', 'p2', 'p3', 'p4'], name='worker'))
    assert_series_equal(get_accuracy(data, true_labels, by='worker'), skills)

    skills = pd.Series([3/4, 3/5], index=pd.Index(['b', 'c'], name='true_label'))
    assert_series_equal(get_accuracy(data, true_labels, by='true_label'), skills)

    # Without weights
    data = data[['task', 'worker', 'label']]

    assert get_accuracy(data, true_labels) == 1 / 2

    skills = pd.Series([1/3, 2/3], index=pd.Index(['t1', 't2'], name='task'))
    assert_series_equal(get_accuracy(data, true_labels, by='task'), skills)

    skills = pd.Series([1/2, 0, 1/2, 1], index=pd.Index(['p1', 'p2', 'p3', 'p4'], name='worker'))
    assert_series_equal(get_accuracy(data, true_labels, by='worker'), skills)

    skills = pd.Series([2/3, 1/3], index=pd.Index(['b', 'c'], name='true_label'))
    assert_series_equal(get_accuracy(data, true_labels, by='true_label'), skills)
