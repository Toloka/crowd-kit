"""
Simplest aggregation algorithms tests on toy YSDA dataset
Testing all boundary conditions and asserts
"""
import numpy as np
import pandas as pd
import pytest
from crowdkit.aggregation import DawidSkene
from pandas.testing import assert_frame_equal, assert_series_equal


def test_aggregate_ds_on_toy_ysda(toy_answers_df, toy_ground_truth_df):
    np.random.seed(42)
    assert_series_equal(
        DawidSkene(10).fit(toy_answers_df).labels_.sort_index(),
        toy_ground_truth_df.sort_index(),
    )


def test_aggregate_ds_on_simple(simple_answers_df, simple_ground_truth_df):
    np.random.seed(42)
    assert_series_equal(
        DawidSkene(10).fit(simple_answers_df).labels_.sort_index(),
        simple_ground_truth_df.sort_index(),
    )


def _make_probas(data):
    # TODO: column should not be an index!
    columns = pd.Index(['task', 'no', 'yes'], name='label')
    return pd.DataFrame(data, columns=columns).set_index('task')


def _make_tasks_labels(data):
    # TODO: should task be indexed?
    return pd.DataFrame(data, columns=['task', 'label']).set_index('task').squeeze().rename()


def _make_errors(data):
    return pd.DataFrame(
        data,
        columns=['performer', 'label', 'no', 'yes'],
    ).set_index(['performer', 'label'])


@pytest.fixture
def data():
    return pd.DataFrame(
        [
            ['t1', 'w1', 'no'],
            ['t1', 'w2', 'yes'],
            # ['t1', 'w3', np.NaN],
            ['t1', 'w4', 'yes'],
            ['t1', 'w5', 'no'],

            ['t2', 'w1', 'yes'],
            ['t2', 'w2', 'yes'],
            ['t2', 'w3', 'yes'],
            ['t2', 'w4', 'no'],
            ['t2', 'w5', 'no'],

            ['t3', 'w1', 'yes'],
            ['t3', 'w2', 'no'],
            ['t3', 'w3', 'no'],
            ['t3', 'w4', 'yes'],
            ['t3', 'w5', 'no'],

            ['t4', 'w1', 'yes'],
            ['t4', 'w2', 'yes'],
            ['t4', 'w3', 'yes'],
            ['t4', 'w4', 'yes'],
            ['t4', 'w5', 'yes'],

            ['t5', 'w1', 'yes'],
            ['t5', 'w2', 'no'],
            ['t5', 'w3', 'no'],
            ['t5', 'w4', 'no'],
            ['t5', 'w5', 'no'],
        ],
        columns=['task', 'performer', 'label']
    )


@pytest.fixture
def probas_iter_0():
    return _make_probas([
        ['t1', 0.5, 0.5],
        ['t2', 0.4, 0.6],
        ['t3', 0.6, 0.4],
        ['t4', 0.0, 1.0],
        ['t5', 0.8, 0.2],
    ])


@pytest.fixture
def priors_iter_0():
    return pd.Series([0.46, 0.54], pd.Index(['no', 'yes'], name='label'))


@pytest.fixture
def tasks_labels_iter_0():
    return _make_tasks_labels([
        ['t1', 'no'],
        ['t2', 'yes'],
        ['t3', 'no'],
        ['t4', 'yes'],
        ['t5', 'no'],
    ])


@pytest.fixture
def errors_iter_0():
    return _make_errors([
        ['w1', 'no',  0.22, 0.19],
        ['w1', 'yes', 0.78, 0.81],

        ['w2', 'no',  0.61, 0.22],
        ['w2', 'yes', 0.39, 0.78],

        ['w3', 'no',  0.78, 0.27],
        ['w3', 'yes', 0.22, 0.73],

        ['w4', 'no',  0.52, 0.30],
        ['w4', 'yes', 0.48, 0.70],

        ['w5', 'no',  1.00, 0.63],
        ['w5', 'yes', 0.00, 0.37],
    ])


@pytest.fixture
def probas_iter_1():
    return _make_probas([
        ['t1', 0.35, 0.65],
        ['t2', 0.26, 0.74],
        ['t3', 0.87, 0.13],
        ['t4', 0.00, 1.00],
        ['t5', 0.95, 0.05],
    ])


@pytest.fixture
def priors_iter_1():
    # return pd.Series([0.49, 0.51], pd.Index(['no', 'yes'], name='label'))
    return pd.Series([0.49, 0.51], pd.Index(['no', 'yes']))


@pytest.fixture
def tasks_labels_iter_1():
    return _make_tasks_labels([
        ['t1', 'yes'],
        ['t2', 'yes'],
        ['t3', 'no'],
        ['t4', 'yes'],
        ['t5', 'no'],
    ])


@pytest.fixture
def errors_iter_1():
    return _make_errors([
        ['w1', 'no',  0.14, 0.25],
        ['w1', 'yes', 0.86, 0.75],

        ['w2', 'no',  0.75, 0.07],
        ['w2', 'yes', 0.25, 0.93],

        ['w3', 'no',  0.87, 0.09],
        ['w3', 'yes', 0.13, 0.91],

        ['w4', 'no',  0.50, 0.31],
        ['w4', 'yes', 0.50, 0.69],

        ['w5', 'no',  1.00, 0.61],
        ['w5', 'yes', 0.00, 0.39],
    ])


@pytest.mark.parametrize('n_iter', [0, 1])
def test_dawid_skene_step_by_step(request, data, n_iter):
    probas = request.getfixturevalue(f'probas_iter_{n_iter}')
    labels = request.getfixturevalue(f'tasks_labels_iter_{n_iter}')
    errors = request.getfixturevalue(f'errors_iter_{n_iter}')
    priors = request.getfixturevalue(f'priors_iter_{n_iter}')

    ds = DawidSkene(n_iter).fit(data)
    assert_frame_equal(probas, ds.probas_, check_like=True, atol=0.005)
    assert_frame_equal(errors, ds.errors_, check_like=True, atol=0.005)
    assert_series_equal(priors, ds.priors_, atol=0.005)
    assert_series_equal(labels, ds.labels_, atol=0.005)


def test_dawid_skene_on_empty_input(request, data):
    ds = DawidSkene(10).fit(pd.DataFrame([], columns=['task', 'performer', 'label']))
    assert_frame_equal(pd.DataFrame(), ds.probas_, check_like=True, atol=0.005)
    assert_frame_equal(pd.DataFrame(), ds.errors_, check_like=True, atol=0.005)
    assert_series_equal(pd.Series(dtype=float), ds.priors_, atol=0.005)
    assert_series_equal(pd.Series(dtype=float), ds.labels_, atol=0.005)


@pytest.mark.xfail
@pytest.mark.parametrize('overlap', [3, 300, 30000])
def test_dawid_skene_overlap(overlap):
    data = pd.DataFrame([
        {
            'task': task_id,
            'performer': perf_id,
            'label': 'yes' if (perf_id - task_id) % 3 else 'no',
        }
        for perf_id in range(overlap)
        for task_id in range(3)
    ])

    ds = DawidSkene(20).fit(data)

    expected_probas = _make_probas([[task_id, 1/3., 2/3] for task_id in range(3)])
    expected_labels = _make_tasks_labels([[task_id, 'yes'] for task_id in range(3)])

    # TODO: check errors_
    assert_frame_equal(expected_probas, ds.probas_, check_like=True, atol=0.005)
    assert_series_equal(expected_labels, ds.labels_, atol=0.005)
    assert_series_equal(pd.Series({'no': 1/3, 'yes': 2/3}), ds.priors_, atol=0.005)
