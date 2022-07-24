"""
Simplest aggregation algorithms tests on toy YSDA dataset
Testing all boundary conditions and asserts
"""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal
from crowdkit.aggregation import KOS


@pytest.mark.parametrize(
    'n_iter', [10, 100]
)
def test_aggregate_kos_on_data_with_bool_labels(n_iter, data_with_bool_labels, bool_labels_ground_truth):
    np.random.seed(42)
    assert_series_equal(
        KOS(n_iter=n_iter).fit(data_with_bool_labels).labels_.sort_index(),
        bool_labels_ground_truth.sort_index(),
    )


def test_kos_on_empty_input():
    result = KOS(n_iter=10).fit(pd.DataFrame([], columns=['task', 'worker', 'label']))
    assert_series_equal(pd.Series(dtype=float, name='agg_label'), result.labels_, atol=0.005)


def test_kos_not_binary_data(simple_answers_df):
    with pytest.raises(ValueError):
        KOS(n_iter=10).fit(simple_answers_df)
