"""
Simplest aggregation algorithms tests on toy YSDA dataset
Testing all boundary conditions and asserts
"""
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

from crowdkit.aggregation import KOS


@pytest.mark.parametrize("n_iter", [10, 100])
def test_aggregate_kos_on_data_with_bool_labels(
    n_iter: int,
    data_with_bool_labels: pd.DataFrame,
    bool_labels_ground_truth: 'pd.Series[Any]',
) -> None:
    np.random.seed(42)
    kos = KOS(n_iter=n_iter).fit(data_with_bool_labels)
    assert kos.labels_ is not None, 'no labels_'
    assert_series_equal(
        kos.labels_,
        bool_labels_ground_truth,
    )


def test_kos_on_empty_input() -> None:
    kos = KOS(n_iter=10).fit(pd.DataFrame([], columns=["task", "worker", "label"]))
    assert kos.labels_ is not None, 'no labels_'
    assert_series_equal(
        pd.Series([], dtype="O", name="agg_label"), kos.labels_, atol=0.005
    )


def test_kos_not_binary_data(simple_answers_df: pd.DataFrame) -> None:
    with pytest.raises(ValueError):
        KOS(n_iter=10).fit(simple_answers_df)
