"""
Simplest aggregation algorythms tests on different datasets
Testing all boundary conditions and asserts
"""
from typing import Any, Callable

import pytest

import pandas as pd

from crowdkit.aggregation import MajorityVote, MMSR, Wawa, GoldMajorityVote, ZeroBasedSkill

from .data_mv import *  # noqa: F401, F403
from .data_mmsr import *  # noqa: F401, F403
from .data_gold_mv import *  # noqa: F401, F403
from .data_wawa import *  # noqa: F401, F403
from .data_zbs import *  # noqa: F401, F403
from pandas.testing import assert_series_equal, assert_frame_equal


def asserts_compare_df(left_df: pd.DataFrame, right_df: pd.DataFrame, sort_flds: str) -> None:
    left_df = left_df.sort_values(sort_flds).reset_index(drop=True)
    right_df = right_df.sort_values(sort_flds).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_df, right_df, rtol=1e-5)


def asserts_compare_matrix_df(left_df: pd.DataFrame, right_df: pd.DataFrame) -> None:
    left_df = left_df[sorted(left_df.columns.values)]
    right_df = right_df[sorted(right_df.columns.values)]
    pd.testing.assert_frame_equal(left_df, right_df, rtol=1e-5)


@pytest.mark.parametrize(
    'agg_class, fit_method, predict_method, dataset, results_dataset',
    [
        (MajorityVote, None, 'fit_predict', 'toy', 'mv'),
        (MajorityVote, None, 'fit_predict', 'simple', 'mv'),
        (MajorityVote, None, 'fit_predict_proba', 'toy', 'mv'),
        (MajorityVote, None, 'fit_predict_proba', 'simple', 'mv'),
        (MMSR, None, 'fit_predict', 'toy', 'mmsr'),
        (MMSR, None, 'fit_predict', 'simple', 'mmsr'),
        (MMSR, None, 'fit_predict_score', 'toy', 'mmsr'),
        (MMSR, None, 'fit_predict_score', 'simple', 'mmsr'),
        (Wawa, None, 'fit_predict', 'toy', 'wawa'),
        (Wawa, None, 'fit_predict', 'simple', 'wawa'),
        (Wawa, None, 'fit_predict_proba', 'toy', 'wawa'),
        (Wawa, None, 'fit_predict_proba', 'simple', 'wawa'),
        (GoldMajorityVote, 'fit', 'predict', 'toy', 'gold'),
        (GoldMajorityVote, 'fit', 'predict', 'simple', 'gold'),
        (GoldMajorityVote, 'fit', 'predict_proba', 'toy', 'gold'),
        (GoldMajorityVote, 'fit', 'predict_proba', 'simple', 'gold'),
        (ZeroBasedSkill, None, 'fit_predict', 'toy', 'zbs'),
        (ZeroBasedSkill, None, 'fit_predict', 'simple', 'zbs'),
        (ZeroBasedSkill, None, 'fit_predict_proba', 'toy', 'zbs'),
        (ZeroBasedSkill, None, 'fit_predict_proba', 'simple', 'zbs'),
    ],
    ids=[
        'Majority Vote predict labels on toy YSDA',
        'Majority Vote predict labels on simple dataset',
        'Majority Vote predict probabilities on toy YSDA',
        'Majority Vote predict probabilities on simple dataset',
        'MMSR predict labels on toy YSDA',
        'MMSR predict labelson simple dataset',
        'MMSR predict scores on toy YSDA',
        'MMSR predict scores on simple dataset',
        'Wawa predict labels on toy YSDA',
        'Wawa predict labels on simple dataset',
        'Wawa predict probabilities on toy YSDA',
        'Wawa predict probabilities on simple dataset',
        'Gold predict labels on toy YSDA',
        'Gold predict labels on simple dataset',
        'Gold predict probabilities on toy YSDA',
        'Gold predict probabilities on simple dataset',
        'ZBS predict labels on toy YSDA',
        'ZBS predict labels on simple dataset',
        'ZBS predict probabilities on toy YSDA',
        'ZBS predict probabilities on simple dataset',
    ],
)
def test_fit_predict_classification_aggregations_methods(
    request: Any, not_random: Callable[[], None],
    agg_class: Any, fit_method: str, predict_method: str,
    dataset: pd.DataFrame, results_dataset: pd.Series
) -> None:
    """
    Tests all aggregation methods, that fit->predict chain works well, and at each step we have the correct values for:
        - tasks_labels
        - probas
        - workers_skills
    """
    # incoming datasets
    answers = request.getfixturevalue(f'{dataset}_answers_df')
    gold = request.getfixturevalue(f'{dataset}_gold_df')

    # result datasets for comparison
    labels_result = request.getfixturevalue(f'{dataset}_labels_result_{results_dataset}')
    skills_result = request.getfixturevalue(f'{dataset}_skills_result_{results_dataset}')

    aggregator = agg_class()

    if hasattr(aggregator, 'predict_score'):
        scores_result = request.getfixturevalue(f'{dataset}_scores_result_{results_dataset}')
    else:
        probas_result = request.getfixturevalue(f'{dataset}_probas_result_{results_dataset}')

    if fit_method is not None:
        ret_val = getattr(aggregator, fit_method)(answers, gold)
        assert isinstance(ret_val, agg_class)
        assert_series_equal(aggregator.skills_, skills_result, rtol=1e-5)

    somethings_predict = getattr(aggregator, predict_method)(answers)

    # checking after predict
    assert_series_equal(aggregator.labels_.sort_index(), labels_result.sort_index(), rtol=1e-5)
    assert_series_equal(aggregator.skills_.sort_index(), skills_result.sort_index(), rtol=1e-5)
    if hasattr(aggregator, 'predict_score'):
        assert_frame_equal(aggregator.scores_, scores_result, check_like=True, rtol=1e-5)
    else:
        assert_frame_equal(aggregator.probas_, probas_result, check_like=True, rtol=1e-5)

    if 'proba' in predict_method:
        assert somethings_predict is aggregator.probas_
    elif 'score' in predict_method:
        assert somethings_predict is aggregator.scores_
    else:
        assert somethings_predict is aggregator.labels_

    # check result series names
    for member_name in dir(aggregator):
        member = getattr(aggregator, member_name)
        if member_name.endswith('_') and isinstance(member, pd.Series):
            if member_name == 'labels_':
                assert member.name == 'agg_label'
            else:
                # current convention for available after fit series names is to strip trailing underscore and use
                # variable name in singular form
                assert member_name[:-2] == member.name
