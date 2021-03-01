"""
Simplest aggregation algorythms tests on different datasets
Testing all boundary conditions and asserts
"""
import pytest

import pandas as pd

from crowdkit.aggregation import MajorityVote, MMSR, Wawa, GoldMajorityVote, ZeroBasedSkill

from .data_mv import *  # noqa: F401, F403
from .data_mmsr import *  # noqa: F401, F403
from .data_gold_mv import *  # noqa: F401, F403
from .data_wawa import *  # noqa: F401, F403
from .data_zbs import *  # noqa: F401, F403


def asserts_compare_df(left_df, right_df, sort_flds):
    left_df = left_df.sort_values(sort_flds).reset_index(drop=True)
    right_df = right_df.sort_values(sort_flds).reset_index(drop=True)
    pd.testing.assert_frame_equal(left_df, right_df, rtol=1e-5)


def asserts_compare_matrix_df(left_df, right_df):
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
        (MMSR, None, 'fit_predict_proba', 'toy', 'mmsr'),
        (MMSR, None, 'fit_predict_proba', 'simple', 'mmsr'),
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
        'Majority Vote predict labelson simple dataset',
        'Majority Vote predict probabilities on toy YSDA',
        'Majority Vote predict probabilities on simple dataset',
        'MMSR predict labels on toy YSDA',
        'MMSR predict labelson simple dataset',
        'MMSR predict probabilities on toy YSDA',
        'MMSR predict probabilities on simple dataset',
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
def test_fit_predict_aggregations_methods(
    request, not_random,
    agg_class, fit_method, predict_method,
    dataset, results_dataset
):
    """
    Tests all aggregation methods, that fit->predict chain works well, and at each step we have the correct values for:
        - tasks_labels
        - probas
        - performers_skills
    """
    # incoming datasets
    answers = request.getfixturevalue(f'{dataset}_answers_df')
    gold = request.getfixturevalue(f'{dataset}_gold_df')

    # result datasets for comparison
    labels_result = request.getfixturevalue(f'{dataset}_labels_result_{results_dataset}')
    skills_result = request.getfixturevalue(f'{dataset}_skills_result_{results_dataset}')
    probas_result = request.getfixturevalue(f'{dataset}_probas_result_{results_dataset}')

    aggregator = agg_class()
    if fit_method is not None:
        ret_val = getattr(aggregator, fit_method)(answers, gold)
        assert isinstance(ret_val, agg_class)
        asserts_compare_df(aggregator.performers_skills, skills_result, ['performer', 'skill'])

    somethings_predict = getattr(aggregator, predict_method)(answers)

    # checking after predict
    asserts_compare_df(aggregator.tasks_labels, labels_result, ['label', 'task'])
    asserts_compare_matrix_df(aggregator.probas, probas_result)
    asserts_compare_df(aggregator.performers_skills, skills_result, ['performer', 'skill'])
    if 'proba' in predict_method:
        assert somethings_predict is aggregator.probas
    else:
        assert somethings_predict is aggregator.tasks_labels
