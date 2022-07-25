"""
Simplest aggregation algorythms tests on different datasets
Testing all boundary conditions and asserts
"""
from typing import Any, Callable, Type

import pandas as pd
import pytest

# from sklearn.exceptions import NotFittedError
from crowdkit.aggregation import MajorityVote, Wawa, GoldMajorityVote, ZeroBasedSkill, MMSR, DawidSkene, GLAD
from .data_gold_mv import *  # noqa: F401, F403

# less field in all crowd datasets


@pytest.fixture
def answers_no_task() -> pd.DataFrame:
    return pd.DataFrame({'worker': ['w1'], 'label': ['no']})


@pytest.fixture
def answers_no_label() -> pd.DataFrame:
    return pd.DataFrame({'worker': ['w1'], 'task': ['t1']})


@pytest.fixture
def answers_no_worker() -> pd.DataFrame:
    return pd.DataFrame({'task': ['t1'], 'label': ['no']})


@pytest.fixture
def gold_no_task() -> pd.DataFrame:
    return pd.DataFrame({'label': ['no']})


@pytest.fixture
def gold_no_label() -> pd.DataFrame:
    return pd.DataFrame({'task': ['t1']})


@pytest.mark.parametrize(
    'agg_class, predict_method, exception, answers_dataset',
    [
        (MajorityVote, 'fit_predict',       KeyError, 'answers_no_task'),
        (MajorityVote, 'fit_predict',       KeyError, 'answers_no_label'),
        (MajorityVote, 'fit_predict',       KeyError, 'answers_no_worker'),
        (MajorityVote, 'fit_predict_proba', KeyError, 'answers_no_task'),
        (MajorityVote, 'fit_predict_proba', KeyError, 'answers_no_label'),
        (MajorityVote, 'fit_predict_proba', KeyError, 'answers_no_worker'),
        (MMSR, 'fit_predict',       KeyError, 'answers_no_task'),
        (MMSR, 'fit_predict',       KeyError, 'answers_no_label'),
        (MMSR, 'fit_predict',       KeyError, 'answers_no_worker'),
        (MMSR, 'fit_predict_score', KeyError, 'answers_no_task'),
        (MMSR, 'fit_predict_score', KeyError, 'answers_no_label'),
        (MMSR, 'fit_predict_score', KeyError, 'answers_no_worker'),
        (Wawa,         'fit_predict',       KeyError, 'answers_no_task'),
        (Wawa,         'fit_predict',       KeyError, 'answers_no_label'),
        (Wawa,         'fit_predict',       KeyError, 'answers_no_worker'),
        (Wawa,         'fit_predict_proba', KeyError, 'answers_no_task'),
        (Wawa,         'fit_predict_proba', KeyError, 'answers_no_label'),
        (Wawa,         'fit_predict_proba', KeyError, 'answers_no_worker'),
        (ZeroBasedSkill,         'fit_predict',       KeyError, 'answers_no_task'),
        (ZeroBasedSkill,         'fit_predict',       KeyError, 'answers_no_label'),
        (ZeroBasedSkill,         'fit_predict',       KeyError, 'answers_no_worker'),
        (ZeroBasedSkill,         'fit_predict_proba', KeyError, 'answers_no_task'),
        (ZeroBasedSkill,         'fit_predict_proba', KeyError, 'answers_no_label'),
        (ZeroBasedSkill,         'fit_predict_proba', KeyError, 'answers_no_worker'),
    ],
    ids=[
        'Majority Vote predict raises on no "task"',
        'Majority Vote predict raises on no "label"',
        'Majority Vote predict raises on no "worker"',
        'Majority Vote predict_proba raises on no "task"',
        'Majority Vote predict_proba raises on no "label"',
        'Majority Vote predict_proba raises on no "worker"',
        'MMSR predict raises on no "task"',
        'MMSR predict raises on no "label"',
        'MMSR predict raises on no "worker"',
        'MMSR predict_score raises on no "task"',
        'MMSR predict_score raises on no "label"',
        'MMSR predict_score raises on no "worker"',
        'Wawa predict raises on no "task"',
        'Wawa predict raises on no "label"',
        'Wawa predict raises on no "worker"',
        'Wawa predict_proba raises on no "task"',
        'Wawa predict_proba raises on no "label"',
        'Wawa predict_proba raises on no "worker"',
        'ZBS predict raises on no "task"',
        'ZBS predict raises on no "label"',
        'ZBS predict raises on no "worker"',
        'ZBS predict_proba raises on no "task"',
        'ZBS predict_proba raises on no "label"',
        'ZBS predict_proba raises on no "worker"',
    ],
)
def test_agg_raise_on_less_columns(request: Any, agg_class: Any, predict_method: str,
                                   exception: Type[Exception], answers_dataset: str) -> None:
    """
    Tests all aggregation methods raises basik exceptions
    """
    answers = request.getfixturevalue(answers_dataset)
    aggregator = agg_class()
    with pytest.raises(exception):
        getattr(aggregator, predict_method)(answers)


@pytest.mark.parametrize(
    'exception, answers_on_gold_dataset',
    [
        # test raises in fit
        (KeyError, 'answers_no_task'),
        (KeyError, 'answers_no_label'),
        (KeyError, 'answers_no_worker'),
        (KeyError, 'gold_no_task'),
        (KeyError, 'gold_no_label'),
        # raises on mismatch datasets
        # TODO: check
        # (NotFittedError, 'toy_answers_on_gold_df_cannot_fit'),
    ],
    ids=[
        # test raises in fit
        'no "task" in answers_on_gold',
        'no "label" in answers_on_gold',
        'no "worker" in answers_on_gold',
        'no "task" in gold_df',
        'no "label" in gold_df',
        # raises on mismatch datasets
        # 'cannot compute workers skills',
    ],
)
def test_gold_mv_raise_in_fit(request: Any, not_random: Callable[[], None], toy_gold_df: pd.Series,
                              exception: Type[Exception], answers_on_gold_dataset: str) -> None:
    """
    Tests Gold MajorityVote on raises basik exceptions
    """
    answers_on_gold = request.getfixturevalue(answers_on_gold_dataset)

    aggregator = GoldMajorityVote()
    with pytest.raises(exception):
        aggregator.fit(answers_on_gold, toy_gold_df)


@pytest.mark.parametrize(
    'predict_method, exception, answers_on_gold_dataset, answers_dataset',
    [
        # test raises in predict
        ('predict', KeyError, 'toy_answers_df', 'answers_no_task'),
        ('predict', KeyError, 'toy_answers_df', 'answers_no_label'),
        ('predict', KeyError, 'toy_answers_df', 'answers_no_worker'),
        # test raises in predict_proba
        ('predict_proba', KeyError, 'toy_answers_df', 'answers_no_task'),
        ('predict_proba', KeyError, 'toy_answers_df', 'answers_no_label'),
        ('predict_proba', KeyError, 'toy_answers_df', 'answers_no_worker'),
        # raises on mismatch datasets
        # ('predict', NotFittedError, 'toy_answers_on_gold_df_cannot_predict', 'toy_answers_df'),
        # ('predict_proba', NotFittedError, 'toy_answers_on_gold_df_cannot_predict', 'toy_answers_df'),
    ],
    ids=[
        # test raises in predict
        'raise in predict on no "task" in answers_on_gold',
        'raise in predict on no "label" in answers_on_gold',
        'raise in predict on no "worker" in answers_on_gold',
        # test raises in predict_proba
        'raise in predict_proba on no "task" in answers_on_gold',
        'raise in predict_proba on no "label" in answers_on_gold',
        'raise in predict_proba on no "worker" in answers_on_gold',
        # raises on mismatch datasets
        # 'raise in predict - cannot compute labels',
        # 'raise in predict_proba - cannot compute probas',
    ],
)
def test_gold_mv_raise_in_predict(
    request: Any, not_random: Callable[[], None], toy_gold_df: pd.Series,
    predict_method: str, exception: Type[Exception], answers_on_gold_dataset: str, answers_dataset: str
) -> None:
    """
    Tests Gold MajorityVote on raises basic exceptions
    """
    answers_on_gold = request.getfixturevalue(answers_on_gold_dataset)
    answers = request.getfixturevalue(answers_dataset)

    aggregator = GoldMajorityVote()
    aggregator.fit(answers_on_gold, toy_gold_df)
    with pytest.raises(exception):
        getattr(aggregator, predict_method)(answers)


def test_gold_mv_empty() -> None:
    aggregator = GoldMajorityVote()
    probas = aggregator.fit_predict_proba(
        pd.DataFrame({'task': [], 'worker': [], 'label': []}),
        pd.Series(dtype=float)
    )
    assert probas.empty


@pytest.mark.parametrize(
    'agg_class',
    [MMSR, ZeroBasedSkill, DawidSkene, GLAD]
)
def test_zero_iter(agg_class: Any, simple_answers_df: pd.DataFrame,
                   simple_ground_truth: pd.Series) -> None:
    aggregator = agg_class(n_iter=0)
    answers = aggregator.fit_predict(simple_answers_df)
    assert len(answers.index.difference(simple_ground_truth.index)) == 0
