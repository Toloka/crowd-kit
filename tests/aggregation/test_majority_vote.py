import pytest
import pandas as pd
from typing import Tuple
from pandas.testing import assert_series_equal
from crowdkit.aggregation import MajorityVote, SegmentationMajorityVote, GoldMajorityVote

from .data_mv import simple_skills_result_mv  # noqa F401
from .data_gold_mv import multiple_gt_gt, multiple_gt_df, multiple_gt_skills, multiple_gt_aggregated  # noqa F401
from .data_image import image_with_skills_df, image_with_skills_mv_result  # noqa F401


def test_majority_vote_with_skills(simple_answers_df: pd.DataFrame,
                                   simple_skills_result_mv: pd.Series) -> None:  # noqa F811
    mv = MajorityVote()
    mv.fit_predict(simple_answers_df, skills=simple_skills_result_mv)
    assert_series_equal(mv.skills_, simple_skills_result_mv)


def test_majority_vote_with_missing_skills_value(simple_answers_df: pd.DataFrame,
                                   simple_skills_result_mv: pd.Series) -> None:  # noqa F811
    mv = MajorityVote(on_missing_skill='value', default_skill=1000000)
    simple_skills_result_mv = simple_skills_result_mv.drop('0c3eb7d5fcc414db137c4180a654c06e')
    mv.fit_predict(simple_answers_df, skills=simple_skills_result_mv)

    frauded_tasks = simple_answers_df[simple_answers_df['worker'] == '0c3eb7d5fcc414db137c4180a654c06e']['task']
    for task in frauded_tasks:
        assert mv.labels_[task] == 'parrot'  # type: ignore


def test_majority_vote_with_missing_skills_error(simple_answers_df: pd.DataFrame,
                                   simple_skills_result_mv: pd.Series) -> None:  # noqa F811
    mv = MajorityVote(on_missing_skill='error')
    simple_skills_result_mv = simple_skills_result_mv.drop('0c3eb7d5fcc414db137c4180a654c06e')
    with pytest.raises(ValueError):
        mv.fit_predict(simple_answers_df, skills=simple_skills_result_mv)


def test_majority_vote_with_missing_skills_ignore(simple_answers_df: pd.DataFrame,
                                   simple_skills_result_mv: pd.Series) -> None:  # noqa F811
    mv = MajorityVote(on_missing_skill='ignore')
    simple_skills_result_mv = simple_skills_result_mv.drop('0c3eb7d5fcc414db137c4180a654c06e')
    mv.fit_predict(simple_answers_df, skills=simple_skills_result_mv)
    assert '0c3eb7d5fcc414db137c4180a654c06e' not in mv.skills_.index  # type: ignore


def test_majority_vote_with_missing_skills_ignore_all(simple_answers_df: pd.DataFrame,
                                   simple_skills_result_mv: pd.Series) -> None:  # noqa F811
    mv = MajorityVote(on_missing_skill='ignore')
    with pytest.raises(ValueError):
        mv.fit_predict(simple_answers_df, skills=pd.Series([], dtype=float))


def test_segmentation_majority_vote_with_missing_skills_value(image_with_skills_df: Tuple[pd.DataFrame, pd.Series],  # noqa F811
                                                              image_with_skills_mv_result: pd.Series) -> None:  # noqa F811
    answers_df, skills = image_with_skills_df
    mv = SegmentationMajorityVote(on_missing_skill='value', default_skill=3)
    skills = skills.drop('e044b0849dfa9ce3dee5debbefb3b5da')
    assert_series_equal(mv.fit_predict(answers_df, skills=skills), image_with_skills_mv_result)


def test_segmentation_majority_vote_with_missing_skills_error(image_with_skills_df: Tuple[pd.DataFrame, pd.Series]) -> None:  # noqa F811
    answers_df, skills = image_with_skills_df
    mv = SegmentationMajorityVote(on_missing_skill='error', default_skill=3)
    skills = skills.drop('e044b0849dfa9ce3dee5debbefb3b5da')
    with pytest.raises(ValueError):
        mv.fit_predict(answers_df, skills=skills)


def test_segmentation_majority_vote_with_missing_skills_ignore(image_with_skills_df: Tuple[pd.DataFrame, pd.Series]) -> None:  # noqa F811
    answers_df, skills = image_with_skills_df
    mv = SegmentationMajorityVote(on_missing_skill='ignore')
    skills = skills.drop(['e044b0849dfa9ce3dee5debbefb3b5da', 'be37db5784b50d08d2702f36317a3074'])

    assert len(mv.fit_predict(answers_df, skills=skills)) == 1


def test_segmentation_majority_vote_with_missing_skills_ignore_all(image_with_skills_df: Tuple[pd.DataFrame, pd.Series]) -> None:  # noqa F811
    answers_df, skills = image_with_skills_df
    mv = SegmentationMajorityVote(on_missing_skill='ignore')
    with pytest.raises(ValueError):
        mv.fit_predict(answers_df, skills=pd.Series([], dtype=float))


def test_gold_mv_multiple_gt(multiple_gt_df: pd.DataFrame, multiple_gt_gt: pd.Series,  # noqa F811
                             multiple_gt_skills: pd.Series, multiple_gt_aggregated: pd.Series) -> None:  # noqa F811
    gmv = GoldMajorityVote()
    aggregated = gmv.fit_predict(multiple_gt_df, true_labels=multiple_gt_gt)
    skills = gmv.skills_
    assert_series_equal(skills, multiple_gt_skills)
    assert_series_equal(aggregated, multiple_gt_aggregated)
