from typing import Any

import pandas as pd
import pytest

from crowdkit.aggregation import SegmentationEM, SegmentationMajorityVote, SegmentationRASA
from pandas.testing import assert_series_equal

from .data_image import *  # noqa:


def test_simple_segmentation_mv(simple_image_df: pd.DataFrame, simple_image_mv_result: pd.Series) -> None:
    output = SegmentationMajorityVote().fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_mv_result)


def test_skills_segmentation_mv(image_with_skills_df: pd.DataFrame, image_with_skills_mv_result: pd.Series) -> None:
    output = SegmentationMajorityVote().fit_predict(*image_with_skills_df)
    assert_series_equal(output, image_with_skills_mv_result)


@pytest.mark.parametrize(
    'n_iter, tol', [(10, 0), (100500, 1e-5)]
)
def test_simple_segmentation_rasa_iter(n_iter: int, tol: float, simple_image_df: pd.DataFrame,
                                       simple_image_rasa_result: pd.Series) -> None:
    output = SegmentationRASA(n_iter=n_iter, tol=tol).fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_rasa_result)


@pytest.mark.parametrize(
    'n_iter, tol', [(10, 0), (100500, 1e-5)]
)
def test_simple_segmentation_em_iter(n_iter: int, tol: float, simple_image_df: pd.DataFrame,
                                     simple_image_em_result: pd.Series) -> None:
    output = SegmentationEM(n_iter=n_iter, tol=tol).fit_predict(simple_image_df)
    assert_series_equal(output, simple_image_em_result)


@pytest.mark.parametrize(
    'agg_class', [SegmentationEM, SegmentationRASA]
)
def test_zero_iter(agg_class: Any, simple_image_df: pd.DataFrame,
                   simple_image_mv_result: pd.Series) -> None:
    aggregator = agg_class(n_iter=0)
    answers = aggregator.fit_predict(simple_image_df)
    assert len(answers.index.difference(simple_image_mv_result.index)) == 0
