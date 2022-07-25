from typing import Any

import pandas as pd
import pytest

from crowdkit.aggregation import RASA, HRRASA
from pandas.testing import assert_frame_equal
from .data_hrrasa import *  # noqa:


@pytest.mark.parametrize(
    'n_iter, tol', [(100, 0), (100500, 1e-9)]
)
def test_rasa(n_iter: int, tol: float, simple_text_df: pd.DataFrame, simple_text_true_embeddings: pd.Series,
              simple_text_result_rasa: pd.DataFrame) -> None:
    output = RASA(n_iter=n_iter, tol=tol).fit_predict(simple_text_df, simple_text_true_embeddings)
    assert_frame_equal(output, simple_text_result_rasa)


@pytest.mark.parametrize(
    'n_iter, tol', [(100, 0), (100500, 1e-9)]
)
def test_hrrasa(n_iter: int, tol: float, simple_text_df: pd.DataFrame, simple_text_true_embeddings: pd.Series,
                simple_text_result_hrrasa: pd.DataFrame) -> None:
    output = HRRASA(n_iter=n_iter, tol=tol).fit_predict(simple_text_df, simple_text_true_embeddings)
    assert_frame_equal(output, simple_text_result_hrrasa)


def test_hrrasa_single_overlap(simple_text_df: pd.DataFrame) -> None:
    hrrasa = HRRASA()
    output = hrrasa.fit_predict(simple_text_df[:1])
    assert len(output) == 1


@pytest.mark.parametrize(
    'agg_class', [RASA, HRRASA]
)
def test_zero_iter(agg_class: Any, simple_text_df: pd.DataFrame, simple_text_true_embeddings: pd.Series,
                   simple_text_result_hrrasa: pd.DataFrame) -> None:
    aggregator = agg_class(n_iter=0)
    answers = aggregator.fit_predict(simple_text_df, simple_text_true_embeddings)
    assert len(answers.index.difference(simple_text_result_hrrasa.index)) == 0
