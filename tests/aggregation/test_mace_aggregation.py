"""
Simple aggregation tests.
"""
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import pytest

from crowdkit.aggregation import MACE
from crowdkit.aggregation.utils import evaluate, evaluate_equal


@pytest.mark.parametrize(
    'method', ['vb', 'em']
)
def test_aggregate_mace_on_toy_ysda(
        method: str,
        toy_answers_df: pd.DataFrame,
        toy_ground_truth_df: pd.Series
) -> None:
    np.random.seed(42)
    predict_df = MACE(n_restarts=1, n_iter=5, method=method).fit_predict(toy_answers_df)
    accuracy = evaluate(
        toy_ground_truth_df.to_frame('label'),
        predict_df.to_frame('label'),
        evaluate_func=evaluate_equal
    )
    assert accuracy == 1.0


@pytest.mark.parametrize(
    'method', ['vb', 'em']
)
def test_aggregate_mace_on_simple(
        method: str,
        simple_answers_df: pd.DataFrame,
        simple_ground_truth: pd.Series
) -> None:
    np.random.seed(42)
    predict_df = MACE(n_restarts=1, n_iter=5, method=method).fit_predict(simple_answers_df)
    accuracy = evaluate(
        simple_ground_truth.to_frame('label'),
        predict_df.to_frame('label'),
        evaluate_func=evaluate_equal
    )
    if method == 'vb':
        assert accuracy == 1.0
    else:
        assert accuracy == 0.9


@pytest.fixture
def thetas_simple() -> NDArray[np.float_]:
    return np.array(
        [
            [0.27249486, 0.27249486, 0.42663689],
            [0.32260257, 0.3225825, 0.32139492],
            [0.32187643, 0.32235373, 0.32234894],
            [0.32823984, 0.3184931, 0.32028815],
            [0.32156453, 0.3225007, 0.32249734],
            [0.29468813, 0.35241372, 0.32222609]
        ]
    )


def test_thetas_on_simple(simple_answers_df: pd.DataFrame, thetas_simple: pd.DataFrame) -> None:
    thetas = MACE(n_restarts=1, n_iter=5, method='vb', random_state=0).fit(simple_answers_df).thetas_
    assert np.allclose(thetas, thetas_simple)
