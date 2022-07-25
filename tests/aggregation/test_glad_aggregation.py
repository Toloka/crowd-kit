"""
Simplest aggregation algorithms tests on toy YSDA dataset
Testing all boundary conditions and asserts
"""
from typing import Tuple, Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from crowdkit.aggregation import GLAD
from crowdkit.aggregation.utils import evaluate, evaluate_equal


@pytest.mark.parametrize(
    'n_iter, tol', [(100, 0), (100500, 1e-5)]
)
def test_aggregate_glad_on_toy_ysda(n_iter: int, tol: float, toy_answers_df: pd.DataFrame,
                                    toy_ground_truth_df: pd.Series) -> None:
    np.random.seed(42)
    predict_df = GLAD(n_iter=n_iter, tol=tol).fit_predict(toy_answers_df)
    accuracy = evaluate(
        toy_ground_truth_df.to_frame('label'),
        predict_df.to_frame('label'),
        evaluate_func=evaluate_equal
    )
    assert accuracy == 1.0


@pytest.mark.parametrize(
    'n_iter, tol', [(100, 0), (100500, 1e-5)]
)
def test_aggregate_glad_on_simple(n_iter: int, tol: float, simple_answers_df: pd.DataFrame,
                                  simple_ground_truth: pd.Series) -> None:
    np.random.seed(42)
    predict_df = GLAD(n_iter=n_iter, tol=tol).fit_predict(simple_answers_df)
    accuracy = evaluate(
        simple_ground_truth.to_frame('label'),
        predict_df.to_frame('label'),
        evaluate_func=evaluate_equal
    )
    assert accuracy == 1.0


@pytest.fixture
def single_task_simple_df(simple_answers_df: pd.DataFrame) -> pd.DataFrame:
    return simple_answers_df[simple_answers_df['task'] == '1231239876--5fac0d234ffb2f3b00893eec']


@pytest.fixture
def single_task_simple_df_e_step_probas() -> npt.NDArray[np.float_]:
    return np.array([[0.995664, 0.004336]])


@pytest.fixture
def single_task_initialized_glad(single_task_simple_df: pd.DataFrame) -> Tuple[pd.DataFrame, GLAD]:
    glad = GLAD()
    glad._init(single_task_simple_df)
    data = glad._join_all(single_task_simple_df, glad.alphas_, glad.betas_, glad.priors_)
    return glad._e_step(data), glad


def test_glad_e_step(single_task_initialized_glad: Tuple[pd.DataFrame, GLAD],
                     single_task_simple_df_e_step_probas: pd.DataFrame) -> None:
    data, glad = single_task_initialized_glad
    assert np.allclose(glad.probas_.values, single_task_simple_df_e_step_probas, atol=1e-6)  # type: ignore


def test_glad_derivative(single_task_initialized_glad: Tuple[pd.DataFrame, GLAD]) -> None:
    data, glad = single_task_initialized_glad
    glad._current_data = data
    x_0 = np.concatenate([glad.alphas_.values, glad.betas_.values])  # type: ignore

    def Q_by_alpha_beta(x: npt.NDArray[Any]) -> float:
        glad._update_alphas_betas(*glad._get_alphas_betas_by_point(x))
        new_Q = glad._compute_Q(glad._current_data)
        glad._update_alphas_betas(*glad._get_alphas_betas_by_point(x_0))
        return new_Q

    eps = np.sqrt(np.finfo(float).eps)
    numerical_grad = np.sort(approx_fprime(x_0, Q_by_alpha_beta, eps))
    dQalpha, dQbeta = glad._gradient_Q(data)
    analytical_grad = np.sort(np.concatenate([dQalpha.values, dQbeta.values]))  # type: ignore
    assert np.allclose(analytical_grad, numerical_grad)
