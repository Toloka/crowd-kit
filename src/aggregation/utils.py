from typing import Union, Callable, Optional

import numpy as np
import pandas as pd

from . import annotations
from .annotations import manage_docstring


def _argmax_random_ties(array: np.ndarray) -> int:
    # Returns the index of the maximum element
    # If there are several such elements, it returns a random one
    return int(np.random.choice(np.flatnonzero(array == array.max())))


def evaluate_in(row: pd.Series) -> int:
    return int(row['label_pred'] in row['label_true'])


def evaluate_equal(row: pd.Series) -> int:
    return int(row['label_pred'] == row['label_true'])


def evaluate(df_true: pd.DataFrame, df_pred: pd.DataFrame,
             evaluate_func: Callable[[pd.Series], int] = evaluate_in) -> Union[str, float]:
    df = df_true.merge(df_pred, on='task', suffixes=('_true', '_pred'))

    assert len(df_true) == len(df), f'Dataset length mismatch, expected {len(df_true):d}, got {len(df):d}'

    df['evaluation'] = df.apply(evaluate_func, axis=1)
    return float(df['evaluation'].mean())


@manage_docstring
def get_most_probable_labels(proba: annotations.TASKS_LABEL_PROBAS):
    """Returns most probable labels"""
    return proba.idxmax(axis='columns')


@manage_docstring
def normalize_rows(scores: annotations.TASKS_LABEL_SCORES) -> annotations.TASKS_LABEL_PROBAS:
    """Scales values so that every raw sums to 1"""
    return scores.div(scores.sum(axis=1), axis=0)


@manage_docstring
def manage_data(data: annotations.LABELED_DATA, weights: Optional[pd.Series] = None, skills: annotations.SKILLS = None) -> pd.DataFrame:
    data = data[['task', 'performer', 'label']]

    if weights is None:
        data['weight'] = 1
    else:
        data = data.join(weights.rename('weight'), on='task')

    if skills is None:
        data['skill'] = 1
    else:
        data = data.join(skills.rename('skill'), on='task')

    return data


@manage_docstring
def get_accuracy(data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS, by: str = None) -> annotations.SKILLS:
    if 'weight' in data.columns:
        data = data[['task', 'performer', 'label', 'weight']]
    else:
        data = data[['task', 'performer', 'label']]
        data['weight'] = 1

    data = data.join(pd.Series(true_labels, name='true_label'), on='task')
    data = data[data.true_label.notna()]
    data.eval('score = weight * (label == true_label)', inplace=True)

    if by is not None:
        data = data.groupby(by)

    return data.score.sum() / data.weight.sum()
