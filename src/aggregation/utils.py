__all__ = [
    'evaluate_in',
    'evaluate_equal',
    'evaluate',
    'factorize',
    'get_most_probable_labels',
    'normalize_rows',
    'manage_data',
    'get_accuracy',
    'add_skills_to_data',
    'named_series_attrib',
]
from typing import Tuple, Union, Callable, Optional

import attr
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


def factorize(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    unique_values, coded = np.unique(data, return_inverse=True)
    return unique_values, coded.reshape(data.shape)


@manage_docstring
def get_most_probable_labels(proba: annotations.TASKS_LABEL_PROBAS):
    """Returns most probable labels"""
    # patch for pandas<=1.1.5
    if not proba.size:
        return pd.Series([], dtype='O')
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
def get_accuracy(data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS, by: Optional[str] = None) -> annotations.SKILLS:
    if 'weight' in data.columns:
        data = data[['task', 'performer', 'label', 'weight']]
    else:
        data = data[['task', 'performer', 'label']]

    if data.empty:
        data['true_label'] = []
    else:
        data = data.join(pd.Series(true_labels, name='true_label'), on='task')

    data = data[data.true_label.notna()]

    if 'weight' not in data.columns:
        data['weight'] = 1

    data.eval('score = weight * (label == true_label)', inplace=True)

    if by is not None:
        data = data.groupby(by)

    return data.score.sum() / data.weight.sum()


def named_series_attrib(name: str):
    """Attrs attribute with converter and setter which preserves specified attribute name"""

    def converter(series: pd.Series) -> pd.Series:
        series.name = name
        return series

    return attr.ib(init=False, converter=converter, on_setattr=attr.setters.convert)


@manage_docstring
def add_skills_to_data(
    data: pd.DataFrame,
    skills: annotations.SKILLS,
    on_missing_skill: annotations.ON_MISSING_SKILL,
    default_skill: float
) -> pd.DataFrame:

    data = data.join(skills.rename('skill'), on='performer')

    if on_missing_skill != 'value' and default_skill is not None:
        raise ValueError('default_skill is used but on_missing_skill is not "value"')

    if on_missing_skill == 'error':
        missing_skills_count = data['skill'].isna().sum()
        if missing_skills_count > 0:
            raise ValueError(
                f"Skill value is missing in {missing_skills_count} assignments. Specify skills for every"
                f"used worker or use different 'on_unknown_skill' value."
            )
    elif on_missing_skill == 'ignore':
        data.set_index('task', inplace=True)
        index_before_drop = data.index
        data.dropna(inplace=True)
        dropped_tasks_count = len(index_before_drop.difference(data.index))
        if dropped_tasks_count > 0:
            raise ValueError(
                f"{dropped_tasks_count} tasks has no workers with known skills. Provide at least one worker with known"
                f"skill for every task or use different 'on_unknown_skill' value."
            )
        data.reset_index(inplace=True)
    elif on_missing_skill == 'value':
        if default_skill is None:
            raise ValueError('Default skill value must be specified when using on_missing_skill="value"')
        data.loc[data['skill'].isna(), 'skill'] = default_skill
    else:
        raise ValueError(f'Unknown option {on_missing_skill!r} of "on_missing_skill" argument.')
    return data
