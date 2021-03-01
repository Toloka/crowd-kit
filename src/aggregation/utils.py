from typing import Union, Callable

import pandas as pd
import numpy as np


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
