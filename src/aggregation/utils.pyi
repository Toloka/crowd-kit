from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import (
    Callable,
    Optional,
    Tuple,
    Union
)

def evaluate_in(row: Series) -> int: ...


def evaluate_equal(row: Series) -> int: ...


def evaluate(df_true: DataFrame, df_pred: DataFrame, evaluate_func: Callable[Series, int] = ...) -> Union[str, float]: ...


def factorize(data: ndarray) -> Tuple[ndarray, ndarray]: ...


def get_most_probable_labels(proba: DataFrame):
    """Returns most probable labels
    Args:
        proba (DataFrame): Tasks' label probability distributions
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """
    ...


def normalize_rows(scores: DataFrame) -> DataFrame:
    """Scales values so that every raw sums to 1
    Args:
        scores (DataFrame): Tasks' label scores
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the score of `label` for `task`.

    Returns:
        DataFrame: Tasks' label probability distributions
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """
    ...


def manage_data(data: DataFrame, weights: Optional[Series] = None, skills: Series = None) -> DataFrame:
    """Args:
        data (DataFrame): Performers' labeling results
            A pandas.DataFrame containing `task`, `performer` and `label` columns.
        skills (Series): Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
    """
    ...


def get_accuracy(data: DataFrame, true_labels: Series, by: str = None) -> Series:
    """Args:
        data (DataFrame): Performers' labeling results
            A pandas.DataFrame containing `task`, `performer` and `label` columns.
        true_labels (Series): Tasks' ground truth labels
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's ground truth label.

    Returns:
        Series: Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
    """
    ...
