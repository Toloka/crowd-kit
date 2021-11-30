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
import numpy
import pandas.core.frame
import pandas.core.series
import typing


def evaluate_in(row: pandas.core.series.Series) -> int: ...


def evaluate_equal(row: pandas.core.series.Series) -> int: ...


def evaluate(
    df_true: pandas.core.frame.DataFrame,
    df_pred: pandas.core.frame.DataFrame,
    evaluate_func: typing.Callable[[pandas.core.series.Series], int] = ...
) -> typing.Union[str, float]: ...


def factorize(data: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]: ...


def get_most_probable_labels(proba: pandas.core.frame.DataFrame):
    """Returns most probable labels
    Args:
        proba (DataFrame): Tasks' label probability distributions
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """
    ...


def normalize_rows(scores: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
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


def manage_data(
    data: pandas.core.frame.DataFrame,
    weights: typing.Optional[pandas.core.series.Series] = None,
    skills: pandas.core.series.Series = None
) -> pandas.core.frame.DataFrame:
    """Args:
        data (DataFrame): Performers' labeling results
            A pandas.DataFrame containing `task`, `performer` and `label` columns.
        skills (Series): Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
    """
    ...


def get_accuracy(
    data: pandas.core.frame.DataFrame,
    true_labels: pandas.core.series.Series,
    by: typing.Optional[str] = None
) -> pandas.core.series.Series:
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


def named_series_attrib(name: str):
    """Attrs attribute with converter and setter which preserves specified attribute name
    """
    ...


def add_skills_to_data(
    data: pandas.core.frame.DataFrame,
    skills: pandas.core.series.Series,
    on_missing_skill: str,
    default_skill: float
) -> pandas.core.frame.DataFrame:
    """Args:
        skills (Series): Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
        on_missing_skill (str): How to handle assignments done by workers with unknown skill
            Possible values:
                    * "error" — raise an exception if there is at least one assignment done by user with unknown skill;
                    * "ignore" — drop assignments with unknown skill values during prediction. Raise an exception if there is no 
                    assignments with known skill for any task;
                    * value — default value will be used if skill is missing.
    """
    ...
