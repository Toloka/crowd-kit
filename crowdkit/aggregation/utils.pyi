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
import pandas
import typing


def evaluate_in(row: pandas.Series) -> int: ...


def evaluate_equal(row: pandas.Series) -> int: ...


def evaluate(
    df_true: pandas.DataFrame,
    df_pred: pandas.DataFrame,
    evaluate_func: typing.Callable[[pandas.Series], int] = evaluate_in
) -> typing.Union[str, float]: ...


def factorize(data: numpy.ndarray) -> typing.Tuple[numpy.ndarray, numpy.ndarray]: ...


def get_most_probable_labels(proba: pandas.DataFrame):
    """Returns most probable labels
    Args:
        proba (DataFrame): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """
    ...


def normalize_rows(scores: pandas.DataFrame) -> pandas.DataFrame:
    """Scales values so that every raw sums to 1
    Args:
        scores (DataFrame): Tasks' label scores.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the score of `label` for `task`.

    Returns:
        DataFrame: Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """
    ...


def manage_data(
    data: pandas.DataFrame,
    weights: typing.Optional[pandas.Series] = None,
    skills: pandas.Series = None
) -> pandas.DataFrame:
    """Args:
        data (DataFrame): Workers' labeling results.
            A pandas.DataFrame containing `task`, `worker` and `label` columns.
        skills (Series): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
    """
    ...


def get_accuracy(
    data: pandas.DataFrame,
    true_labels: pandas.Series,
    by: typing.Optional[str] = None
) -> pandas.Series:
    """Args:
        data (DataFrame): Workers' labeling results.
            A pandas.DataFrame containing `task`, `worker` and `label` columns.
        true_labels (Series): Tasks' ground truth labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's ground truth label.

    Returns:
        Series: workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
    """
    ...


def named_series_attrib(name: str):
    """Attrs attribute with converter and setter which preserves specified attribute name
    """
    ...


def add_skills_to_data(
    data: pandas.DataFrame,
    skills: pandas.Series,
    on_missing_skill: str,
    default_skill: float
) -> pandas.DataFrame:
    """Args:
        skills (Series): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
        on_missing_skill (str): How to handle assignments done by workers with unknown skill.
            Possible values:
                    * "error" — raise an exception if there is at least one assignment done by user with unknown skill;
                    * "ignore" — drop assignments with unknown skill values during prediction. Raise an exception if there is no 
                    assignments with known skill for any task;
                    * value — default value will be used if skill is missing.
    """
    ...
