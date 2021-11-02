from typing import Any, Callable, Hashable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from nltk.metrics.agreement import AnnotationTask
from nltk.metrics.distance import binary_distance

from crowdkit.aggregation.base import BaseClassificationAggregator
from crowdkit.aggregation import MajorityVote


def _check_answers(answers: pd.DataFrame) -> None:
    if not isinstance(answers, pd.DataFrame):
        raise TypeError('Working only with pandas DataFrame')
    assert 'task' in answers, 'There is no "task" column in answers'
    assert 'performer' in answers, 'There is no "performer" column in answers'
    assert 'label' in answers, 'There is no "label" column in answers'


def _label_probability(row: pd.Series, label: Any, n_labels: int) -> float:
    """Numerator in the Bayes formula"""
    return row['skill'] if row['label'] == label else (1.0 - row['skill']) / (n_labels - 1)


def _task_consistency(row: pd.Series) -> float:
    """Posterior probability for a single task"""
    return row[row['aggregated_label']] / row['denominator'] if row['denominator'] != 0 else 0.0


def consistency(answers: pd.DataFrame,
                performers_skills: Optional[pd.Series] = None,
                aggregator: BaseClassificationAggregator = MajorityVote(),
                by_task: bool = False) -> Union[float, pd.Series]:
    """
    Consistency metric: posterior probability of aggregated label given performers skills
    calculated using standard Dawid-Skene model.
    Args:
        answers (pandas.DataFrame): A data frame containing `task`, `performer` and `label` columns.
        performers_skills (Optional[pandas.Series]): performers skills e.g. golden set skills. If not provided,
            uses aggregator's `performers_skills` attribute.
        aggregator (aggregation.base.BaseClassificationAggregator): aggregation method, default: MajorityVote
        by_task (bool): if set, returns consistencies for every task in provided data frame.

    Returns:
        Union[float, pd.Series]
    """
    _check_answers(answers)
    aggregated = aggregator.fit_predict(answers)
    if performers_skills is None:
        if hasattr(aggregator, 'skills_'):
            performers_skills = aggregator.skills_
        else:
            raise AssertionError('This aggregator is not supported. Please, provide performers skills.')

    answers = answers.copy(deep=False)
    answers.set_index('task', inplace=True)
    answers = answers.reset_index().set_index('performer')
    answers['skill'] = performers_skills
    answers.reset_index(inplace=True)
    labels = pd.unique(answers.label)
    for label in labels:
        answers[label] = answers.apply(lambda row: _label_probability(row, label, len(labels)), axis=1)
    labels_proba = answers.groupby('task').prod()
    labels_proba['aggregated_label'] = aggregated
    labels_proba['denominator'] = labels_proba[list(labels)].sum(axis=1)
    consistecies = labels_proba.apply(_task_consistency, axis=1)

    if by_task:
        return consistecies
    else:
        return consistecies.mean()


def _task_uncertainty(row: pd.Series, labels: List[Hashable]) -> float:
    if row['denominator'] == 0:
        row[labels] = 1 / len(labels)
    else:
        row[labels] /= row['denominator']
    softmax = row[labels]
    log_softmax = np.log(row[list(labels)])
    return -np.sum(softmax * log_softmax)


def uncertainty(answers: pd.DataFrame,
                performers_skills: pd.Series,
                by_task: bool = False) -> Union[float, pd.Series]:
    """
    Label uncertainty metric: entropy of labels probability distribution.
    Args:
        answers (pandas.DataFrame): A data frame containing `task`, `performer` and `label` columns.
        performers_skills (pandas.Series): performers skills e.g. golden set skills. If not provided,
            uses aggregator's `performers_skills` attribute.
        by_task (bool): if set, returns consistencies for every task in provided data frame.

    Returns:
        Union[float, pd.Series]
    """
    _check_answers(answers)
    answers = answers.copy(deep=False)
    answers.set_index('task', inplace=True)
    answers = answers.reset_index().set_index('performer')
    answers['skill'] = performers_skills
    answers.reset_index(inplace=True)
    labels = pd.unique(answers.label)
    for label in labels:
        answers[label] = answers.apply(lambda row: _label_probability(row, label, len(labels)), axis=1)
    labels_proba = answers.groupby('task').prod()
    labels_proba['denominator'] = labels_proba[list(labels)].sum(axis=1)
    uncertainties = labels_proba.apply(lambda row: _task_uncertainty(row, list(labels)), axis=1)

    if by_task:
        return uncertainties
    else:
        return uncertainties.mean()


def alpha_krippendorff(answers: pd.DataFrame,
                       distance: Callable[[Hashable, Hashable], float] = binary_distance) -> float:
    """Inter-annotator agreement coefficient (Krippendorff 1980).

    Amount that annotators agreed on label assignments beyond what is expected by chance.
    The value of alpha should be interpreted as follows.
        alpha >= 0.8 indicates a reliable annotation,
        alpha >= 0.667 allows making tentative conclusions only,
        while the lower values suggest the unreliable annotation.

    Args:
        answers: A data frame containing `task`, `performer` and `label` columns.
        distance: Distance metric, that takes two arguments,
            and returns a value between 0.0 and 1.0
            By default: binary_distance (0.0 for equal labels 1.0 otherwise).

    Returns:
        Float value.

    Examples:
        Consistent answers.

        >>> alpha_krippendorff(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'performer': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'performer': 'B', 'label': 'Yes'},
        >>>     {'task': 'Y', 'performer': 'A', 'label': 'No'},
        >>>     {'task': 'Y', 'performer': 'B', 'label': 'No'},
        >>> ]))
        1.0

        Partially inconsistent answers.

        >>> alpha_krippendorff(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'performer': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'performer': 'B', 'label': 'Yes'},
        >>>     {'task': 'Y', 'performer': 'A', 'label': 'No'},
        >>>     {'task': 'Y', 'performer': 'B', 'label': 'No'},
        >>>     {'task': 'Z', 'performer': 'A', 'label': 'Yes'},
        >>>     {'task': 'Z', 'performer': 'B', 'label': 'No'},
        >>> ]))
        0.4444444444444444
    """
    _check_answers(answers)
    data: List[Tuple[Any, Hashable, Hashable]] = answers[['performer', 'task', 'label']].values.tolist()
    return AnnotationTask(data, distance).alpha()
