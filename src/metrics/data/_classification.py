from typing import Any, Optional, Union
import pandas as pd

from crowdkit.aggregation.base_aggregator import BaseAggregator
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
                aggregator: BaseAggregator = MajorityVote(),
                by_task: bool = False) -> Union[float, pd.Series]:
    """
    Consistency metric: posterior probability of aggregated label given performers skills
    calculated using standard Dawid-Skene model.
    Args:
            answers (pandas.DataFrame): A data frame containing `task`, `performer` and `label` columns.
            performers_skills (Optional[pandas.Series]): performers skills e.g. golden set skills. If not provided,
                uses aggregator's `performers_skills` attribute.
            aggregator (aggregation.BaseAggregator): aggregation method, default: MajorityVote
            by_task (bool): if set, returns consistencies for every task in provided data frame.

        Returns:
            Union[float, pd.Series]
    """
    _check_answers(answers)
    aggregated = aggregator.fit_predict(answers)
    if performers_skills is None and hasattr(aggregator, 'skills_'):
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
