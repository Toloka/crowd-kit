__all__ = ['BaseAggregator']

import random
from typing import Union, Tuple

import attr
import pandas as pd

from . import annotations
from .annotations import manage_docstring


@attr.attrs(auto_attribs=True)
@manage_docstring
class BaseAggregator:
    """Base functions and fields for all aggregators"""

    tasks_labels: annotations.OPTIONAL_CLASSLEVEL_TASKS_LABELS = None
    probas: annotations.OPTIONAL_CLASSLEVEL_PROBAS = None
    performers_skills: annotations.OPTIONAL_CLASSLEVEL_PERFORMERS_SKILLS = None

    @staticmethod
    def _max_probas_random_on_ties(x: Union[pd.DataFrame, pd.Series]) -> Tuple[str, float]:
        """Chooses max 'proba' value and return 'label' from same rows
        If several rows have same 'proba' - choose random
        """
        max_proba = x.proba.max()
        max_label_index = random.choice(x[x.proba==max_proba].index)
        return x.label[max_label_index], max_proba

    @manage_docstring
    def _calculate_probabilities(self, estimated_answers: pd.DataFrame) -> annotations.PROBAS:
        """Calculate probabilities for each task for each label

        Note:
            All "score" must be positive.
            If the sum of scores for a task is zero, then all probabilities for this task will be NaN.

        Args:
            estimated_answers(pandas.DataFrame): Frame with "score" for each pair task-label.
                Should contain columns 'score', 'task', 'label'

        """
        assert (estimated_answers.score >= 0).all(), 'In answers exists some "score" with negative value'

        estimated_answers['proba'] = estimated_answers.score / estimated_answers.groupby('task').score.transform('sum')
        self.probas = estimated_answers.pivot(index='task', columns='label', values='proba')
        return self.probas

    @manage_docstring
    def _choose_labels(self, labels_probas: annotations.PROBAS) -> annotations.TASKS_LABELS:
        """Selection of the labels with the most probalitities"""
        self.tasks_labels = labels_probas.idxmax(axis="columns").reset_index(name='label')
        return self.tasks_labels

    @manage_docstring
    def _calc_performers_skills(self, answers: pd.DataFrame, task_truth: pd.DataFrame) -> annotations.PERFORMERS_SKILLS:
        """Calculates skill for each performer

        Note:
            There can be only one * correct label *

        Args:
            answers (pandas.DataFrame): performers answers for tasks
                Should contain columns 'task', 'performer', 'label'
            task_truth (pandas.DataFrame): label regarding which to count the skill
                Should contain columns 'task', 'label'
                Could contain column 'weight'
        """
        def _agreed_on_task(x):
            """Calculates performers agreed for each based on:
            - result label in 'task_truth',
            - performer label in 'answers',
            - and 'weight' if it's exist
            """
            return int(x['label'] == x['label_truth']) * x.get('weight', 1)

        answers_with_results = answers.merge(task_truth, on='task', suffixes=('', '_truth'))
        answers_with_results['skill'] = answers_with_results.apply(_agreed_on_task, axis=1)
        self.performers_skills = answers_with_results.groupby('performer')['skill'].agg('mean').reset_index()
        return self.performers_skills

    def _answers_base_checks(self, answers: pd.DataFrame) -> None:
        """Checks basic 'answers' dataset requirements"""
        if not isinstance(answers, pd.DataFrame):
            raise TypeError('Working only with pandas DataFrame')
        assert 'task' in answers, 'There is no "task" column in answers'
        assert 'performer' in answers, 'There is no "performer" column in answers'
        assert 'label' in answers, 'There is no "label" column in answers'
