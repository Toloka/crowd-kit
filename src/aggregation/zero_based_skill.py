import attr
import numpy as np
import pandas as pd
from typing import Any, Optional, Set

from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote


@attr.attrs(auto_attribs=True)
class LRScheduler:

    steps_to_reduce: int = 20
    reduce_rate: float = 2.0
    _lr: Optional[float] = None
    _steps: int = 1

    def step(self) -> float:
        if self._lr is None:
            raise AssertionError('step called before setting learning rate')
        if self._steps % self.steps_to_reduce == 0:
            self._lr /= self.reduce_rate
        return self._lr

    def reset(self) -> None:
        self._steps = 1


@attr.attrs(auto_attribs=True)
class ZeroBasedSkill(BaseAggregator):
    """The Zero-Based Skill aggregation model

    Performs weighted majority voting on tasks. After processing a pool of tasks,
    reestimates performers' skills according to the correctness of their answers.
    Repeats this process until labels do not change or the number of iterations exceeds.

    It's necessary that all performers in a dataset that send to 'predict' existed in answers
    the dataset that was sent to 'fit'.

    After fit stored 'performers_skills' - estimated skills of performers.

    After predicting stored different data frames (details in BaseAggregator):
        tasks_labels: Predicted labels for each task
        probas: Probabilities for each label for the task
    """

    lr: float = 1e-1
    n_iter: int = 100
    performers_skills: Optional[pd.DataFrame] = None
    labels_set: Optional[Set[Any]] = None
    num_labels: Optional[int] = None
    early_stopping: int = 3
    eps: float = 1e-5
    lr_scheduler = LRScheduler()

    def fit(self, answers: pd.DataFrame) -> 'ZeroBasedSkill':
        """Calculates the skill for each performers, based on answers on provided dataset
        The calculated skills are stored in an instance of the class and can be obtained by the field 'performers_skills'
        After 'fit' you can get 'performer_skills' from class field.

        Args:
            answers(pandas.DataFrame): Frame contains performers answers. One row per answer.
                Should contain columns 'performer', 'task', 'label'.
        Returns:
            ZeroBasedSkill: self for call next methods

        Raises:
            TypeError: If the input datasets are not of type pandas.DataFrame.
            AssertionError: If there is some collumn missing in 'dataframes'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        return self

    def predict(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Predict correct labels for tasks. Using calculated performers skill, stored in self instance.
        After 'predict' you can get probabilities for all labels from class field 'probas'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Predicted label for each task.
                - task - unique values from input dataset
                - label - most likely label

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers' in 'fit'.
        """
        self._answers_base_checks(answers)
        self._predict_impl(answers)
        return self.tasks_labels

    def predict_proba(self, answers) -> pd.DataFrame:
        """Calculates Probabilities for each label of task.
        After 'predict_proba' you can get predicted labels from class field 'tasks_labels'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task and the likelihood of correctness.
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers' in 'fit'.
        """
        self._answers_base_checks(answers)
        self._predict_impl(answers)
        return self.probas

    def fit_predict(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Performes 'fit' and 'predict' in one call.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Predicted label for each task.
                - task - unique values from input dataset
                - label - most likely label

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        self._predict_impl(answers)
        return self.tasks_labels

    def fit_predict_proba(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Performes 'fit' and 'predict_proba' in one call.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task and the likelihood of correctness.
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
        """
        self._answers_base_checks(answers)
        self._fit_impl(answers)
        self._predict_impl(answers)
        return self.probas

    def _predict_impl(self, answers: pd.DataFrame) -> pd.DataFrame:
        self._init_by_input(answers)

        weighted_mv = MajorityVote()
        labels = weighted_mv.fit_predict(answers, self.performers_skills.rename(columns={'skill': 'weight'}))
        self.tasks_labels = labels
        self.probas = weighted_mv.probas
        return self.tasks_labels

    def _init_by_input(self, answers: pd.DataFrame) -> None:
        if not self.labels_set:
            self.labels_set = set(answers['label'])
            self.num_labels = len(self.labels_set)

        if self.performers_skills is None:
            self.performers_skills = pd.DataFrame(
                {
                    'performer': answers.performer.unique(),
                    'skill': 1 / self.num_labels + self.eps,
                }
            )
        else:
            new_performers_index = pd.Index(answers.performer, name='performer').difference(self.performers_skills.performer)
            new_performers_skills = pd.DataFrame({'skill': 1 / self.num_labels + self.eps}, index=new_performers_index)
            self.performers_skills = pd.concat([self.performers_skills.set_index('performer'), new_performers_skills], copy=False).reset_index()

        self.tasks_labels = pd.DataFrame({
            'task': answers.task.unique(),
            'label': np.NaN,
        })

    def _fit_impl(self, answers: pd.DataFrame) -> None:
        self._init_by_input(answers)

        self.lr_scheduler._lr = self.lr
        no_change = 0
        for _ in range(self.n_iter):
            labels_changed = self._train_iter(answers)
            self.lr = self.lr_scheduler.step()
            if not labels_changed:
                no_change += 1
            else:
                no_change = 0
            if no_change == self.early_stopping:
                break
        self._calc_performers_skills(answers, self.tasks_labels)
        self.lr_scheduler.reset()

    def _train_iter(self, answers: pd.DataFrame) -> bool:
        prev_labels = self.tasks_labels.copy()
        weighted_mv = MajorityVote()
        labels = weighted_mv.fit_predict(answers, self.performers_skills.rename(columns={'skill': 'weight'}))
        self.tasks_labels = labels
        self.probas = weighted_mv.probas

        labels_changed = not prev_labels.set_index('task')['label'].equals(labels.set_index('task')['label'])

        prev_skills = self.performers_skills.copy().set_index('performer')
        self._calc_performers_skills(answers, labels)
        self.performers_skills = self.performers_skills.set_index('performer')

        self.performers_skills = prev_skills + self.lr * (self.performers_skills - prev_skills)
        self.performers_skills = self.performers_skills.reset_index()

        return labels_changed
