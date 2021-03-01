__all__ = ['GoldMajorityVote']

import attr
import pandas as pd

from . import annotations
from .annotations import manage_docstring
from .base_aggregator import BaseAggregator


@attr.attrs(auto_attribs=True)
@manage_docstring
class GoldMajorityVote(BaseAggregator):
    """Majority Vote when exist golden dataset (ground truth) for some tasks

    Calculates the probability of a correct label for each performer based on the golden set
    Based on this, for each task, calculates the sum of the probabilities of each label
    The correct label is the one where the sum of the probabilities is greater

    For Example: You have 10k tasks completed by 3k different performers. And you have 100 tasks where you already
    know ground truth labels. First you can call 'fit' to calc percents of correct labels for each performers.
    And then call 'predict' to calculate labels for you 10k tasks.

    It's necessary that:
    1. All performers must done at least one task from golden dataset.
    2. All performers in dataset that send to 'predict', existed in answers dataset that was sent to 'fit'

    After fit stored 'performers_skills' - Predicted labels for each task.

    After predicting stored different data frames (details in BaseAggregator):
        tasks_labels: Predicted labels for each task
        probas: Probabilities for each label for task
    """

    def fit(self, answers_on_gold: pd.DataFrame, gold_df: pd.DataFrame) -> 'GoldMajorityVote':
        """Calculates the skill for each performers, based on answers on golden dataset
        The calculated skills are stored in an instance of the class and can be obtained by the field 'performers_skills'
        After 'fit' you can get 'performer_skills' from class field.

        Args:
            answers_on_gold(pandas.DataFrame): Frame contains performers answers on golden tasks. One row per answer.
                Should contain columns 'performer', 'task', 'label'. Dataframe could contains answers not only for golden
                tasks. This answers will be ignored.
            gold_df(pandas.DataFrame): Frame with ground truth labels for tasks.
                Should contain columns 'performer', 'task'. And may contain column 'weight', if you have different scores
                for different tasks.
        Returns:
            GoldMajorityVote: self for call next methods

        Raises:
            TypeError: If the input datasets are not of type pandas.DataFrame.
            AssertionError: If there is some collumn missing in 'dataframes'. Or if it's impossible to calculate the
                skill for any performer. For example, some performers do not have answers to tasks from the golden dataset.
        """
        self._answers_base_checks(answers_on_gold)

        if not isinstance(gold_df, pd.DataFrame):
            raise TypeError('"gold_df" parameter must be of type pandas DataFrame')
        assert 'task' in gold_df, 'There is no "task" column in "gold_df"'
        assert 'label' in gold_df, 'There is no "label" column in "gold_df"'

        # checking that we can compute skills for all performers
        answers_with_truth = answers_on_gold.merge(gold_df, on='task', suffixes=('', '_truth'))
        performers_without_skill = set(answers_on_gold['performer'].unique()) - set(answers_with_truth['performer'].unique())
        assert not performers_without_skill, 'It is impossible to compute skills for some performers in "crowd_on_gold_df"'\
            ' because of that performers did not complete any golden task (no tasks for this performers in "gold_df"))'

        self._calc_performers_skills(answers_on_gold, gold_df)
        return self

    @manage_docstring
    def predict(self, data: annotations.DATA) -> annotations.TASKS_LABELS:
        """Predict correct labels for tasks. Using calculated performers skill, stored in self instance.
        After 'predict' you can get probabilities for all labels from class field 'probas'.

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers_on_gold' in 'fit'.
        """
        self._predict_impl(data)
        return self.tasks_labels

    @manage_docstring
    def predict_proba(self, data: annotations.DATA) -> annotations.PROBAS:
        """Calculates Probabilities for each label of task.
        If it was no such label for some task, this task doesn't has probs for this label.
        After 'predict_proba' you can get predicted labels from class field 'tasks_labels'.

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'.
                Or when 'predict' called without 'fit'.
                Or if there are new performers in 'answer' that were not in 'answers_on_gold' in 'fit'.
        """
        self._predict_impl(data)
        return self.probas

    @manage_docstring
    def _predict_impl(self, answers: annotations.DATA) -> None:
        self._answers_base_checks(answers)

        assert self.performers_skills is not None, '"Predict" called without "fit".'

        # checking that all performers in crowd_df has skills in "performers_skills"
        performers_without_skill_in_crowd = set(answers['performer'].unique()) - set(self.performers_skills['performer'].unique())
        assert not performers_without_skill_in_crowd, 'Unknown skill for some performers in "crowd_df"'\
            ' because of that performers have no tasks in "crowd_on_gold_df"'

        # join labels and skills
        labels_probas = answers.merge(self.performers_skills, on='performer')
        labels_probas = (
            labels_probas
            .groupby(['task', 'label'])
            .agg({'skill': sum})
            .reset_index()
            .rename(columns={'skill': 'score'}))

        labels_probas = self._calculate_probabilities(labels_probas)
        self._choose_labels(labels_probas)
