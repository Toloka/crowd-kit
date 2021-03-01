import attr
import pandas as pd

from .majority_vote import MajorityVote
from.base_aggregator import BaseAggregator


@attr.attrs(auto_attribs=True)
class Wawa(BaseAggregator):
    """
    Worker Agreement with Aggregate

    Calculates the considers the likelihood of coincidence of the performers opinion with the majority
    Based on this, for each task, calculates the sum of the agreement of each label
    The correct label is the one where the sum of the agreements is greater

    After predicting stored different data frames (details in BaseAggregator):
        tasks_labels: Predicted labels for each task
        probas: Probabilities for each label for task
        performers_skills: Predicted labels for each task

    """

    def fit_predict(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Predict correct labels for tasks.
        After 'fit_predict' you can get probabilities for all labels from class field 'probas', and
        workers skills from 'workers_skills'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task.
                - task - unique values from input dataset
                - label - most likely label

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'
        """
        self._predict_impl(answers)
        return self.tasks_labels

    def fit_predict_proba(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Calculates probabilities for each label of task.
        If it was no such label for some task, this task doesn't has probs for this label.
        After 'fit_predict_proba' you can get predicted labels from class field 'tasks_labels', and
        workers skills from 'workers_skills'.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: probabilities for each label.
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'
        """
        self._predict_impl(answers)
        return self.probas

    def _predict_impl(self, answers: pd.DataFrame) -> pd.DataFrame:
        self._answers_base_checks(answers)

        # calculating performers skills
        mv_aggregation = MajorityVote()
        mv_aggregation.fit_predict(answers)
        self.performers_skills = mv_aggregation.performers_skills

        # join labels and skills
        labels_probas = answers.merge(self.performers_skills, on='performer')
        labels_probas = (
            labels_probas
            .groupby(['task', 'label'])
            .agg({'skill': sum})
            .reset_index()
            .rename(columns={'skill': 'score'})
        )

        labels_probas = self._calculate_probabilities(labels_probas)
        self._choose_labels(labels_probas)
