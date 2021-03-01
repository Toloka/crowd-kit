import attr
import pandas as pd
from typing import Optional

from .base_aggregator import BaseAggregator


@attr.attrs(auto_attribs=True)
class MajorityVote(BaseAggregator):
    """
    Majority Vote - chooses the correct label for which more performers voted

    After predicting stored different data frames (details in BaseAggregator):
        tasks_labels: Predicted labels for each task
        probas: Probabilities for each label for task
        performers_skills: Predicted labels for each task
    """

    def fit_predict(self, answers: pd.DataFrame, performers_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Chooses the correct label for which more performers voted

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'

        Returns:
            pandas.DataFrame: Scores for each task and the likelihood of correctness.
                - task - unique values from input dataset
                - label - most likely label
            performers_weights(pandas.DataFrame): Optional, frame with performers weights. Should contain columns
                'performer', 'weight'.

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'
        """
        self._predict_impl(answers, performers_weights)
        return self.tasks_labels

    def fit_predict_proba(self, answers: pd.DataFrame, performers_weights: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Calculates Probabilities for each label of task.
        If it was no such label for some task, this task doesn't has probs for this label.

        Args:
            answers(pandas.DataFrame): Frame with performers answers on task. One row per answer.
                Should contain columns 'performer', 'task', 'label'
            performers_weights(pandas.DataFrame): Optional, frame with performers weights. Should contain columns
                'performer', 'weight'.

        Returns:
            pandas.DataFrame: Probabilities for each label of task
                - task - as dataframe index
                - label - as dataframe columns
                - proba - dataframe values

        Raises:
            TypeError: If answers don't has pandas.DataFrame type
            AssertionError: If there is some collumn missing in 'answers'
        """
        self._predict_impl(answers, performers_weights)
        return self.probas

    def _predict_impl(self, answers: pd.DataFrame, performers_weights: Optional[pd.DataFrame] = None) -> None:
        self._answers_base_checks(answers)
        if performers_weights is not None:
            if not isinstance(performers_weights, pd.DataFrame):
                raise TypeError('"performers_weights" parameter must be of type pandas DataFrame')
            assert 'performer' in performers_weights, 'There is no "performer" column in "performers_weights"'
            assert 'weight' in performers_weights, 'There is no "weight" column in "performers_weights"'

        if performers_weights is None:
            answ_scores = answers.groupby(['task', 'label'], as_index=False)['performer'].count()
            answ_scores = answ_scores.rename(columns={'performer': 'score'})
        else:
            answ_scores = answers.join(performers_weights.set_index('performer'), on='performer', rsuffix='_r').groupby(['task', 'label'], as_index=False).sum('weight')
            answ_scores = answ_scores.rename(columns={'weight': 'score'})

        probas = self._calculate_probabilities(answ_scores)
        labels = self._choose_labels(probas)
        self._calc_performers_skills(answers, labels)
