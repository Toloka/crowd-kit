__all__ = ['Wawa']

from typing import Optional

import attr
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .majority_vote import MajorityVote
from ..base import BaseClassificationAggregator
from ..utils import get_accuracy, named_series_attrib


@attr.s
class Wawa(BaseClassificationAggregator):
    """Worker Agreement with Aggregate.

        This algorithm does three steps:
        1. Calculate the majority vote label
        2. Estimate workers' skills as a fraction of responses that are equal to the majority vote
        3. Calculate the weigthed majority vote based on skills from the previous step

        Examples:
            >>> from crowdkit.aggregation import Wawa
            >>> from crowdkit.datasets import load_dataset
            >>> df, gt = load_dataset('relevance-2')
            >>> result = Wawa().fit_predict(df)

        Attributes:
            labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.

            skills_ (typing.Optional[pandas.core.series.Series]): workers' skills.
                A pandas.Series index by workers and holding corresponding worker's skill
            probas_ (typing.Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

    skills_: Optional[pd.Series] = named_series_attrib(name='skill')
    probas_: Optional[pd.DataFrame] = attr.ib(init=False)

    # labels_

    def _apply(self, data: pd.DataFrame) -> 'Wawa':
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, skills=self.skills_)
        self.probas_ = mv.probas_
        self.labels_ = mv.labels_
        return self

    def fit(self, data: pd.DataFrame) -> 'Wawa':
        """Fit the model.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            Wawa: self.
        """

        # TODO: support weights?
        data = data[['task', 'worker', 'label']]
        mv = MajorityVote().fit(data)
        self.skills_ = get_accuracy(data, true_labels=mv.labels_, by='worker')
        return self

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """Infer the true labels when the model is fitted.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """

        return self._apply(data).labels_

    def predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return probability distributions on labels for each task when the model is fitted.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

        return self._apply(data).probas_

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fit the model and return aggregated results.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """

        return self.fit(data).predict(data)

    def fit_predict_proba(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and return probability distributions on labels for each task.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

        return self.fit(data).predict_proba(data)
