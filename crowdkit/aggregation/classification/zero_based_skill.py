__all__ = ['ZeroBasedSkill']

from typing import Optional

import attr
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from .majority_vote import MajorityVote
from ..base import BaseClassificationAggregator
from ..utils import get_accuracy, named_series_attrib


@attr.attrs(auto_attribs=True)
class ZeroBasedSkill(BaseClassificationAggregator):
    """The Zero-Based Skill aggregation model aka ZBS.

    Performs weighted majority voting on tasks. After processing a pool of tasks,
    re-estimates workers' skills through a gradient descend step of optimization
    of the mean squared error of current skills and the fraction of responses that
    are equal to the aggregated labels.

    Repeats this process until labels do not change or the number of iterations exceeds.

    It's necessary that all workers in a dataset that send to 'predict' existed in answers
    the dataset that was sent to 'fit'.

    Args:
        n_iter: A number of iterations to perform.
        lr_init: A starting learning rate.
        lr_steps_to_reduce: A number of steps necessary to decrease the learning rate.
        lr_reduce_factor: A factor that the learning rate will be multiplied by every `lr_steps_to_reduce` steps.
        eps: A convergence threshold.

    Examples:
        >>> from crowdkit.aggregation import ZeroBasedSkill
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = ZeroBasedSkill().fit_predict(df)
    """

    n_iter: int = 100
    lr_init: float = 1.0
    lr_steps_to_reduce: int = 20
    lr_reduce_factor: float = 0.5
    eps: float = 1e-5

    # Available after fit
    skills_: Optional[pd.Series] = named_series_attrib(name='skill')

    # Available after predict or predict_proba
    # labels_
    probas_: Optional[pd.DataFrame] = attr.ib(init=False)

    def _init_skills(self, data: pd.DataFrame) -> pd.Series:
        skill_value = 1 / data.label.unique().size + self.eps
        skill_index = pd.Index(data.worker.unique(), name='worker')
        return pd.Series(skill_value, index=skill_index)

    def _apply(self, data: pd.DataFrame) -> 'ZeroBasedSkill':
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, self.skills_)
        self.labels_ = mv.labels_
        self.probas_ = mv.probas_
        return self

    def fit(self, data: pd.DataFrame) -> 'ZeroBasedSkill':
        """Fit the model.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.

        Returns:
            ZeroBasedSkill: self.
        """

        # Initialization
        data = data[['task', 'worker', 'label']]
        skills = self._init_skills(data)
        mv = MajorityVote()

        # Updating skills and re-fitting majority vote n_iter times
        learning_rate = self.lr_init
        for iteration in range(1, self.n_iter + 1):
            if iteration % self.lr_steps_to_reduce == 0:
                learning_rate *= self.lr_reduce_factor
            mv.fit(data, skills=skills)
            skills = skills + learning_rate * (get_accuracy(data, mv.labels_, by='worker') - skills)

        # Saving results
        self.skills_ = skills

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
