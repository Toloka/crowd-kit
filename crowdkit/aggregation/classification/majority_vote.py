__all__ = ['MajorityVote']

from typing import Optional

import attr
import pandas as pd

from ..base import BaseClassificationAggregator
from ..utils import normalize_rows, get_most_probable_labels, get_accuracy, add_skills_to_data, named_series_attrib


@attr.s
class MajorityVote(BaseClassificationAggregator):
    """Majority Vote aggregation algorithm.

    Majority vote is a straightforward approach for categorical aggregation: for each task,
    it outputs a label which has the largest number of responses. Additionaly, the majority vote
    can be used when different weights assigned for workers' votes. In this case, the
    resulting label will be the one with the largest sum of weights.


    {% note info %}

     In case when two or more labels have the largest number of votes, the resulting
     label will be the same for all tasks which have the same set of labels with equal count of votes.

     {% endnote %}

    Args:
        default_skill: Defualt worker's weight value.

    Examples:
        Basic majority voting:
        >>> from crowdkit.aggregation import MajorityVote
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = MajorityVote().fit_predict(df)

        Weighted majority vote:
        >>> import pandas as pd
        >>> from crowdkit.aggregation import MajorityVote
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', 0],
        >>>         ['t1', 'p2', 0],
        >>>         ['t1', 'p3', 1],
        >>>         ['t2', 'p1', 1],
        >>>         ['t2', 'p2', 0],
        >>>         ['t2', 'p3', 1],
        >>>     ],
        >>>     columns=['task', 'worker', 'label']
        >>> )
        >>> skills = pd.Series({'p1': 0.5, 'p2': 0.7, 'p3': 0.4})
        >>> result = MajorityVote.fit_predict(df, skills)

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

        on_missing_skill (str): How to handle assignments done by workers with unknown skill.
            Possible values:
                    * "error" — raise an exception if there is at least one assignment done by user with unknown skill;
                    * "ignore" — drop assignments with unknown skill values during prediction. Raise an exception if there is no
                    assignments with known skill for any task;
                    * value — default value will be used if skill is missing.
    """

    # TODO: remove skills_
    skills_: Optional[pd.Series] = named_series_attrib(name='skill')
    probas_: Optional[pd.DataFrame] = attr.ib(init=False)
    # labels_
    on_missing_skill: str = attr.ib(default='error')
    default_skill: Optional[float] = attr.ib(default=None)

    def fit(self, data: pd.DataFrame, skills: pd.Series = None) -> 'MajorityVote':
        """Fit the model.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            skills (Series): workers' skills.
                A pandas.Series index by workers and holding corresponding worker's skill

        Returns:
            MajorityVote: self.
        """

        data = data[['task', 'worker', 'label']]

        if skills is None:
            scores = data[['task', 'label']].value_counts()
        else:
            data = add_skills_to_data(data, skills, self.on_missing_skill, self.default_skill)
            scores = data.groupby(['task', 'label'])['skill'].sum()

        self.probas_ = normalize_rows(scores.unstack('label', fill_value=0))
        self.labels_ = get_most_probable_labels(self.probas_)
        self.skills_ = get_accuracy(data, self.labels_, by='worker')

        return self

    def fit_predict_proba(self, data: pd.DataFrame, skills: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit the model and return probability distributions on labels for each task.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            skills (Series): workers' skills.
                A pandas.Series index by workers and holding corresponding worker's skill

        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

        return self.fit(data, skills).probas_

    def fit_predict(self, data: pd.DataFrame, skills: pd.Series = None) -> pd.Series:
        """Fit the model and return aggregated results.

         Args:
             data (DataFrame): Workers' labeling results.
                 A pandas.DataFrame containing `task`, `worker` and `label` columns.
             skills (Series): workers' skills.
                 A pandas.Series index by workers and holding corresponding worker's skill

         Returns:
             Series: Tasks' labels.
                 A pandas.Series indexed by `task` such that `labels.loc[task]`
                 is the tasks's most likely true label.
         """

        return self.fit(data, skills).labels_
