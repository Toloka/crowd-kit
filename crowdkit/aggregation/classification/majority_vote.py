__all__ = ['MajorityVote']

from typing import Optional

import attr
import pandas as pd

from ..base import BaseClassificationAggregator
from ..utils import normalize_rows, get_most_probable_labels, get_accuracy, add_skills_to_data, named_series_attrib


@attr.s
class MajorityVote(BaseClassificationAggregator):
    r"""The **Majority Vote** aggregation algorithm is a straightforward approach for categorical aggregation: for each task,
    it outputs a label with the largest number of responses. Additionaly, the Majority Vote
    can be used when different weights are assigned to workers' votes. In this case, the
    resulting label will have the largest sum of weights.


    {% note info %}

     If two or more labels have the largest number of votes, the resulting
     label will be the same for all tasks that have the same set of labels with the same number of votes.

     {% endnote %}

    Args:
        default_skill: Default worker weight value.

    Examples:
        Basic Majority Vote:
        >>> from crowdkit.aggregation import MajorityVote
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = MajorityVote().fit_predict(df)

        Weighted Majority Vote:
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
        labels_ (typing.Optional[pandas.core.series.Series]): The task labels. The `pandas.Series` data is indexed by `task`
            so that `labels.loc[task]` is the most likely true label of tasks.

        skills_ (typing.Optional[pandas.core.series.Series]): The workers' skills. The `pandas.Series` data is indexed by `worker`
            and has the corresponding worker skill.

        probas_ (typing.Optional[pandas.core.frame.DataFrame]): The probability distributions of task labels.
            The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that the `task` true label is equal to `label`.
            Each probability is in the range from 0 to 1, all task probabilities must sum up to 1.

        on_missing_skill (str): A value which specifies how to handle assignments performed by workers with an unknown skill.

            Possible values:
                    * "error" — raises an exception if there is at least one assignment performed by a worker with an unknown skill;
                    * "ignore" — drops assignments performed by workers with an unknown skill during prediction. Raises an exception if there are no
                    assignments with a known skill for any task;
                    * value — the default value will be used if a skill is missing.
    """

    # TODO: remove skills_
    skills_: Optional[pd.Series] = named_series_attrib(name='skill')
    probas_: Optional[pd.DataFrame] = attr.ib(init=False)
    # labels_
    on_missing_skill: str = attr.ib(default='error')
    default_skill: Optional[float] = attr.ib(default=None)

    def fit(self, data: pd.DataFrame, skills: pd.Series = None) -> 'MajorityVote':
        """Fits the model to the training data.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

            skills (Series): The workers' skills. The `pandas.Series` data is indexed by `worker
                and has the corresponding worker skill.

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
        """Fits the model to the training data and returns probability distributions of labels for each task.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

            skills (Series): The workers' skills. The `pandas.Series` data is indexed by `worker`
                and has the corresponding worker skill.

        Returns:
            DataFrame: The probability distributions of task labels.
                The `pandas.DataFrame` data is indexed by `task` so that `result.loc[task, label]` is the probability that the `task` true label is equal to `label`.
                Each probability is in the range from 0 to 1, all task probabilities must sum up to 1.
        """

        return self.fit(data, skills).probas_

    def fit_predict(self, data: pd.DataFrame, skills: pd.Series = None) -> pd.Series:
        """Fits the model to the training data and returns the aggregated results.

         Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

            skills (Series): The workers' skills. The `pandas.Series` data is indexed by `worker`
                and has the corresponding worker skill.

         Returns:
            Series: The task labels. The `pandas.Series` data is indexed by `task`
                so that `labels.loc[task]` is the most likely true label of tasks.
         """

        return self.fit(data, skills).labels_
