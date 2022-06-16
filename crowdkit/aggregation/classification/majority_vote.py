__all__ = ['MajorityVote']

from typing import Optional

import attr

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseClassificationAggregator
from ..utils import normalize_rows, get_most_probable_labels, get_accuracy, add_skills_to_data, named_series_attrib


@attr.s
@manage_docstring
class MajorityVote(BaseClassificationAggregator):
    """
    Majority Vote aggregation algorithm.

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
    """

    # TODO: remove skills_
    skills_: annotations.OPTIONAL_SKILLS = named_series_attrib(name='skill')
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    # labels_
    on_missing_skill: annotations.ON_MISSING_SKILL = attr.ib(default='error')
    default_skill: Optional[float] = attr.ib(default=None)

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA, skills: annotations.SKILLS = None) -> Annotation(type='MajorityVote', title='self'):
        """
        Fit the model.
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

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABEL_PROBAS:
        """
        Fit the model and return probability distributions on labels for each task.
        """

        return self.fit(data, skills).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABELS:
        """
        Fit the model and return aggregated results.
        """

        return self.fit(data, skills).labels_
