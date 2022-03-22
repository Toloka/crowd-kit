__all__ = ['GoldMajorityVote']

import attr
from sklearn.utils.validation import check_is_fitted

from .. import annotations
from ..annotations import manage_docstring, Annotation
from ..base import BaseClassificationAggregator
from .majority_vote import MajorityVote
from ..utils import get_accuracy, named_series_attrib


@attr.s
@manage_docstring
class GoldMajorityVote(BaseClassificationAggregator):
    """Majority Vote when exist golden dataset (ground truth) for some tasks.

    Calculates the probability of a correct label for each worker based on the golden set.
    Based on this, for each task, calculates the sum of the probabilities of each label.
    The correct label is the one where the sum of the probabilities is greater.

    For Example: You have 10k tasks completed by 3k different workers. And you have 100 tasks where you already
    know ground truth labels. First you can call `fit` to calc percents of correct labels for each workers.
    And then call `predict` to calculate labels for you 10k tasks.

    It's necessary that:
    1. All workers must done at least one task from golden dataset.
    2. All workers in dataset that send to `predict`, existed in answers dataset that was sent to `fit`.

    Examples:
        >>> import pandas as pd
        >>> from crowdkit.aggregation import GoldMajorityVote
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
        >>> true_labels = pd.Series({'t1': 0})
        >>> gold_mv = GoldMajorityVote()
        >>> result = gold_mv.fit_predict(df, true_labels)
    """

    # Available after fit
    skills_: annotations.OPTIONAL_SKILLS = named_series_attrib(name='skill')

    # Available after predict or predict_proba
    # labels_
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)

    @manage_docstring
    def _apply(self, data: annotations.LABELED_DATA) -> Annotation('GoldMajorityVote', 'self'):
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, self.skills_)
        self.labels_ = mv.labels_
        self.probas_ = mv.probas_
        return self

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> Annotation('GoldMajorityVote', 'self'):
        """
        Estimate the workers' skills.
        """

        data = data[['task', 'worker', 'label']]
        self.skills_ = get_accuracy(data, true_labels=true_labels, by='worker')
        return self

    @manage_docstring
    def predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        """
        Infer the true labels when the model is fitted.
        """

        return self._apply(data).labels_

    @manage_docstring
    def predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        """
        Return probability distributions on labels for each task when the model is fitted.
        """

        return self._apply(data).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> annotations.TASKS_LABELS:
        """
        Fit the model and return aggregated results.
        """

        return self.fit(data, true_labels).predict(data)

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> annotations.TASKS_LABEL_PROBAS:
        """
        Fit the model and return probability distributions on labels for each task.
        """

        return self.fit(data, true_labels).predict_proba(data)
