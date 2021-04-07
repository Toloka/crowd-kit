__all__ = ['GoldMajorityVote']

import attr
from sklearn.utils.validation import check_is_fitted

from . import annotations
from .annotations import manage_docstring, Annotation
from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote
from .utils import get_accuracy


@attr.s
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
    """

    # Available after fit
    skills_: annotations.OPTIONAL_SKILLS = attr.ib(init=False)

    # Available after predict or predict_proba
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    labels_: annotations.OPTIONAL_LABELS = attr.ib(init=False)

    @manage_docstring
    def _apply(self, data: annotations.LABELED_DATA) -> Annotation('GoldMajorityVote', 'self'):
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, self.skills_)
        self.labels_ = mv.labels_
        self.probas_ = mv.probas_
        return self

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> Annotation('GoldMajorityVote', 'self'):
        data = data[['task', 'performer', 'label']]
        self.skills_ = get_accuracy(data, true_labels=true_labels, by='performer')
        return self

    @manage_docstring
    def predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self._apply(data).labels_

    @manage_docstring
    def predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self._apply(data).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> annotations.TASKS_LABELS:
        return self.fit(data, true_labels).predict(data)

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA, true_labels: annotations.TASKS_TRUE_LABELS) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data, true_labels).predict_proba(data)
