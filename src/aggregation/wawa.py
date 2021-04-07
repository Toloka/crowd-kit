__all__ = ['Wawa']

import attr

from sklearn.utils.validation import check_is_fitted
from . import annotations
from .annotations import Annotation, manage_docstring
from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote
from .utils import get_accuracy


@attr.attrs(auto_attribs=True)
@manage_docstring
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

    skills_: annotations.OPTIONAL_SKILLS = attr.ib(init=False)
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    labels_: annotations.OPTIONAL_LABELS = attr.ib(init=False)

    @manage_docstring
    def _apply(self, data: annotations.LABELED_DATA) -> Annotation('Wawa', 'self'):
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, skills=self.skills_)
        self.probas_ = mv.probas_
        self.labels_ = mv.labels_
        return self

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> Annotation('Wawa', 'self'):
        # TODO: support weights?
        data = data[['task', 'performer', 'label']]
        mv = MajorityVote().fit(data)
        self.skills_ = get_accuracy(data, true_labels=mv.labels_, by='performer')
        return self

    @manage_docstring
    def predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self._apply(data).labels_

    @manage_docstring
    def predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self._apply(data).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self.fit(data).predict(data)

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data).predict_proba(data)
