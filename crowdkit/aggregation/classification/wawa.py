__all__ = ['Wawa']

import attr

from sklearn.utils.validation import check_is_fitted
from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseClassificationAggregator
from .majority_vote import MajorityVote
from ..utils import get_accuracy, named_series_attrib


@attr.s
@manage_docstring
class Wawa(BaseClassificationAggregator):
    """
    Worker Agreement with Aggregate.

    This algorithm does three steps:
    1. Calculate the majority vote label
    2. Estimate workers' skills as a fraction of responses that are equal to the majority vote
    3. Calculate the weigthed majority vote based on skills from the previous step

    Examples:
        >>> from crowdkit.aggregation import Wawa
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = Wawa().fit_predict(df)
    """

    skills_: annotations.OPTIONAL_SKILLS = named_series_attrib(name='skill')
    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    # labels_

    @manage_docstring
    def _apply(self, data: annotations.LABELED_DATA) -> Annotation('Wawa', 'self'):
        check_is_fitted(self, attributes='skills_')
        mv = MajorityVote().fit(data, skills=self.skills_)
        self.probas_ = mv.probas_
        self.labels_ = mv.labels_
        return self

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> Annotation('Wawa', 'self'):
        """
        Fit the model.
        """

        # TODO: support weights?
        data = data[['task', 'worker', 'label']]
        mv = MajorityVote().fit(data)
        self.skills_ = get_accuracy(data, true_labels=mv.labels_, by='worker')
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
    def fit_predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        """
        Fit the model and return aggregated results.
        """

        return self.fit(data).predict(data)

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        """
        Fit the model and return probability distributions on labels for each task.
        """

        return self.fit(data).predict_proba(data)
