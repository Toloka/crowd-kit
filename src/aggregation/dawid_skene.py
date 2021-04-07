__all__ = ['DawidSkene']

import attr
import numpy as np

from . import annotations
from .annotations import manage_docstring, Annotation
from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote
from .utils import get_most_probable_labels

_EPS = np.float_power(10, -10)


@attr.s
@manage_docstring
class DawidSkene(BaseAggregator):
    """
    Dawid-Skene aggregation model
    A. Philip Dawid and Allan M. Skene. 1979.
    Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, 1 (1979), 20–28.

    https://doi.org/10.2307/2346806
    """

    n_iter: int = attr.ib()

    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    priors_: annotations.OPTIONAL_PRIORS = attr.ib(init=False)
    labels_: annotations.OPTIONAL_LABELS = attr.ib(init=False)
    errors_: annotations.OPTIONAL_ERRORS = attr.ib(init=False)

    @staticmethod
    @manage_docstring
    def _m_step(data: annotations.LABELED_DATA, probas: annotations.TASKS_LABEL_PROBAS) -> annotations.ERRORS:
        """Perform M-step of Dawid-Skene algorithm.

        Given performers' answers and tasks' true labels probabilities estimates
        performer's errors probabilities matrix.
        """
        joined = data.join(probas, on='task')
        errors = joined.groupby(['performer', 'label'], sort=False).sum()
        errors.clip(lower=_EPS, inplace=True)
        errors /= errors.groupby('performer', sort=False).sum()
        return errors

    @staticmethod
    @manage_docstring
    def _e_step(data: annotations.LABELED_DATA, priors: annotations.LABEL_PRIORS, errors: annotations.ERRORS) -> annotations.TASKS_LABEL_PROBAS:
        """
        Perform E-step of Dawid-Skene algorithm.

        Given performer's answers, labels' prior probabilities and performer's performer's
        errors probabilities matrix estimates tasks' true labels probabilities.
        """
        joined = data.join(errors, on=['performer', 'label'])
        joined.drop(columns=['performer', 'label'], inplace=True)
        probas = priors * joined.groupby('task', sort=False).prod()
        return probas.div(probas.sum(axis=1), axis=0)

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> Annotation(type='DawidSkene', title='self'):

        # Initialization
        data = data[['task', 'performer', 'label']]
        probas = MajorityVote().fit_predict_proba(data)
        priors = probas.mean()
        errors = self._m_step(data, probas)

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            probas = self._e_step(data, priors, errors)
            priors = probas.mean()
            errors = self._m_step(data, probas)

        # Saving results
        self.probas_ = probas
        self.priors_ = priors
        self.errors_ = errors
        self.labels_ = get_most_probable_labels(probas)

        return self

    @manage_docstring
    def fit_predict_proba(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data).probas_

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA) -> annotations.TASKS_LABELS:
        return self.fit(data).labels_
