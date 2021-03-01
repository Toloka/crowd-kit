__all__ = ['DawidSkene']

import numpy as np

from . import annotations
from .annotations import manage_docstring, Annotation
from .base_aggregator import BaseAggregator
from .majority_vote import MajorityVote

_EPS = np.float_power(10, -10)


@manage_docstring
class DawidSkene(BaseAggregator):
    """
    Dawid-Skene aggregation model
    A. Philip Dawid and Allan M. Skene. 1979.
    Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, 1 (1979), 20–28.

    https://doi.org/10.2307/2346806
    """

    probas: annotations.OPTIONAL_CLASSLEVEL_PROBAS
    priors: annotations.OPTIONAL_CLASSLEVEL_PRIORS
    task_labels: annotations.OPTIONAL_CLASSLEVEL_TASKS_LABELS
    errors: annotations.OPTIONAL_CLASSLEVEL_ERRORS

    def __init__(self, n_iter: int):
        """
        Args:
            n_iter: Number of iterations to perform
        """
        self.n_iter = n_iter
        self.proba = None
        self.priors = None
        self.tasks_labels = None
        self.errors = None

    @staticmethod
    @manage_docstring
    def _m_step(data: annotations.DATA, probas: annotations.PROBAS) -> annotations.ERRORS:
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
    def _e_step(data: annotations.DATA, priors: annotations.PROBAS, errors: annotations.ERRORS) -> annotations.PROBAS:
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
    def fit(self, data: annotations.DATA) -> Annotation(type='DawidSkene', title='self'):

        # Initialization
        data = data[['task', 'performer', 'label']]
        self.probas = MajorityVote().fit_predict_proba(data).fillna(0)
        self.priors = self.probas.mean()
        self.errors = self._m_step(data, self.probas)

        # Updating proba and errors n_iter times
        for _ in range(self.n_iter):
            self.probas = self._e_step(data, self.priors, self.errors)
            self.priors = self.probas.mean()
            self.errors = self._m_step(data, self.probas)

        # Saving results
        self.task_labels = self._choose_labels(self.probas)

        return self

    @manage_docstring
    def fit_predict_proba(self, data: annotations.DATA) -> annotations.PROBAS:
        return self.fit(data).probas

    @manage_docstring
    def fit_predict(self, data: annotations.DATA) -> annotations.TASKS_LABELS:
        return self.fit(data).tasks_labels
