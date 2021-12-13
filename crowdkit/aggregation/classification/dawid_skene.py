__all__ = ['DawidSkene']

import attr
import numpy as np
import pandas as pd

from .. import annotations
from ..annotations import manage_docstring, Annotation
from ..base import BaseClassificationAggregator
from .majority_vote import MajorityVote
from ..utils import get_most_probable_labels, named_series_attrib

_EPS = np.float_power(10, -10)


@attr.s
@manage_docstring
class DawidSkene(BaseClassificationAggregator):
    """
    Dawid-Skene aggregation model
    A. Philip Dawid and Allan M. Skene. 1979.
    Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, 1 (1979), 20–28.

    https://doi.org/10.2307/2346806
    """

    n_iter: int = attr.ib()

    probas_: annotations.OPTIONAL_PROBAS = attr.ib(init=False)
    priors_: annotations.OPTIONAL_PRIORS = named_series_attrib(name='prior')
    # labels_
    errors_: annotations.OPTIONAL_ERRORS = attr.ib(init=False)

    @staticmethod
    @manage_docstring
    def _m_step(data: annotations.LABELED_DATA, probas: annotations.TASKS_LABEL_PROBAS) -> annotations.ERRORS:
        """Perform M-step of Dawid-Skene algorithm.

        Given performers' answers and tasks' true labels probabilities estimates
        performer's errors probabilities matrix.
        """
        joined = data.join(probas, on='task')
        joined.drop(columns=['task'], inplace=True)

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

        # We have to multiply lots of probabilities and such products are known to converge
        # to zero exponentialy fast. To avoid floating-point precision problems we work with
        # logs of original values
        joined = data.join(np.log2(errors), on=['performer', 'label'])
        joined.drop(columns=['performer', 'label'], inplace=True)
        log_likelihoods = np.log2(priors) + joined.groupby('task', sort=False).sum()

        # Exponentiating log_likelihoods 'as is' may still get us beyond our precision.
        # So we shift every row of log_likelihoods by a constant (which is equivalent to
        # multiplying likelihoods rows by a constant) so that max log_likelihood in each
        # row is equal to 0. This trick ensures proper scaling after exponentiating and
        # does not affect the result of E-step
        scaled_likelihoods = np.exp2(log_likelihoods.sub(log_likelihoods.max(axis=1), axis=0))
        return scaled_likelihoods.div(scaled_likelihoods.sum(axis=1), axis=0)

    @manage_docstring
    def fit(self, data: annotations.LABELED_DATA) -> Annotation(type='DawidSkene', title='self'):

        data = data[['task', 'performer', 'label']]

        # Early exit
        if not data.size:
            self.probas_ = pd.DataFrame()
            self.priors_ = pd.Series(dtype=float)
            self.errors_ = pd.DataFrame()
            self.labels_ = pd.Series(dtype=float)
            return self

        # Initialization
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
