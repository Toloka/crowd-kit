__all__ = ['GLAD']

import attr
import numpy as np
import pandas as pd
import scipy
import scipy.stats
from scipy.optimize import minimize
from tqdm.auto import tqdm
from typing import Optional, Tuple

# logsumexp was moved to scipy.special in 0.19.0rc1 version of scipy
try:
    from scipy.special import logsumexp
except ImportError:
    from scipy.misc.common import logsumexp

from ..annotations import (
    manage_docstring,
    Annotation,
    OPTIONAL_PROBAS,
    LABELED_DATA,
    TASKS_LABEL_PROBAS,
    TASKS_LABELS,
    GLAD_ALPHAS,
    GLAD_BETAS,
)
from ..base import BaseClassificationAggregator
from ..utils import named_series_attrib


@attr.s
@manage_docstring
class GLAD(BaseClassificationAggregator):
    """
    Generative model of Labels, Abilities, and Difficulties
    J. Whitehill, P. Ruvolo, T. Wu, J. Bergsma, and J. Movellan
    Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise.
    Proceedings of the 22nd International Conference on Neural Information Processing Systems, 2009

    https://proceedings.neurips.cc/paper/2009/file/f899139df5e1059396431415e770c6dd-Paper.pdf
    """

    max_iter: int = attr.ib(default=100)
    eps: float = attr.ib(default=1e-5)
    silent: bool = attr.ib(default=True)
    labels_priors: Optional[pd.Series] = attr.ib(default=None)
    alphas_priors_mean: Optional[pd.Series] = attr.ib(default=None)
    betas_priors_mean: Optional[pd.Series] = attr.ib(default=None)
    m_step_max_iter: int = attr.ib(default=25)
    m_step_tol: float = attr.ib(default=1e-2)

    # Available after fit
    # labels_
    probas_: OPTIONAL_PROBAS = attr.ib(init=False)
    alphas_: GLAD_ALPHAS = named_series_attrib(name='alpha')
    betas_: GLAD_BETAS = named_series_attrib(name='beta')

    @manage_docstring
    def _join_all(
        self,
        data: LABELED_DATA,
        alphas: GLAD_ALPHAS,
        betas: GLAD_BETAS,
        priors: pd.Series
    ) -> pd.DataFrame:
        """Make a data frame with format `(task, performer, label, variable) -> (alpha, beta, posterior, delta)`
        """
        labels = list(priors.index)
        data.set_index('task', inplace=True)
        data[labels] = 0
        data.reset_index(inplace=True)
        data = data.melt(id_vars=['task', 'performer', 'label'], value_vars=labels, value_name='posterior')
        data = data.set_index('variable')
        data.reset_index(inplace=True)
        data.set_index('task', inplace=True)
        data['beta'] = betas
        data = data.reset_index().set_index('performer')
        data['alpha'] = alphas
        data.reset_index(inplace=True)
        data['delta'] = data['label'] == data['variable']
        return data

    @manage_docstring
    def _e_step(self, data: LABELED_DATA) -> TASKS_LABEL_PROBAS:
        """
        Perform E-step of GLAD algorithm.

        Given performer's alphas, labels' prior probabilities and task's beta parameters.
        """
        alpha_beta = data['alpha'] * np.exp(data['beta'])
        log_sigma = -self._softplus(-alpha_beta)
        log_one_minus_sigma = -self._softplus(alpha_beta)
        data['posterior'] = data['delta'] * log_sigma + \
                            (1 - data['delta']) * (log_one_minus_sigma - np.log(len(self.prior_labels_) - 1))
        # sum up by workers
        probas = data.groupby(['task', 'variable']).sum()['posterior']
        # add priors to every label
        probas = probas.add(np.log(self.priors_), level=1)
        # exponentiate and normalize
        probas = probas.groupby(['task']).transform(self._softmax)
        # put posterior in data['posterior']
        probas.name = 'posterior'
        data = pd.merge(data.drop('posterior', axis=1), probas, on=['task', 'variable'], copy=False)

        self.probas_ = probas.unstack()
        return data

    @manage_docstring
    def _gradient_Q(self, data: LABELED_DATA):
        """Compute gradient of loss function
        """

        sigma = scipy.special.expit(data['alpha'] * np.exp(data['beta']))
        # multiply by exponent of beta because of beta -> exp(beta) reparameterization
        data['dQb'] = data['posterior'] * (data['delta'] - sigma) * data['alpha'] * np.exp(data['beta'])
        dQbeta = data.groupby('task').sum()['dQb']
        # gradient of priors on betas
        dQbeta -= (self.betas_ - self.betas_priors_mean_)

        data['dQa'] = data['posterior'] * (data['delta'] - sigma) * np.exp(data['beta'])
        dQalpha = data.groupby('performer').sum()['dQa']
        # gradient of priors on alphas
        dQalpha -= (self.alphas_ - self.alphas_priors_mean_)
        return dQalpha, dQbeta

    @manage_docstring
    def _compute_Q(self, data: LABELED_DATA):
        """Compute loss function
        """

        alpha_beta = data['alpha'] * np.exp(data['beta'])
        log_sigma = -self._softplus(-alpha_beta)
        log_one_minus_sigma = -self._softplus(alpha_beta)
        data['task_expectation'] = data['posterior'] * \
                                   (data['delta'] * log_sigma +
                                    (1 - data['delta']) * (log_one_minus_sigma - np.log(len(self.prior_labels_) - 1)))
        Q = data['task_expectation'].sum()

        # priors on alphas and betas
        Q += np.log(scipy.stats.norm.pdf(self.alphas_ - self.alphas_priors_mean_)).sum()
        Q += np.log(scipy.stats.norm.pdf(self.betas_ - self.betas_priors_mean_)).sum()
        if np.isnan(Q):
            return -np.inf
        return Q

    @manage_docstring
    def _optimize_f(self, x: np.ndarray) -> float:
        """Compute loss by parameters represented by numpy array
        """
        alpha, beta = self._get_alphas_betas_by_point(x)
        self._update_alphas_betas(alpha, beta)
        return -self._compute_Q(self._current_data)

    @manage_docstring
    def _optimize_df(self, x: np.ndarray) -> np.ndarray:
        """Compute loss gradient by parameters represented by numpy array
        """
        alpha, beta = self._get_alphas_betas_by_point(x)
        self._update_alphas_betas(alpha, beta)
        dQalpha, dQbeta = self._gradient_Q(self._current_data)

        minus_grad = np.zeros_like(x)
        minus_grad[:len(self.performers_)] = -dQalpha[self.performers_].values
        minus_grad[len(self.performers_):] = -dQbeta[self.tasks_].values
        return minus_grad

    @manage_docstring
    def _update_alphas_betas(self, alphas: GLAD_ALPHAS, betas: GLAD_BETAS):
        self.alphas_ = alphas
        self.betas_ = betas
        self._current_data.set_index('performer', inplace=True)
        self._current_data['alpha'] = alphas
        self._current_data.reset_index(inplace=True)
        self._current_data.set_index('task', inplace=True)
        self._current_data['beta'] = betas
        self._current_data.reset_index(inplace=True)

    @manage_docstring
    def _get_alphas_betas_by_point(self, x: np.ndarray) -> Tuple[pd.Series, pd.Series]:
        alphas = pd.Series(x[:len(self.performers_)], index=self.performers_, name='alpha')
        alphas.index.name = 'performer'
        betas = pd.Series(x[len(self.performers_):], index=self.tasks_, name='beta')
        betas.index.name = 'task'
        return alphas, betas

    @manage_docstring
    def _m_step(self, data: LABELED_DATA) -> LABELED_DATA:
        """Optimize alpha and beta using conjugate gradient method
        """
        x_0 = np.concatenate([self.alphas_.values, self.betas_.values])
        self._current_data = data
        res = minimize(self._optimize_f, x_0, method='CG', jac=self._optimize_df, tol=self.m_step_tol,
                       options={'disp': False, 'maxiter': self.m_step_max_iter})
        self.alphas_, self.betas_ = self._get_alphas_betas_by_point(res.x)
        self._update_alphas_betas(self.alphas_, self.betas_)
        return self._current_data

    @manage_docstring
    def _init(self, data: LABELED_DATA) -> None:
        self.alphas_ = pd.Series(1.0, index=pd.unique(data.performer))
        self.betas_ = pd.Series(1.0, index=pd.unique(data.task))
        self.tasks_ = pd.unique(data['task'])
        self.performers_ = pd.unique(data['performer'])
        self.priors_ = self.labels_priors
        if self.priors_ is None:
            self.prior_labels_ = pd.unique(data['label'])
            self.priors_ = pd.Series(1. / len(self.prior_labels_), index=self.prior_labels_)
        self.alphas_priors_mean_ = self.alphas_priors_mean
        if self.alphas_priors_mean_ is None:
            self.alphas_priors_mean_ = pd.Series(1., index=self.alphas_.index)
        self.betas_priors_mean_ = self.betas_priors_mean
        if self.betas_priors_mean_ is None:
            self.betas_priors_mean_ = pd.Series(1., index=self.betas_.index)

    @staticmethod
    def _softplus(x: pd.Series, limit=30) -> np.ndarray:
        """log(1 + exp(x)) stable version

        For x > 30 or x < -30 error is less than 1e-13
        """
        positive_mask = x > limit
        negative_mask = x < -limit
        mask = positive_mask | negative_mask
        return np.log1p(np.exp(x * (1 - mask))) * (1 - mask) + x * positive_mask

    # backport for scipy < 1.12.0
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        return np.exp(x - logsumexp(x, keepdims=True))

    @manage_docstring
    def fit(self, data: LABELED_DATA) -> Annotation(type='GLAD', title='self'):
        # Initialization
        data = data.filter(['task', 'performer', 'label'])
        self._init(data)
        data = self._join_all(data, self.alphas_, self.betas_, self.priors_)
        data = self._e_step(data)
        Q = self._compute_Q(data)

        iterations_range = tqdm(range(self.max_iter)) if not self.silent else range(self.max_iter)
        for _ in iterations_range:
            last_Q = Q
            if not self.silent:
                iterations_range.set_description(f'Q = {round(Q, 4)}')

            # E-step
            data = self._e_step(data)

            # M-step
            data = self._m_step(data)

            Q = self._compute_Q(data)
            if np.abs((Q - last_Q) / last_Q) < self.eps:
                break

        self.labels_ = self.probas_.idxmax(axis=1)
        return self

    @manage_docstring
    def fit_predict_proba(self, data: LABELED_DATA) -> TASKS_LABEL_PROBAS:
        return self.fit(data).probas_

    @manage_docstring
    def fit_predict(self, data: LABELED_DATA) -> TASKS_LABELS:
        return self.fit(data).labels_
