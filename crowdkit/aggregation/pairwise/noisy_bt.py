__all__ = ['NoisyBradleyTerry']

import attr
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BasePairwiseAggregator
from ..utils import factorize, named_series_attrib


@attr.s
@manage_docstring
class NoisyBradleyTerry(BasePairwiseAggregator):
    """
    A modification of Bradley-Terry with parameters for workers' skills and
    their biases.
    """
    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-5)
    random_state: int = attr.ib(default=0)
    skills_: annotations.SKILLS = named_series_attrib(name='skill')
    biases_: annotations.BIASES = named_series_attrib(name='bias')
    # scores_

    @manage_docstring
    def fit(self, data: annotations.PAIRWISE_DATA) -> Annotation(type='NoisyBradleyTerry', title='self'):
        unique_labels, np_data = factorize(data[['left', 'right', 'label']].values)
        unique_workers, np_workers = factorize(data.worker.values)
        np.random.seed(self.random_state)
        x_0 = np.random.rand(unique_labels.size + 2 * unique_workers.size)

        x = minimize(self._compute_log_likelihood, x_0, jac=self._compute_gradient,
                     args=(np_data, np_workers, unique_labels.size, unique_workers.size),
                     method='L-BFGS-B', options={'maxiter': self.n_iter, 'ftol': self.tol})

        biases_begin = unique_labels.size
        workers_begin = biases_begin + unique_workers.size

        self.scores_ = pd.Series(expit(x.x[:biases_begin]), index=pd.Index(unique_labels, name='label'), name='score')
        self.biases_ = pd.Series(expit(x.x[biases_begin:workers_begin]), index=unique_workers)
        self.skills_ = pd.Series(expit(x.x[workers_begin:]), index=unique_workers)

        return self

    @manage_docstring
    def fit_predict(self, data: annotations.PAIRWISE_DATA) -> annotations.LABEL_SCORES:
        return self.fit(data).scores_

    @staticmethod
    def _compute_log_likelihood(x: np.ndarray, np_data: np.ndarray, np_workers: np.ndarray, labels: int, workers: int) -> float:
        s_i = x[np_data[:, 0]]
        s_j = x[np_data[:, 1]]
        y = np_data[:, 0] == np_data[:, 2]
        q = x[np_workers + labels]
        gamma = x[np_workers + labels + workers]

        total = np.sum(np.log(expit(gamma) * expit(y * (s_i - s_j)) + (1 - expit(gamma)) * expit(y * q)))

        return -total

    @staticmethod
    def _compute_gradient(x: np.ndarray, np_data: np.ndarray, np_workers: np.ndarray, labels: int, workers: int) -> np.ndarray:
        gradient = np.zeros_like(x)

        for worker_idx, (left_idx, right_idx, label) in zip(np_workers, np_data):
            s_i = x[left_idx]
            s_j = x[right_idx]
            y = label == left_idx
            q = x[labels + worker_idx]
            gamma = x[labels + workers + worker_idx]

            # We'll use autograd in the future
            gradient[left_idx] += (y * np.exp(y * (-(s_i - s_j)))) / ((np.exp(-gamma) + 1) * (np.exp(y * (-(s_i - s_j))) + 1) ** 2 * (1 / ((np.exp(-gamma) + 1) * (np.exp(y * (-(s_i - s_j))) + 1)) + (1 - 1 / (np.exp(-gamma) + 1)) / (np.exp(-q * y) + 1)))  # noqa
            gradient[right_idx] += -(y * (np.exp(q * y) + 1) * np.exp(y * (s_i - s_j) + gamma)) / ((np.exp(y * (s_i - s_j)) + 1) * (np.exp(y * (s_i - s_j) + gamma + q * y) + np.exp(y * (s_i - s_j) + gamma) + np.exp(y * (s_i - s_j) + q * y) + np.exp(q * y)))  # noqa
            gradient[labels + worker_idx] = (y * np.exp(q * y) * (np.exp(s_i * y) + np.exp(s_j * y))) / ((np.exp(q * y) + 1) * (np.exp(y * (s_i + q) + gamma) + np.exp(s_i * y + gamma) + np.exp(y * (s_i + q)) + np.exp(y * (s_j + q))))  # noqa
            gradient[labels + workers + worker_idx] = (np.exp(gamma) * (np.exp(s_i * y) - np.exp(y * (s_j + q)))) / ((np.exp(gamma) + 1) * (np.exp(y * (s_i + q) + gamma) + np.exp(s_i * y + gamma) + np.exp(y * (s_i + q)) + np.exp(y * (s_j + q))))  # noqa
        return -gradient
