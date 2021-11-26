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
    A modification of Bradley-Terry with parameters for performers' skills and
    their biases.
    """
    n_iter: int = attr.ib(default=100)
    random_state: int = attr.ib(default=0)
    skills_: annotations.SKILLS = named_series_attrib(name='skill')
    biases_: annotations.BIASES = named_series_attrib(name='bias')
    # scores_

    @manage_docstring
    def fit(self, data: annotations.PAIRWISE_DATA) -> Annotation(type='NoisyBradleyTerry', title='self'):
        unique_labels, np_data = factorize(data[['left', 'right', 'label']].values)
        unique_performers, np_performers = factorize(data.performer.values)
        np.random.seed(self.random_state)
        x_0 = np.random.rand(unique_labels.size + 2 * unique_performers.size)

        x = minimize(self._compute_log_likelihood, x_0, jac=self._compute_gradient,
                     args=(np_data, np_performers, unique_labels.size, unique_performers.size),
                     method='L-BFGS-B', options={'maxiter': self.n_iter})

        biases_begin = unique_labels.size
        performers_begin = biases_begin + unique_performers.size

        self.scores_ = pd.Series(expit(x.x[:biases_begin]), index=pd.Index(unique_labels, name='label'), name='score')
        self.biases_ = pd.Series(expit(x.x[biases_begin:performers_begin]), index=unique_performers)
        self.skills_ = pd.Series(expit(x.x[performers_begin:]), index=unique_performers)

        return self

    @manage_docstring
    def fit_predict(self, data: annotations.PAIRWISE_DATA) -> annotations.LABEL_SCORES:
        return self.fit(data).scores_

    @staticmethod
    def _compute_log_likelihood(x: np.ndarray, np_data: np.ndarray, np_performers: np.ndarray, labels: int, performers: int) -> float:
        s_i = x[np_data[:, 0]]
        s_j = x[np_data[:, 1]]
        y = np_data[:, 0] == np_data[:, 2]
        q = x[np_performers + labels]
        gamma = x[np_performers + labels + performers]

        total = np.sum(np.log(expit(gamma) * expit(y * (s_i - s_j)) + (1 - expit(gamma)) * expit(y * q)))

        return -total

    @staticmethod
    def _compute_gradient(x: np.ndarray, np_data: np.ndarray, np_performers: np.ndarray, labels: int, performers: int) -> np.ndarray:
        gradient = np.zeros_like(x)

        for performer_idx, (left_idx, right_idx, label) in zip(np_performers, np_data):
            s_i = x[left_idx]
            s_j = x[right_idx]
            y = label == left_idx
            q = x[labels + performer_idx]
            gamma = x[labels + performers + performer_idx]

            # We'll use autograd in the future
            gradient[left_idx] += (y * np.exp(y * (-(s_i - s_j)))) / ((np.exp(-gamma) + 1) * (np.exp(y * (-(s_i - s_j))) + 1) ** 2 * (1 / ((np.exp(-gamma) + 1) * (np.exp(y * (-(s_i - s_j))) + 1)) + (1 - 1 / (np.exp(-gamma) + 1)) / (np.exp(-q * y) + 1)))  # noqa
            gradient[right_idx] += -(y * (np.exp(q * y) + 1) * np.exp(y * (s_i - s_j) + gamma)) / ((np.exp(y * (s_i - s_j)) + 1) * (np.exp(y * (s_i - s_j) + gamma + q * y) + np.exp(y * (s_i - s_j) + gamma) + np.exp(y * (s_i - s_j) + q * y) + np.exp(q * y)))  # noqa
            gradient[labels + performer_idx] = (y * np.exp(q * y) * (np.exp(s_i * y) + np.exp(s_j * y))) / ((np.exp(q * y) + 1) * (np.exp(y * (s_i + q) + gamma) + np.exp(s_i * y + gamma) + np.exp(y * (s_i + q)) + np.exp(y * (s_j + q))))  # noqa
            gradient[labels + performers + performer_idx] = (np.exp(gamma) * (np.exp(s_i * y) - np.exp(y * (s_j + q)))) / ((np.exp(gamma) + 1) * (np.exp(y * (s_i + q) + gamma) + np.exp(s_i * y + gamma) + np.exp(y * (s_i + q)) + np.exp(y * (s_j + q))))  # noqa
        return -gradient
