__all__ = ['BradleyTerry']


from typing import Tuple

import attr
import numpy as np
import pandas as pd

from . import annotations
from .annotations import Annotation, manage_docstring
from .base_aggregator import BaseAggregator

_EPS = np.float_power(10, -10)


@attr.s
@manage_docstring
class BradleyTerry(BaseAggregator):
    """
    Bradley-Terry, the classic algorithm for aggregating pairwise comparisons.

    David R. Hunter. 2004.
    MM algorithms for generalized Bradley-Terry models
    Ann. Statist., Vol. 32, 1 (2004): 384–406.

    Bradley, R. A. and Terry, M. E. 1952.
    Rank analysis of incomplete block designs. I. The method of paired comparisons.
    Biometrika, Vol. 39 (1952): 324–345.
    """

    n_iter: int = attr.ib()
    result_: annotations.LABEL_SCORES = attr.ib(init=False)

    @manage_docstring
    def fit(self, data: annotations.PAIRWISE_DATA) -> Annotation(type='BradleyTerry', title='self'):
        M, unique_labels = self._build_win_matrix(data)

        if not unique_labels.size:
            self.result_ = pd.Series([])
            return self

        T = M.T + M
        active = T > 0

        w = M.sum(axis=1)

        Z = np.zeros_like(M, dtype=float)

        p = np.ones(M.shape[0])
        p_new = p.copy() / p.sum()

        for _ in range(self.n_iter):
            P = np.broadcast_to(p, M.shape)

            Z[active] = T[active] / (P[active] + P.T[active])

            p_new[:] = w
            p_new /= Z.sum(axis=0)
            p_new /= p_new.sum()

            p[:] = p_new

        self.result_ = pd.Series(p_new, index=unique_labels)

        return self

    @manage_docstring
    def fit_predict(self, data: annotations.PAIRWISE_DATA) -> annotations.LABEL_SCORES:
        return self.fit(data).result_

    @staticmethod
    @manage_docstring
    def _build_win_matrix(data: annotations.PAIRWISE_DATA) -> Tuple[np.ndarray, np.ndarray]:
        data = data[['left', 'right', 'label']]

        unique_labels, np_data = np.unique(data.values, return_inverse=True)
        np_data = np_data.reshape(data.shape)

        left_wins = np_data[np_data[:, 0] == np_data[:, 2], :2].T
        right_wins = np_data[np_data[:, 1] == np_data[:, 2], 1::-1].T

        win_matrix = np.zeros((unique_labels.size, unique_labels.size), dtype='int')

        np.add.at(win_matrix, tuple(left_wins), 1)
        np.add.at(win_matrix, tuple(right_wins), 1)

        return win_matrix, unique_labels
