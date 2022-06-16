__all__ = ['BradleyTerry']


from typing import Tuple, List

import attr
import numpy as np
import pandas as pd

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BasePairwiseAggregator

_EPS = np.float_power(10, -10)


@attr.s
@manage_docstring
class BradleyTerry(BasePairwiseAggregator):
    r"""
    Bradley-Terry model for pairwise comparisons.

    The model implements the classic algorithm for aggregating pairwise comparisons.
    The algorithm constructs an items' ranking based on pairwise comparisons. Given
    a pair of two items $i$ and $j$, the probability of $i$ to be ranked higher is,
    according to the Bradley-Terry's probabilistic model,
    $$
    P(i > j) = \frac{p_i}{p_i + p_j}.
    $$
    Here $\boldsymbol{p}$ is a vector of positive real-valued parameters that the algorithm optimizes. These
    optimization process maximizes the log-likelihood of observed comparisons outcomes by the MM-algorithm:
    $$
    L(\boldsymbol{p}) = \sum_{i=1}^n\sum_{j=1}^n[w_{ij}\ln p_i - w_{ij}\ln (p_i + p_j)],
    $$
    where $w_{ij}$ denotes the number of comparisons of $i$ and $j$ "won" by $i$.

    {% note info %}

    The Bradley-Terry model needs the comparisons graph to be **strongly connected**.

    {% endnote %}

    David R. Hunter.
    MM algorithms for generalized Bradley-Terry models
    *Ann. Statist.*, Vol. 32, 1 (2004): 384–406.

    Bradley, R. A. and Terry, M. E.
    Rank analysis of incomplete block designs. I. The method of paired comparisons.
    *Biometrika*, Vol. 39 (1952): 324–345.

    Args:
        n_iter: A number of optimization iterations.

    Examples:
        The Bradley-Terry model needs the data to be a `DataFrame` containing columns
        `left`, `right`, and `label`. `left` and `right` contain identifiers of left and
        right items respectively, `label` contains identifiers of items that won these
        comparisons.

        >>> import pandas as pd
        >>> from crowdkit.aggregation import BradleyTerry
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['item1', 'item2', 'item1'],
        >>>         ['item2', 'item3', 'item2']
        >>>     ],
        >>>     columns=['left', 'right', 'label']
        >>> )
    """

    n_iter: int = attr.ib()
    tol: float = attr.ib(default=1e-5)
    # scores_
    loss_history_: List[float] = attr.ib(init=False)

    @manage_docstring
    def fit(self, data: annotations.PAIRWISE_DATA) -> Annotation(type='BradleyTerry', title='self'):
        M, unique_labels = self._build_win_matrix(data)

        if not unique_labels.size:
            self.scores_ = pd.Series([], dtype=np.float64)
            return self

        T = M.T + M
        active = T > 0

        w = M.sum(axis=1)

        Z = np.zeros_like(M, dtype=float)

        p = np.ones(M.shape[0])
        p_new = p.copy() / p.sum()

        p_old = None
        self.loss_history_ = []
        for _ in range(self.n_iter):
            P = np.broadcast_to(p, M.shape)

            Z[active] = T[active] / (P[active] + P.T[active])

            p_new[:] = w
            p_new /= Z.sum(axis=0)
            p_new /= p_new.sum()
            p[:] = p_new

            if p_old is not None:
                loss = np.abs(p_new - p_old).sum()
                if loss < self.tol:
                    break
            p_old = p_new

        self.scores_ = pd.Series(p_new, index=unique_labels)

        return self

    @manage_docstring
    def fit_predict(self, data: annotations.PAIRWISE_DATA) -> annotations.LABEL_SCORES:
        return self.fit(data).scores_

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
