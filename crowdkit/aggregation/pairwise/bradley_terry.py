__all__ = ['BradleyTerry']

from typing import Tuple, List

import attr
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..base import BasePairwiseAggregator

_EPS = np.float_power(10, -10)


@attr.s
class BradleyTerry(BasePairwiseAggregator):
    r"""The **Bradley-Terry model for paired comparisons** implements the classic algorithm
    for aggregating paired comparisons. The algorithm constructs the ranking of items based on paired comparisons.
    Given a pair of two items $i$ and $j$, the probability that $i$ is ranked higher than $j$,
    according to the probabilistic Bradley-Terry model, is
    $$
    P(i > j) = \frac{p_i}{p_i + p_j},
    $$
    where $\boldsymbol{p}$ is a vector of the positive real-valued parameters that the algorithm optimizes. These
    optimization process maximizes the log-likelihood of the outcomes of the observed comparisons using the MM algorithm:
    $$
    L(\boldsymbol{p}) = \sum_{i=1}^n\sum_{j=1}^n[w_{ij}\ln p_i - w_{ij}\ln (p_i + p_j)],
    $$
    where $w_{ij}$ denotes the number of times object $i$ has beaten object $j$.

    {% note info %}

    The Bradley-Terry model requires the comparison graph to be **strongly connected**.

    {% endnote %}

    David R. Hunter. MM Algorithms for Generalized Bradley-Terry Models.
    *Ann. Statist. Vol. 32*, 1 (2004), 384–406.

    <https://projecteuclid.org/journals/annals-of-statistics/volume-32/issue-1/MM-algorithms-for-generalized-Bradley-Terry-models/10.1214/aos/1079120141.full>

    R. A. Bradley, M. E. Terry. Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.
    *Biometrika. Vol. 39*, 3/4 (1952), 324–345.

    <https://doi.org/10.2307/2334029>

    Args:
        n_iter: The maximum number of optimization iterations.
        tol: The tolerance stopping criterion for iterative methods with a variable number of steps.
            The algorithm converges when the loss change is less than the `tol` parameter.

    Examples:
        The Bradley-Terry model requires the `DataFrame` data containing columns
        `left`, `right`, and `label`. `left` and `right` contain the identifiers of the left and
        right items respectively, `label` contains the identifiers of the items that won these
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

    Attributes:
        scores_ (Series): The label scores.
            The `pandas.Series` data is indexed by `label` and contains the corresponding label scores.
        loss_history_ (List[float]): A list of loss values during training.
    """

    n_iter: int = attr.ib()
    tol: float = attr.ib(default=1e-5)
    # scores_
    loss_history_: List[float] = attr.ib(init=False)

    def fit(self, data: pd.DataFrame) -> 'BradleyTerry':
        """Fits the model to the training data.
        Args:
            data (DataFrame): The training dataset of workers' paired comparison results
                which is represented as the `pandas.DataFrame` data containing `worker`, `left`, `right`, and `label` columns.
                Each row `label` must be equal to either the `left` or `right` column.

        Returns:
            BradleyTerry: self.
        """

        M, unique_labels = self._build_win_matrix(data)

        if not unique_labels.size:
            self.scores_ = pd.Series([], dtype=np.float64)
            return self

        T: npt.NDArray[np.int_] = M.T + M
        active: npt.NDArray[np.bool_] = T > 0

        w = M.sum(axis=1)

        Z = np.zeros_like(M, dtype=float)

        p = np.ones(M.shape[0])
        p_new = p.copy() / p.sum()

        p_old = None

        self.loss_history_ = []

        for _ in range(self.n_iter):
            P: npt.NDArray[np.float_] = np.broadcast_to(p, M.shape)  # type: ignore

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

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fits the model to the training data and returns the aggregated results.
        Args:
            data (DataFrame): The training dataset of workers' paired comparison results
                which is represented as the `pandas.DataFrame` data containing `worker`, `left`, `right`, and `label` columns.
                Each row `label` must be equal to either the `left` or `right` column.

        Returns:
            Series: The label scores.
                The `pandas.Series` data is indexed by `label` and contains the corresponding label scores.
        """
        return self.fit(data).scores_

    @staticmethod
    def _build_win_matrix(data: pd.DataFrame) -> Tuple[npt.NDArray[np.int_], npt.NDArray[np.int_]]:
        data = data[['left', 'right', 'label']]

        unique_labels, np_data = np.unique(data.values, return_inverse=True)  # type: ignore
        np_data = np_data.reshape(data.shape)

        left_wins = np_data[np_data[:, 0] == np_data[:, 2], :2].T
        right_wins = np_data[np_data[:, 1] == np_data[:, 2], 1::-1].T

        win_matrix = np.zeros((unique_labels.size, unique_labels.size), dtype='int')

        np.add.at(win_matrix, tuple(left_wins), 1)
        np.add.at(win_matrix, tuple(right_wins), 1)

        return win_matrix, unique_labels
