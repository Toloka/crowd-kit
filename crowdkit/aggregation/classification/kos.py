__all__ = [
    'KOS'
]

import attr
import numpy as np
import pandas as pd

from ..base import BaseClassificationAggregator

_MAX = np.float_power(10, 300)


@attr.s
class KOS(BaseClassificationAggregator):
    r"""Karger-Oh-Shah aggregation model.

    The model learns the aggregated labels by...

    Karger, David R., Sewoong Oh, and Devavrat Shah. Budget-optimal task allocation for reliable crowdsourcing systems.
    Operations Research 62.1 (2014): 1-24.

    <https://arxiv.org/abs/1110.3564>

    Args:
        n_iter: The number of iterations.

    Examples:
        >>> from crowdkit.aggregation import DawidSkene
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> ds = DawidSkene(100)
        >>> result = ds.fit_predict(df)

    Attributes:
        labels_ (Optional[pd.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

    """

    n_iter: int = attr.ib(default=100)
    random_state: int = attr.ib(default=0)

    def fit(self, data: pd.DataFrame) -> 'KOS':
        """Fit the model.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            KOS: self.
        """

        np.random.seed(self.random_state)

        # Early exit
        if not data.size:
            self.labels_ = pd.Series(dtype=float)

        # Initialization
        kos_data = data.copy()
        labels = kos_data.label.unique()
        if len(labels) != 2:
            raise ValueError('KOS aggregation method is for binary classification only.')
        mapping = {labels[0]: 1, labels[1]: -1}
        kos_data.label = kos_data.label.apply(lambda x: mapping[x])
        kos_data['reliabilities'] = np.random.normal(loc=1, scale=1, size=len(kos_data))

        # Updating reliabilities
        for _ in range(self.n_iter):
            # Update inferred labels for (task, worker)
            kos_data['multiplied'] = kos_data.label * kos_data.reliabilities
            kos_data['summed'] = list(kos_data.groupby('task')['multiplied'].sum()[kos_data.task])
            # Early exit to prevent NaN
            if (np.abs(kos_data['summed']) > _MAX).any():
                break
            kos_data['inferred'] = (kos_data['summed'] - kos_data['multiplied']).astype(float)

            # Update reliabilities for (task, worker)
            kos_data['multiplied'] = kos_data.label * kos_data.inferred
            kos_data['summed'] = list(kos_data.groupby('worker')['multiplied'].sum()[kos_data.worker])
            # Early exit to prevent NaN
            if (np.abs(kos_data['summed']) > _MAX).any():
                break
            kos_data['reliabilities'] = (kos_data.summed - kos_data.multiplied).astype('float')

        kos_data['inferred'] = kos_data.label * kos_data.reliabilities
        self.labels_ = np.sign(kos_data.groupby('task')['inferred'].sum())
        back_mapping = {v: k for k, v in mapping.items()}
        self.labels_ = self.labels_.apply(lambda x: back_mapping[x])
        print(self.labels_)
        return self

    def fit_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit the model and return aggregated results.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks' most likely true label.
        """

        return self.fit(data).labels_
