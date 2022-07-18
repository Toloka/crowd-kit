__all__ = ['ClosestToAverage']

from typing import Callable, Optional, Any

import attr
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..base import BaseEmbeddingsAggregator


@attr.s
class ClosestToAverage(BaseEmbeddingsAggregator):
    """Closest to Average - chooses the output with the embedding closest to the average embedding.

    This method takes a `DataFrame` containing four columns: `task`, `worker`, `output`, and `embedding`.
    Here the `embedding` is a vector containing a representation of the `output`. The `output` might be any
    type of data such as text, images, NumPy arrays, etc. As the result, the method returns the output which
    embedding is the closest one to the average embedding of the task's responses.

    Args:
        distance: A callable that takes two NumPy arrays and returns a single `float` number — the distance
            between these two vectors.

    Attributes:
        embeddings_and_outputs_ (DataFrame): Tasks' embeddings and outputs.
            A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.

        scores_ (DataFrame): Tasks' label scores.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the score of `label` for `task`.
    """

    # embeddings_and_outputs_
    scores_: pd.DataFrame

    distance: Callable[[npt.NDArray[Any], npt.NDArray[Any]], float] = attr.ib()

    def fit(self, data: pd.DataFrame, aggregated_embeddings: Optional[pd.Series] = None,
            true_embeddings: pd.Series = None) -> 'ClosestToAverage':
        """Fits the model.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            aggregated_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            ClosestToAverage: self.
        """

        if true_embeddings is not None and not true_embeddings.index.is_unique:
            raise ValueError(
                'Incorrect data in true_embeddings: multiple true embeddings for a single task are not supported.'
            )

        data = data[['task', 'worker', 'output', 'embedding']]
        if aggregated_embeddings is None:
            group = data.groupby('task')
            # we don't use .mean() because it does not work with np.array in older pandas versions
            avg_embeddings = group.embedding.apply(np.sum) / group.worker.count()
            avg_embeddings.update(true_embeddings)
        else:
            avg_embeddings = aggregated_embeddings

        # Calculating distances (scores)
        data = data.join(avg_embeddings.rename('avg_embedding'), on='task')
        # TODO: native Python functions are slow
        data['score'] = data.apply(lambda row: self.distance(row.embedding, row.avg_embedding), axis=1)

        # Selecting best scores and outputs
        scores = data[['task', 'output', 'score', 'embedding']]
        # TODO: process cases when we actually have an answer in true_embeddings
        # TODO: to do that we must make true_embeddings a DataFrame with `output` column
        embeddings_and_outputs = scores[['task', 'output', 'embedding']].loc[scores.groupby('task')['score'].idxmin()]

        #
        self.scores_ = scores.set_index('task')
        self.embeddings_and_outputs_ = embeddings_and_outputs.set_index('task')

        return self

    def fit_predict_scores(self, data: pd.DataFrame, aggregated_embeddings: pd.Series = None) -> pd.DataFrame:
        """Fit the model and return the estimated scores.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            aggregated_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """

        return self.fit(data, aggregated_embeddings).scores_

    def fit_predict(self, data: pd.DataFrame, aggregated_embeddings: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Fit the model and return the aggregated results.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            aggregated_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            DataFrame: Tasks' embeddings and outputs.
                A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
        """

        return self.fit(data, aggregated_embeddings).embeddings_and_outputs_
