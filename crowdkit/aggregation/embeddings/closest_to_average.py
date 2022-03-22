__all__ = ['ClosestToAverage']

from typing import Callable

import attr
import numpy as np

from .. import annotations
from ..annotations import Annotation, manage_docstring
from ..base import BaseEmbeddingsAggregator


@attr.s
@manage_docstring
class ClosestToAverage(BaseEmbeddingsAggregator):
    """
    Closest to Average - chooses the output with the embedding closest to the average embedding.

    This method takes a `DataFrame` containing four columns: `task`, `worker`, `output`, and `embedding`.
    Here the `embedding` is a vector containing a representation of the `output`. The `output` might be any
    type of data such as text, images, NumPy arrays, etc. As the result, the method returns the output which
    embedding is the closest one to the average embedding of the task's responses.

    Args:
        distance: A callable that takes two NumPy arrays and returns a single `float` number — the distance
            between these two vectors.
    """

    # embeddings_and_outputs_
    scores_: annotations.TASKS_LABEL_SCORES

    distance: Callable[[np.ndarray, np.ndarray], float] = attr.ib()

    @manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS = None,
            true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> Annotation(type='ClosestToAverage', title='self'):
        """
        Fits the model.
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

    @manage_docstring
    def fit_predict_scores(
        self,
        data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS = None
    ) -> annotations.TASKS_LABEL_PROBAS:
        """
        Fit the model and return the estimated scores.
        """

        return self.fit(data, aggregated_embeddings).scores_

    @manage_docstring
    def fit_predict(
        self,
        data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS = None
    ) -> annotations.TASKS_EMBEDDINGS_AND_OUTPUTS:
        """
        Fit the model and return the aggregated results.
        """

        return self.fit(data, aggregated_embeddings).embeddings_and_outputs_
