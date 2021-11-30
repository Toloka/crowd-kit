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
    """Closest to Average - chooses the output with the embedding closest to the average embedding"""

    # embeddings_and_outputs_
    scores_: annotations.TASKS_LABEL_SCORES

    distance: Callable[[np.ndarray, np.ndarray], float] = attr.ib()

    @manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS = None,
            true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> Annotation(type='ClosestToAverage', title='self'):

        data = data[['task', 'performer', 'output', 'embedding']]
        if aggregated_embeddings is None:
            group = data.groupby('task')
            # we don't use .mean() because it does not work with np.array in older pandas versions
            avg_embeddings = group.embedding.apply(np.sum) / group.performer.count()
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
        return self.fit(data, aggregated_embeddings).scores_

    @manage_docstring
    def fit_predict(
        self,
        data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS = None
    ) -> annotations.TASKS_EMBEDDINGS_AND_OUTPUTS:
        return self.fit(data, aggregated_embeddings).embeddings_and_outputs_
