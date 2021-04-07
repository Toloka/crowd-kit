__all__ = ['ClosestToAverage']

from typing import Callable

import attr
import numpy as np

from . import annotations
from .annotations import Annotation, manage_docstring
from .base_aggregator import BaseAggregator


@attr.s
@manage_docstring
class ClosestToAverage(BaseAggregator):
    """Majority Vote - chooses the correct label for which more performers voted"""

    outputs_: annotations.TASKS_LABELS
    scores_: annotations.TASKS_LABEL_PROBAS

    distance: Callable[[np.array, np.array], float] = attr.ib()

    @manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA, skills: annotations.SKILLS = None,
            true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> Annotation(type='ClosestToAverage', title='self'):

        data = data[['task', 'performer', 'output', 'embedding']]
        if skills is None:
            avg_embeddings = data.groupby('task')['embedding'].avg()
        else:
            data = data.join(skills.rename('skill'), on='performer')
            data['weighted_embedding'] = data.skill * data.embedding
            group = data.groupby('task')
            avg_embeddings = (group.weighted_embedding.apply(np.sum) / group.skill.sum())

        avg_embeddings.update(true_embeddings)

        # Calculating distances (scores)
        data = data.join(avg_embeddings.rename('avg_embedding'), on='task')
        # TODO: native Python functions are slow
        data['score'] = data.apply(lambda row: self.distance(row.embedding, row.avg_embedding), axis=1)

        # Selecting best scores and outputs
        scores = data[['task', 'output', 'score']]
        # TODO: process cases when we actually have an answer in true_embeddings
        # TODO: to do that we must make true_embeddings a DataFrame with `output` column
        outputs = scores[['task', 'output']].loc[scores.groupby('task')['score'].idxmin()]

        #
        self.scores_ = scores.set_index('task')
        self.outputs_ = outputs.set_index('task').output.rename()

        return self

    @manage_docstring
    def fit_predict_scores(self, data: annotations.EMBEDDED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data, skills).scores_

    @manage_docstring
    def fit_predict(self, data: annotations.EMBEDDED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABELS:
        return self.fit(data, skills).outputs_
