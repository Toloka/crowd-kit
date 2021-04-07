__all__ = ['RASA']

from functools import partial
from typing import Callable

import attr
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.spatial import distance

from . import annotations
from .annotations import Annotation, manage_docstring
from .base_embedding_aggregator import BaseEmbeddingAggregator
from .closest_to_average import ClosestToAverage


_EPS = 1e-5


@attr.s
@manage_docstring
class RASA(BaseEmbeddingAggregator):
    """
    Hybrid Reliability and Representation Aware Sequence Aggregation
    Jiyi Li. 2020.
    Crowdsourced Text Sequence Aggregation based on Hybrid Reliability and Representation.
    Proceedings of the 43rd International ACM SIGIR Conference on Research and Development
    in Information Retrieval (SIGIR ’20), July 25–30, 2020, Virtual Event, China. ACM, New York, NY, USA,

    https://doi.org/10.1145/3397271.3401239
    """

    n_iter: int = attr.ib(default=100)
    alpha: float = attr.ib(default=0.05)

    @staticmethod
    @manage_docstring
    def _aggregate_embeddings(data: annotations.EMBEDDED_DATA, skills: annotations.SKILLS,
                              true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> annotations.TASKS_EMBEDDINGS:
        """Calculates weighted average of embeddings for each task."""
        data = data.join(skills.rename('skill'), on='performer')
        data['weighted_embedding'] = data.skill * data.embedding
        group = data.groupby('task')
        aggregated_embeddings = (group.weighted_embedding.apply(np.sum) / group.skill.sum())
        aggregated_embeddings.update(true_embeddings)
        return aggregated_embeddings

    @staticmethod
    @manage_docstring
    def _update_skills(data: annotations.EMBEDDED_DATA, aggregated_embeddings: annotations.TASKS_EMBEDDINGS,
                       prior_skills: annotations.TASKS_EMBEDDINGS) -> annotations.SKILLS:
        """Estimates global reliabilities by aggregated embeddings."""
        data = data.join(aggregated_embeddings.rename('aggregated_embedding'), on='task')
        data['distance'] = ((data.embedding - data.aggregated_embedding) ** 2).apply(np.sum)
        total_distances = data.groupby('performer').distance.apply(np.sum)
        total_distances.clip(lower=_EPS, inplace=True)
        return prior_skills / total_distances

    @staticmethod
    def _cosine_distance(embedding, avg_embedding):
        if not embedding.any() or not avg_embedding.any():
            return float('inf')
        return distance.cosine(embedding, avg_embedding)

    @manage_docstring
    def _apply(self, data: annotations.EMBEDDED_DATA, true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> Annotation(type='RASA', title='self'):
        cta = ClosestToAverage(distance=self._cosine_distance)
        cta.fit(data, skills=self.skills_, true_embeddings=true_embeddings)
        self.scores_ = cta.scores_
        self.outputs_ = cta.outputs_
        return self

    @manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA, true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> Annotation(type='RASA', title='self'):
        data = data[['task', 'performer', 'embedding']]

        # What we call skills here is called reliabilities in the paper
        prior_skills = data.performer.value_counts().apply(partial(sps.chi2.isf, self.alpha / 2))
        skills = pd.Series(1.0, index=data.performer.unique())
        for _ in range(self.n_iter):
            aggregated_embeddings = self._aggregate_embeddings(data, skills, true_embeddings)
            skills = self._update_skills(data, aggregated_embeddings, prior_skills)

        self.prior_skills_ = prior_skills
        self.skills_ = skills
        return self

    @manage_docstring
    def fit_predict_scores(self, data: annotations.EMBEDDED_DATA, true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> annotations.TASKS_LABEL_PROBAS:
        return self.fit(data, true_embeddings)._apply(data, true_embeddings).scores_

    @manage_docstring
    def fit_predict(self, data: annotations.EMBEDDED_DATA, true_embeddings: annotations.TASKS_EMBEDDINGS = None) -> annotations.TASKS_LABELS:
        return self.fit(data, true_embeddings)._apply(data, true_embeddings).outputs_


@attr.s
class TextRASA:

    encoder: attr.ib(type=Callable)
    n_iter: int = attr.ib(default=100)
    alpha: float = attr.ib(default=0.05)

    def __init__(self, encoder: Callable, *args, **kwargs):
        self.encoder = encoder
        self._rasa = RASA(self.n_iter, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._rasa, name)

    @manage_docstring
    def fit(self, data: annotations.EMBEDDED_DATA, true_objects=None) -> Annotation(type='TextRASA', title='self'):
        self._rasa.fit(self._encode(data), true_objects)
        return self

    # TODO: not labeled data
    @manage_docstring
    def fit_predict_scores(self, data: annotations.LABELED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABEL_PROBAS:
        return self._rasa.fit_predict_scores(self._encode(data), skills)

    @manage_docstring
    def fit_predict(self, data: annotations.LABELED_DATA, skills: annotations.SKILLS = None) -> annotations.TASKS_LABELS:
        return self._rasa.fit_predict(self._encode(data), skills)

    def _encode(self, data):
        data = data[['task', 'performer', 'output']]
        data['embedding'] = data.output.apply(self.encoder)
        return data
