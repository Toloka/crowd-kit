__all__ = [
    'RASA',
]

from typing import Any, List
from functools import partial

import attr
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as sps
from scipy.spatial import distance

from .closest_to_average import ClosestToAverage
from ..base import BaseEmbeddingsAggregator

_EPS = 1e-5


@attr.s
class RASA(BaseEmbeddingsAggregator):
    r"""Reliability Aware Sequence Aggregation.

    RASA estimates *global* workers' reliabilities $\beta$ that are initialized by ones.

    Next, the algorithm iteratively performs two steps:
    1. For each task, estimate the aggregated embedding: $\hat{e}_i = \frac{\sum_k
    \beta_k e_i^k}{\sum_k \beta_k}$
    2. For each worker, estimate the global reliability: $\beta_k = \frac{\chi^2_{(\alpha/2,
    |\mathcal{V}_k|)}}{\sum_i\left(\|e_i^k - \hat{e}_i\|^2\right)}$, where $\mathcal{V}_k$
    is a set of tasks completed by the worker $k$

    Finally, the aggregated result is the output which embedding is
    the closest one to the $\hat{e}_i$.

    Jiyi Li.
    A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
    *Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*,
    pages 24–28 Hong Kong, China, November 3, 2019.
    <https://doi.org/10.18653/v1/D19-5904>

    Args:
        n_iter: A number of iterations.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from crowdkit.aggregation import RASA
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p2', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p3', 'b', np.array([0.0, 1.0])]
        >>>     ],
        >>>     columns=['task', 'worker', 'output', 'embedding']
        >>> )
        >>> result = RASA().fit_predict(df)

    Attributes:
        embeddings_and_outputs_ (DataFrame): Tasks' embeddings and outputs.
            A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
    """

    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-9)
    alpha: float = attr.ib(default=0.05)
    # embeddings_and_outputs_
    loss_history_: List[float] = attr.ib(init=False)

    @staticmethod
    def _aggregate_embeddings(data: pd.DataFrame, skills: pd.Series,
                              true_embeddings: pd.Series = None) -> pd.Series:
        """Calculates weighted average of embeddings for each task."""
        data = data.join(skills.rename('skill'), on='worker')
        data['weighted_embedding'] = data.skill * data.embedding
        group = data.groupby('task')
        aggregated_embeddings = (group.weighted_embedding.apply(np.sum) / group.skill.sum())
        aggregated_embeddings.update(true_embeddings)
        return aggregated_embeddings

    @staticmethod
    def _update_skills(data: pd.DataFrame, aggregated_embeddings: pd.Series,
                       prior_skills: pd.Series) -> pd.Series:
        """Estimates global reliabilities by aggregated embeddings."""
        data = data.join(aggregated_embeddings.rename('aggregated_embedding'), on='task')
        data['distance'] = ((data.embedding - data.aggregated_embedding) ** 2).apply(np.sum)
        total_distances = data.groupby('worker').distance.apply(np.sum)
        total_distances.clip(lower=_EPS, inplace=True)
        return prior_skills / total_distances

    @staticmethod
    def _cosine_distance(embedding: npt.NDArray[Any], avg_embedding: npt.NDArray[Any]) -> float:
        if not embedding.any() or not avg_embedding.any():
            return float('inf')
        return float(distance.cosine(embedding, avg_embedding))

    def _apply(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> 'RASA':
        cta = ClosestToAverage(distance=self._cosine_distance)
        cta.fit(data, aggregated_embeddings=self.aggregated_embeddings_, true_embeddings=true_embeddings)
        self.scores_ = cta.scores_
        self.embeddings_and_outputs_ = cta.embeddings_and_outputs_
        return self

    def fit(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> 'RASA':
        """Fit the model.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            RASA: self.
        """

        data = data[['task', 'worker', 'embedding']]

        if true_embeddings is not None and not true_embeddings.index.is_unique:
            raise ValueError(
                'Incorrect data in true_embeddings: multiple true embeddings for a single task are not supported.'
            )

        # What we call skills here is called reliabilities in the paper
        prior_skills = data.worker.value_counts().apply(partial(sps.chi2.isf, self.alpha / 2))
        skills = pd.Series(1.0, index=data.worker.unique())
        aggregated_embeddings = None
        last_aggregated = None

        for _ in range(self.n_iter):
            aggregated_embeddings = self._aggregate_embeddings(data, skills, true_embeddings)
            skills = self._update_skills(data, aggregated_embeddings, prior_skills)

            if last_aggregated is not None:
                delta = aggregated_embeddings - last_aggregated
                loss = (delta * delta).sum().sum() / (aggregated_embeddings * aggregated_embeddings).sum().sum()
                if loss < self.tol:
                    break
            last_aggregated = aggregated_embeddings

        self.prior_skills_ = prior_skills
        self.skills_ = skills
        self.aggregated_embeddings_ = aggregated_embeddings
        return self

    def fit_predict_scores(self, data: pd.DataFrame,
                           true_embeddings: pd.Series = None) -> pd.DataFrame:
        """Fit the model and return scores.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            DataFrame: Tasks' label scores.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """

        return self.fit(data, true_embeddings)._apply(data, true_embeddings).scores_

    def fit_predict(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> pd.DataFrame:
        """Fit the model and return aggregated outputs.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            DataFrame: Tasks' embeddings and outputs.
                A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
        """

        return self.fit(data, true_embeddings)._apply(data, true_embeddings).embeddings_and_outputs_
