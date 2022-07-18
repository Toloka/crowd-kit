__all__ = [
    'glue_similarity',
    'HRRASA',
]

from functools import partial
from typing import Any, Tuple, List, Optional, Callable, Generator

import attr
import nltk.translate.gleu_score as gleu
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as sps
from scipy.spatial import distance

from .closest_to_average import ClosestToAverage
from ..base import BaseClassificationAggregator

_EPS = 1e-5


def glue_similarity(hyp: str, ref: List[List[str]]) -> float:
    return float(gleu.sentence_gleu([hyp.split()], ref))


@attr.s
class HRRASA(BaseClassificationAggregator):
    r"""Hybrid Reliability and Representation Aware Sequence Aggregation.


    At the first step, the HRRASA estimates *local* workers reliabilities that represent how good is a
    worker's answer to *one particular task*. The local reliability of the worker $k$ on the task $i$ is
    denoted by $\gamma_i^k$ and is calculated as a sum of two terms:
    $$
    \gamma_i^k = \lambda_{emb}\gamma_{i,emb}^k + \lambda_{out}\gamma_{i,out}^k, \; \lambda_{emb} + \lambda_{out} = 1.
    $$
    The $\gamma_{i,emb}^k$ is a reliability calculated on `embedding` and the $\gamma_{i,seq}^k$ is a
    reliability calculated on `output`.

    The $\gamma_{i,emb}^k$ is calculated by the following equation:
    $$
    \gamma_{i,emb}^k = \frac{1}{|\mathcal{U}_i| - 1}\sum_{a_i^{k'} \in \mathcal{U}_i, k \neq k'}
    \exp\left(\frac{\|e_i^k-e_i^{k'}\|^2}{\|e_i^k\|^2\|e_i^{k'}\|^2}\right),
    $$
    where $\mathcal{U_i}$ is a set of workers' responses on task $i$. The $\gamma_{i,out}^k$ makes use
    of some similarity measure $sim$ on the `output` data, e.g. GLUE similarity on texts:
    $$
    \gamma_{i,out}^k = \frac{1}{|\mathcal{U}_i| - 1}\sum_{a_i^{k'} \in \mathcal{U}_i, k \neq k'}sim(a_i^k, a_i^{k'}).
    $$

    The HRRASA also estimates *global* workers' reliabilities $\beta$ that are initialized by ones.

    Next, the algorithm iteratively performs two steps:
    1. For each task, estimate the aggregated embedding: $\hat{e}_i = \frac{\sum_k \gamma_i^k
    \beta_k e_i^k}{\sum_k \gamma_i^k \beta_k}$
    2. For each worker, estimate the global reliability: $\beta_k = \frac{\chi^2_{(\alpha/2,
    |\mathcal{V}_k|)}}{\sum_i\left(\|e_i^k - \hat{e}_i\|^2/\gamma_i^k\right)}$, where $\mathcal{V}_k$
    is a set of tasks completed by the worker $k$

    Finally, the aggregated result is the output which embedding is
    the closest one to the $\hat{e}_i$. If `calculate_ranks` is true, the method also calculates ranks for
    each workers' response as
    $$
    s_i^k = \beta_k \exp\left(-\frac{\|e_i^k - \hat{e}_i\|^2}{\|e_i^k\|^2\|\hat{e}_i\|^2}\right) + \gamma_i^k.
    $$

    Jiyi Li. Crowdsourced Text Sequence Aggregation based on Hybrid Reliability and Representation.
    *Proceedings of the 43rd International ACM SIGIR Conference on Research and Development
    in Information Retrieval (SIGIR ’20)*, July 25–30, 2020, Virtual Event, China. ACM, New York, NY, USA,

    <https://doi.org/10.1145/3397271.3401239>

    Args:
        n_iter: A number of iterations.
        lambda_emb: A weight of reliability calculated on embeddigs.
        lambda_out: A weight of reliability calculated on outputs.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.
        calculate_ranks: If true, calculate additional attribute `ranks_`.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from crowdkit.aggregation import HRRASA
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p2', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p3', 'b', np.array([0.0, 1.0])]
        >>>     ],
        >>>     columns=['task', 'worker', 'output', 'embedding']
        >>> )
        >>> result = HRRASA().fit_predict(df)
    """

    n_iter: int = attr.ib(default=100)
    tol: float = attr.ib(default=1e-9)
    lambda_emb: float = attr.ib(default=0.5)
    lambda_out: float = attr.ib(default=0.5)
    alpha: float = attr.ib(default=0.05)
    calculate_ranks: bool = attr.ib(default=False)
    _output_similarity: Callable[[str, List[List[str]]], float] = attr.ib(default=glue_similarity)
    # embeddings_and_outputs_
    loss_history_: List[float] = attr.ib(init=False)

    def fit(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> 'HRRASA':
        """Fit the model.

        Args:
            data (DataFrame): Workers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.

        Returns:
            HRRASA: self.
        """

        if true_embeddings is not None and not true_embeddings.index.is_unique:
            raise ValueError(
                'Incorrect data in true_embeddings: multiple true embeddings for a single task are not supported.'
            )

        data = data[['task', 'worker', 'embedding', 'output']]
        data, single_overlap_tasks = self._filter_single_overlap(data)
        data = self._get_local_skills(data)

        prior_skills = data.worker.value_counts().apply(partial(sps.chi2.isf, self.alpha / 2))
        skills = pd.Series(1.0, index=data.worker.unique())
        weights = self._calc_weights(data, skills)
        aggregated_embeddings = self._aggregate_embeddings(data, weights, true_embeddings)
        self.loss_history_ = []
        last_aggregated = None

        if len(data) > 0:
            for _ in range(self.n_iter):
                aggregated_embeddings = self._aggregate_embeddings(data, weights, true_embeddings)
                skills = self._update_skills(data, aggregated_embeddings, prior_skills)
                weights = self._calc_weights(data, skills)

                if last_aggregated is not None:
                    delta = aggregated_embeddings - last_aggregated
                    loss = (delta * delta).sum().sum() / (aggregated_embeddings * aggregated_embeddings).sum().sum()
                    self.loss_history_.append(loss)
                    if loss < self.tol:
                        break
                last_aggregated = aggregated_embeddings

        self.prior_skills_ = prior_skills
        self.skills_ = skills
        self.weights_ = weights
        self.aggregated_embeddings_ = aggregated_embeddings
        if self.calculate_ranks:
            self.ranks_ = self._rank_outputs(data, skills)

        if len(single_overlap_tasks) > 0:
            self._fill_single_overlap_tasks_info(single_overlap_tasks)
        return self

    def fit_predict_scores(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> pd.DataFrame:
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

    def fit_predict(self, data: pd.DataFrame, true_embeddings: Optional[pd.Series] = None) -> pd.DataFrame:
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

    @staticmethod
    def _cosine_distance(embedding: npt.NDArray[Any], avg_embedding: npt.NDArray[Any]) -> float:
        if not embedding.any() or not avg_embedding.any():
            return float('inf')
        return float(distance.cosine(embedding, avg_embedding))

    def _apply(self, data: pd.DataFrame, true_embeddings: pd.Series = None) -> 'HRRASA':
        cta = ClosestToAverage(distance=self._cosine_distance)
        cta.fit(data, aggregated_embeddings=self.aggregated_embeddings_, true_embeddings=true_embeddings)
        self.scores_ = cta.scores_
        self.embeddings_and_outputs_ = cta.embeddings_and_outputs_
        return self

    @staticmethod
    def _aggregate_embeddings(data: pd.DataFrame, weights: pd.DataFrame,
                              true_embeddings: Optional[pd.Series] = None) -> pd.Series:
        """Calculates weighted average of embeddings for each task."""
        data = data.join(weights, on=['task', 'worker'])
        data['weighted_embedding'] = data.weight * data.embedding
        group = data.groupby('task')
        aggregated_embeddings = pd.Series((group.weighted_embedding.apply(np.sum) / group.weight.sum()),
                                          dtype=np.float64)
        if true_embeddings is None:
            true_embeddings = pd.Series([], dtype=np.float64)
        aggregated_embeddings.update(true_embeddings)
        return aggregated_embeddings

    def _distance_from_aggregated(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Calculates the square of Euclidian distance from aggregated embedding for each answer.
        """
        with_task_aggregate = answers.set_index('task')
        with_task_aggregate['task_aggregate'] = self.aggregated_embeddings_
        with_task_aggregate['distance'] = with_task_aggregate.apply(
            lambda row: np.sum((row['embedding'] - row['task_aggregate']) ** 2), axis=1)
        with_task_aggregate['distance'] = with_task_aggregate['distance'].replace({0.0: 1e-5})  # avoid division by zero
        return with_task_aggregate.reset_index()

    def _rank_outputs(self, data: pd.DataFrame, skills: pd.Series) -> pd.DataFrame:
        """Returns ranking score for each record in `data` data frame.
        """

        if not data.size:
            return pd.DataFrame(columns=['task', 'output', 'rank'])

        data = self._distance_from_aggregated(data)
        data['norms_prod'] = data.apply(lambda row: np.sum(row['embedding'] ** 2) * np.sum(row['task_aggregate'] ** 2),
                                        axis=1)
        data['rank'] = skills * np.exp(-data.distance / data.norms_prod) + data.local_skill
        return data[['task', 'output', 'rank']]

    @staticmethod
    def _calc_weights(data: pd.DataFrame, worker_skills: pd.Series) -> pd.DataFrame:
        """Calculates the weight for every embedding according to its local and global skills.
        """
        data = data.set_index('worker')
        data['worker_skill'] = worker_skills
        data = data.reset_index()
        data['weight'] = data['worker_skill'] * data['local_skill']
        return data[['task', 'worker', 'weight']].set_index(['task', 'worker'])

    @staticmethod
    def _update_skills(data: pd.DataFrame, aggregated_embeddings: pd.Series, prior_skills: pd.Series) -> pd.Series:
        """Estimates global reliabilities by aggregated embeddings."""
        data = data.join(aggregated_embeddings.rename('aggregated_embedding'), on='task')
        data['distance'] = ((data.embedding - data.aggregated_embedding) ** 2).apply(np.sum)
        data['distance'] = data['distance'] / data['local_skill']
        total_distances = data.groupby('worker').distance.apply(np.sum)
        total_distances.clip(lower=_EPS, inplace=True)
        return prior_skills / total_distances

    def _get_local_skills(self, data: pd.DataFrame) -> pd.DataFrame:
        """Computes local (relative) skills for each task's answer.
        """
        index = []
        local_skills = []
        processed_pairs = set()
        for task, task_answers in data.groupby('task'):
            for worker, skill in self._local_skills_on_task(task_answers):
                if (task, worker) not in processed_pairs:
                    local_skills.append(skill)
                    index.append((task, worker))
                    processed_pairs.add((task, worker))
        data = data.set_index(['task', 'worker'])
        local_skills = pd.Series(local_skills, index=pd.MultiIndex.from_tuples(index, names=['task', 'worker']),
                                 dtype=float)
        data['local_skill'] = local_skills
        return data.reset_index()

    def _local_skills_on_task(self, task_answers: pd.DataFrame) -> Generator[Tuple[Any, float], None, None]:
        overlap = len(task_answers)

        for _, cur_row in task_answers.iterrows():
            worker = cur_row['worker']
            emb_sum = 0.0
            seq_sum = 0.0
            emb = cur_row['embedding']
            seq = cur_row['output']
            emb_norm = np.sum(emb ** 2)
            for __, other_row in task_answers.iterrows():
                if other_row['worker'] == worker:
                    continue
                other_emb = other_row['embedding']
                other_seq = other_row['output']

                # embeddings similarity
                diff_norm = np.sum((emb - other_emb) ** 2)
                other_norm = np.sum(other_emb ** 2)
                emb_sum += np.exp(-diff_norm / (emb_norm * other_norm))

                # sequence similarity
                seq_sum += self._output_similarity(seq, other_seq)
            emb_sum /= (overlap - 1)
            seq_sum /= (overlap - 1)

            yield worker, self.lambda_emb * emb_sum + self.lambda_out * seq_sum

    def _filter_single_overlap(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filter skills, embeddings, weights and ranks for single overlap tasks that couldn't be processed by HRASSA
        """

        single_overlap_task_ids = []
        for task, task_answers in data.groupby('task'):
            if len(task_answers) == 1:
                single_overlap_task_ids.append(task)
        data = data.set_index('task')
        return data.drop(single_overlap_task_ids).reset_index(), data.loc[single_overlap_task_ids].reset_index()

    def _fill_single_overlap_tasks_info(self, single_overlap_tasks: pd.DataFrame) -> None:
        """Fill skills, embeddings, weights and ranks for single overlap tasks
        """

        workers_to_append = []
        aggregated_embeddings_to_append = {}
        weights_to_append = []
        ranks_to_append = []
        for row in single_overlap_tasks.itertuples():
            if row.worker not in self.prior_skills_:
                workers_to_append.append(row.worker)
            if row.task not in self.aggregated_embeddings_:
                aggregated_embeddings_to_append[row.task] = row.embedding
                weights_to_append.append({'task': row.task, 'worker': row.worker, 'weight': np.nan})
                ranks_to_append.append({'task': row.task, 'output': row.output, 'rank': np.nan})

        self.prior_skills_ = pd.concat([self.prior_skills_, pd.Series(np.nan, index=workers_to_append)])
        self.skills_ = pd.concat([self.skills_, pd.Series(np.nan, index=workers_to_append)])
        self.aggregated_embeddings_ = pd.concat(
            [self.aggregated_embeddings_, pd.Series(aggregated_embeddings_to_append)])
        self.weights_ = pd.concat([self.weights_, pd.DataFrame(weights_to_append)])
        if hasattr(self, 'ranks_'):
            self.ranks_ = self.ranks_.append(pd.DataFrame(ranks_to_append))
