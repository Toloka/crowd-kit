__all__ = [
    'glue_similarity',
    'HRRASA',
]
from typing import Any, Iterator, Tuple

from functools import partial
import nltk.translate.gleu_score as gleu
import numpy as np
import pandas as pd
import scipy.stats as sps
from scipy.spatial import distance
import attr

from ..annotations import (
    Annotation,
    manage_docstring,
    TASKS_EMBEDDINGS,
    TASKS_EMBEDDINGS_AND_OUTPUTS,
    EMBEDDED_DATA,
    SKILLS,
    TASKS_LABEL_SCORES,
    WEIGHTS,
)
from ..base import BaseClassificationAggregator
from .closest_to_average import ClosestToAverage

_EPS = 1e-5


def glue_similarity(hyp, ref):
    return gleu.sentence_gleu([hyp.split()], ref)


@attr.s
class HRRASA(BaseClassificationAggregator):
    """
    Hybrid Reliability and Representation Aware Sequence Aggregation
    Jiyi Li. 2020.
    Crowdsourced Text Sequence Aggregation based on Hybrid Reliability and Representation.
    Proceedings of the 43rd International ACM SIGIR Conference on Research and Development
    in Information Retrieval (SIGIR ’20), July 25–30, 2020, Virtual Event, China. ACM, New York, NY, USA,

    https://doi.org/10.1145/3397271.3401239
    """

    n_iter: int = attr.ib(default=100)
    lambda_emb: float = attr.ib(default=0.5)
    lambda_out: float = attr.ib(default=0.5)
    alpha: float = attr.ib(default=0.05)
    calculate_ranks: bool = attr.ib(default=False)
    _output_similarity = attr.ib(default=glue_similarity)
    # embeddings_and_outputs_

    @manage_docstring
    def fit(self, data: EMBEDDED_DATA, true_embeddings: TASKS_EMBEDDINGS = None) -> Annotation(type='HRRASA',
                                                                                               title='self'):
        data = data[['task', 'performer', 'embedding', 'output']]
        data, single_overlap_tasks = self._filter_single_overlap(data)
        data = self._get_local_skills(data)

        prior_skills = data.performer.value_counts().apply(partial(sps.chi2.isf, self.alpha / 2))
        skills = pd.Series(1.0, index=data.performer.unique())
        weights = self._calc_weights(data, skills)
        aggregated_embeddings = None

        for _ in range(self.n_iter):
            aggregated_embeddings = self._aggregate_embeddings(data, weights, true_embeddings)
            skills = self._update_skills(data, aggregated_embeddings, prior_skills)
            weights = self._calc_weights(data, skills)

        self.prior_skills_ = prior_skills
        self.skills_ = skills
        self.weights_ = weights
        self.aggregated_embeddings_ = aggregated_embeddings
        if self.calculate_ranks:
            self.ranks_ = self._rank_outputs(data, skills)

        self._fill_single_overlap_tasks_info(single_overlap_tasks)
        return self

    @manage_docstring
    def fit_predict_scores(self, data: EMBEDDED_DATA, true_embeddings: TASKS_EMBEDDINGS = None) -> TASKS_LABEL_SCORES:
        return self.fit(data, true_embeddings)._apply(data, true_embeddings).scores_

    @manage_docstring
    def fit_predict(self, data: EMBEDDED_DATA, true_embeddings: TASKS_EMBEDDINGS = None) -> TASKS_EMBEDDINGS_AND_OUTPUTS:
        return self.fit(data, true_embeddings)._apply(data, true_embeddings).embeddings_and_outputs_

    @staticmethod
    def _cosine_distance(embedding, avg_embedding):
        if not embedding.any() or not avg_embedding.any():
            return float('inf')
        return distance.cosine(embedding, avg_embedding)

    @manage_docstring
    def _apply(self, data: EMBEDDED_DATA, true_embeddings: TASKS_EMBEDDINGS = None) -> Annotation(type='HRRASA',
                                                                                                  title='self'):
        cta = ClosestToAverage(distance=self._cosine_distance)
        cta.fit(data, aggregated_embeddings=self.aggregated_embeddings_, true_embeddings=true_embeddings)
        self.scores_ = cta.scores_
        self.embeddings_and_outputs_ = cta.embeddings_and_outputs_
        return self

    @staticmethod
    @manage_docstring
    def _aggregate_embeddings(data: EMBEDDED_DATA, weights: WEIGHTS,
                              true_embeddings: TASKS_EMBEDDINGS = None) -> TASKS_EMBEDDINGS:
        """Calculates weighted average of embeddings for each task."""
        data = data.join(weights, on=['task', 'performer'])
        data['weighted_embedding'] = data.weight * data.embedding
        group = data.groupby('task')
        aggregated_embeddings = (group.weighted_embedding.apply(np.sum) / group.weight.sum())
        aggregated_embeddings.update(true_embeddings)
        return aggregated_embeddings

    def _distance_from_aggregated(self, answers: EMBEDDED_DATA):
        """Calculates the square of Euclidian distance from aggregated embedding for each answer.
        """
        with_task_aggregate = answers.set_index('task')
        with_task_aggregate['task_aggregate'] = self.aggregated_embeddings_
        with_task_aggregate['distance'] = with_task_aggregate.apply(lambda row: np.sum((row['embedding'] - row['task_aggregate']) ** 2), axis=1)
        with_task_aggregate['distance'] = with_task_aggregate['distance'].replace({0.0: 1e-5})  # avoid division by zero
        return with_task_aggregate.reset_index()

    def _rank_outputs(self, data: EMBEDDED_DATA, skills: SKILLS) -> TASKS_LABEL_SCORES:
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
    @manage_docstring
    def _calc_weights(data: EMBEDDED_DATA, performer_skills: SKILLS) -> WEIGHTS:
        """Calculates the weight for every embedding according to its local and global skills.
        """
        data = data.set_index('performer')
        data['performer_skill'] = performer_skills
        data = data.reset_index()
        data['weight'] = data['performer_skill'] * data['local_skill']
        return data[['task', 'performer', 'weight']].set_index(['task', 'performer'])

    @staticmethod
    @manage_docstring
    def _update_skills(data: EMBEDDED_DATA, aggregated_embeddings: TASKS_EMBEDDINGS,
                       prior_skills: SKILLS) -> SKILLS:
        """Estimates global reliabilities by aggregated embeddings."""
        data = data.join(aggregated_embeddings.rename('aggregated_embedding'), on='task')
        data['distance'] = ((data.embedding - data.aggregated_embedding) ** 2).apply(np.sum)
        data['distance'] = data['distance'] / data['local_skill']
        total_distances = data.groupby('performer').distance.apply(np.sum)
        total_distances.clip(lower=_EPS, inplace=True)
        return prior_skills / total_distances

    @manage_docstring
    def _get_local_skills(self, data: EMBEDDED_DATA) -> EMBEDDED_DATA:
        """Computes local (relative) skills for each task's answer.
        """
        index = []
        local_skills = []
        processed_pairs = set()
        for task, task_answers in data.groupby('task'):
            for performer, skill in self._local_skills_on_task(task_answers):
                if (task, performer) not in processed_pairs:
                    local_skills.append(skill)
                    index.append((task, performer))
                    processed_pairs.add((task, performer))
        data = data.set_index(['task', 'performer'])
        local_skills = pd.Series(local_skills, index=pd.MultiIndex.from_tuples(index, names=['task', 'performer']), dtype=float)
        data['local_skill'] = local_skills
        return data.reset_index()

    def _local_skills_on_task(self, task_answers: pd.DataFrame) -> Iterator[Tuple[Any, float]]:
        overlap = len(task_answers)

        for _, cur_row in task_answers.iterrows():
            performer = cur_row['performer']
            emb_sum = 0.0
            seq_sum = 0.0
            emb = cur_row['embedding']
            seq = cur_row['output']
            emb_norm = np.sum(emb ** 2)
            for __, other_row in task_answers.iterrows():
                if other_row['performer'] == performer:
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

            yield performer, self.lambda_emb * emb_sum + self.lambda_out * seq_sum

    @manage_docstring
    def _filter_single_overlap(self, data: EMBEDDED_DATA):
        """Filter skills, embeddings, weights and ranks for single overlap tasks that couldn't be processed by HRASSA
        """

        single_overlap_task_ids = []
        for task, task_answers in data.groupby('task'):
            if len(task_answers) == 1:
                single_overlap_task_ids.append(task)
        data = data.set_index('task')
        return data.drop(single_overlap_task_ids).reset_index(), data.loc[single_overlap_task_ids].reset_index()

    @manage_docstring
    def _fill_single_overlap_tasks_info(self, single_overlap_tasks: EMBEDDED_DATA):
        """Fill skills, embeddings, weights and ranks for single overlap tasks
        """

        performers_to_append = []
        aggregated_embeddings_to_append = {}
        weights_to_append = []
        ranks_to_append = []
        for row in single_overlap_tasks.itertuples():
            if row.performer not in self.prior_skills_:
                performers_to_append.append(row.performer)
            if row.task not in self.aggregated_embeddings_:
                aggregated_embeddings_to_append[row.task] = row.embedding
                weights_to_append.append({'task': row.task, 'performer': row.performer, 'weight': np.nan})
                ranks_to_append.append({'task': row.task, 'output': row.output, 'rank': np.nan})

        self.prior_skills_ = self.prior_skills_.append(pd.Series(np.nan, index=performers_to_append))
        self.skills_ = self.skills_.append(pd.Series(np.nan, index=performers_to_append))
        self.aggregated_embeddings_ = self.aggregated_embeddings_.append(pd.Series(aggregated_embeddings_to_append))
        self.weights_ = self.weights_.append(pd.DataFrame(weights_to_append))
        if hasattr(self, 'ranks_'):
            self.ranks_ = self.ranks_.append(pd.DataFrame(ranks_to_append))

        return
