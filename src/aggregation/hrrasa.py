from typing import Any, Iterator, Tuple, Union

import nltk.translate.gleu_score as gleu
import numpy as np
import pandas as pd
import scipy.stats as sps
from tqdm.auto import tqdm
import attr

from .base_embedding_aggregator import BaseEmbeddingAggregator


# from sentence_transformers import SentenceTransformer


def glue_similarity(hyp, ref):
    return gleu.sentence_gleu([hyp.split()], ref)


@attr.s
class HRRASA(BaseEmbeddingAggregator):
    """
    Hybrid Reliability and Representation Aware Sequence Aggregation
    Jiyi Li. 2020.
    Crowdsourced Text Sequence Aggregation based on Hybrid Reliability and Representation.
    Proceedings of the 43rd International ACM SIGIR Conference on Research and Development
    in Information Retrieval (SIGIR ’20), July 25–30, 2020, Virtual Event, China. ACM, New York, NY, USA,

    https://doi.org/10.1145/3397271.3401239
    """

    n_iter: int = attr.ib(default=100)
    encoder = attr.ib(default=None)
    output_similarity = attr.ib(default=glue_similarity)
    lambda_emb: float = attr.ib(default=0.5)
    lambda_out: float = attr.ib(default=0.5)
    alpha: float = attr.ib(default=0.05)
    silent: bool = attr.ib(default=True)

    def fit_predict(self, answers: pd.DataFrame, return_ranks: bool = False) -> Union[pd.Series, pd.DataFrame]:
        """
        Args:
            answers: A pandas.DataFrame containing `task`, `performer` and `output` columns.
                If the `embedding` column exists, embeddings are not obtained by the `encoder`.
            golden_embeddings: A pandas Series containing embeddings of golden outputs with
                `task` as an index. If is not passed, embeddings are computed by the `encoder`.
            return_ranks: if `True` returns ranking score for each of performers answers.

        Returns:
            If `return_ranks=False`, pandas.Series indexed by `task` with values — aggregated outputs.
            If `return_ranks=True`, pandas.DataFrame with columns `task`, `performer`, `output`, `rank`.
        """
        processed_answers = self._preprocess_answers(answers)
        return self._fit_impl(processed_answers, return_ranks=return_ranks)

    def _fit_impl(self, answers: pd.DataFrame, use_local_reliability: bool = True, return_ranks: bool = False) -> Union[pd.Series, pd.DataFrame]:
        self.use_local_reliability = use_local_reliability
        if use_local_reliability:
            answers = self._get_local_reliabilities(answers)
        self.performers_prior_reliability_ = answers.groupby('performer').count()['task'].apply(lambda x: sps.chi2.isf(self.alpha / 2, x))
        self.performers_reliabilities_ = pd.Series(1.0, index=pd.unique(answers.performer))
        answers = self._calc_score(answers)

        for _ in range(self.n_iter) if self.silent else tqdm(range(self.n_iter)):
            self._aggregate_embeddings(answers)
            self._update_reliabilities(answers)
            answers = self._calc_score(answers)

        if not return_ranks:
            return self._choose_nearest_output(answers)
        else:
            return self._rank_outputs(answers)

    def _rank_outputs(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Returns ranking score for each record in `answers` data frame.
        """
        answers = self._distance_from_aggregated(answers)
        answers['norms_prod'] = answers.apply(lambda row: np.sum(row['embedding'] ** 2) * np.sum(row['task_aggregate'] ** 2), axis=1)
        answers['rank'] = answers.performer_reliability * np.exp(-answers.distance / answers.norms_prod) + answers.local_reliability
        return answers[['task', 'performer', 'output', 'rank']]

    def _calc_score(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Calculates the weight for every embedding according to its local and global reliabilities.
        """
        answers = answers.set_index('performer')
        answers['performer_reliability'] = self.performers_reliabilities_
        answers = answers.reset_index()
        if self.use_local_reliability:
            answers['score'] = answers['performer_reliability'] * answers['local_reliability']
        else:
            answers['score'] = answers['performer_reliability']
        return answers

    def _update_reliabilities(self, answers: pd.DataFrame) -> None:
        """Estimates global reliabilities by aggregated embeddings.
        """
        distances = self._distance_from_aggregated(answers)
        if self.use_local_reliability:
            distances['distance'] = distances['distance'] / distances['local_reliability']
        total_distance = distances.groupby('performer').distance.apply(np.sum)
        self.performers_reliabilities_ = self.performers_prior_reliability_ / total_distance

    def _preprocess_answers(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Does basic checks for given data and obtaines embeddings if they are not provided.
        """
        assert not ('golden' in answers and 'golden_embedding' not in answers and self.encoder is None), 'Provide encoder or golden_embeddings'
        processed_answers = answers.copy(deep=False)
        if 'embedding' not in answers:
            assert self.encoder is not None, 'Provide encoder or embedding column'
            self._get_embeddings(processed_answers)
        if 'golden' in answers:
            self._get_golden_embeddings(processed_answers)
        return processed_answers

    def _get_local_reliabilities(self, answers: pd.DataFrame) -> pd.DataFrame:
        """Computes local (relative) reliabilities for each task's answer.
        """
        index = []
        local_reliabilities = []
        processed_pairs = set()
        for task, task_answers in answers.groupby('task'):
            for performer, reliability in self._local_reliabilities_on_task(task_answers):
                if (task, performer) not in processed_pairs:
                    local_reliabilities.append(reliability)
                    index.append((task, performer))
                    processed_pairs.add((task, performer))
        answers = answers.set_index(['task', 'performer'])
        local_reliabilities = pd.Series(local_reliabilities, index=pd.MultiIndex.from_tuples(index, names=['task', 'performer']))
        answers['local_reliability'] = local_reliabilities
        return answers.reset_index()

    def _local_reliabilities_on_task(self, task_answers: pd.DataFrame) -> Iterator[Tuple[Any, float]]:
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
