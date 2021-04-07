__all__ = ['BaseEmbeddingAggregator']


from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from tqdm.auto import tqdm

from .base_aggregator import BaseAggregator


class BaseEmbeddingAggregator(BaseAggregator):
    """Base class for aggregation algorithms that operate with embeddings of performers answers.

    Attributes:
        aggregated_embeddings_ (Optional[pd.Series]): result of embeddings aggregation for each task.
        golden_embeddings_: (Optional[pd.Series]): embeddings of golden outputs if the golden outputs are provided.
    """

    def __init__(self, encoder: Any, silent: bool):
        self.encoder = encoder
        self.silent = silent
        self.aggregated_embeddings_: Optional[pd.Series] = None
        self.golden_embeddings_: Optional[pd.Series] = None

    def _answers_base_checks(self, answers: pd.DataFrame):
        if not isinstance(answers, pd.DataFrame):
            raise TypeError('Working only with pandas DataFrame')
        assert 'task' in answers, 'There is no "task" column in answers'
        assert 'performer' in answers, 'There is no "performer" column in answers'
        assert 'output' in answers, 'There is no "output" column in answers'

    def _get_embeddings(self, answers: pd.DataFrame):
        """Obtaines embeddings for performers answers.
        """
        if not self.silent:
            tqdm.pandas()
            answers['embedding'] = answers.output.progress_apply(self.encoder.encode)
        else:
            answers['embedding'] = answers.output.apply(self.encoder.encode)

    def _get_golden_embeddings(self, answers: pd.DataFrame):
        """Processes embeddings for golden outputs.
        """
        if 'golden_embedding' not in answers:
            golden_tasks = answers[answers['golden'].notna()][['task', 'golden']].drop_duplicates().set_index('task')
            golden_tasks['golden_embedding'] = golden_tasks.golden.apply(self.encoder.encode)
        else:
            golden_tasks = answers[answers['golden'].notna()][['task', 'golden', 'golden_embedding']].drop_duplicates(['task']).set_index('task')
        self.golden_embeddings_ = golden_tasks['golden_embedding']

    def _init_performers_reliabilities(self, answers: pd.DataFrame):
        """Initialize performers reliabilities by ones.
        """
        performers = pd.unique(answers.performer)
        self.performers_reliabilities_ = pd.Series(np.ones(len(performers)), index=performers)

    def _aggregate_embeddings(self, answers: pd.DataFrame):
        """Calculates weighted average of embeddings for each task.
        """
        answers['weighted_embeddings'] = answers.score * answers.embedding
        self.aggregated_embeddings_ = answers.groupby('task').weighted_embeddings.apply(np.sum) / answers.groupby('task').score.sum()
        if self.golden_embeddings_ is not None:
            for task, embedding in self.golden_embeddings_.iteritems():
                self.aggregated_embeddings_[task] = embedding

    def _distance_from_aggregated(self, answers: pd.DataFrame):
        """Calculates the square of Euclidian distance from aggregated embedding for each answer.
        """
        with_task_aggregate = answers.set_index('task')
        with_task_aggregate['task_aggregate'] = self.aggregated_embeddings_
        with_task_aggregate['distance'] = with_task_aggregate.apply(lambda row: np.sum((row['embedding'] - row['task_aggregate']) ** 2), axis=1)
        with_task_aggregate['distance'] = with_task_aggregate['distance'].replace({0.0: 1e-5})  # avoid division by zero
        return with_task_aggregate.reset_index()

    def _choose_nearest_output(self, answers, metric='cosine'):
        """Choses nearest performers answer according to aggregated embeddings.
        """
        aggregated_output = []
        tasks = []
        for task, assignments in answers.groupby('task'):
            embeddigs = np.array(list(assignments['embedding']))
            outputs = list(assignments['output'])
            knn = NearestNeighbors(algorithm='brute', metric='cosine').fit(embeddigs)
            _, res_ind = knn.kneighbors([self.aggregated_embeddings_[task]], 1)
            aggregated_output.append(outputs[res_ind[0][0]])
            tasks.append(task)
        return pd.Series(aggregated_output, index=tasks)
