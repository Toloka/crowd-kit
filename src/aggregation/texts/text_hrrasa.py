__all__ = ['TextHRRASA']
from typing import Callable

from ..annotations import (
    manage_docstring,
    TASKS_LABEL_SCORES,
    TASKS_TEXTS,
    DATA,
    TASKS_TRUE_LABELS,
)
from ..base import BaseTextsAggregator
from ..embeddings.hrrasa import HRRASA, glue_similarity


class TextHRRASA(BaseTextsAggregator):

    # texts_

    def __init__(self, encoder: Callable, n_iter: int = 100, lambda_emb: float = 0.5, lambda_out: float = 0.5,
                 alpha: float = 0.05, calculate_ranks: bool = False, output_similarity: Callable = glue_similarity):
        super().__init__()
        self.encoder = encoder
        self._hrrasa = HRRASA(n_iter, lambda_emb, lambda_out, alpha, calculate_ranks, output_similarity)

    def __getattr__(self, name):
        return getattr(self._hrrasa, name)

    @manage_docstring
    def fit_predict_scores(self, data: DATA, true_objects: TASKS_TRUE_LABELS = None) -> TASKS_LABEL_SCORES:
        return self._hrrasa.fit_predict_scores(self._encode_data(data), self._encode_true_objects(true_objects))

    @manage_docstring
    def fit_predict(self, data: DATA, true_objects: TASKS_TRUE_LABELS = None) -> TASKS_TEXTS:
        hrrasa_results = self._hrrasa.fit_predict(self._encode_data(data), self._encode_true_objects(true_objects))
        self.texts_ = hrrasa_results.reset_index()[['task', 'output']].set_index('task')
        return self.texts_

    def _encode_data(self, data):
        data = data[['task', 'performer', 'output']]
        data['embedding'] = data.output.apply(self.encoder)
        return data

    def _encode_true_objects(self, true_objects):
        return true_objects and true_objects.apply(self.endcoder)
