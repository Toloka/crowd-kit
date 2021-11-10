__all__ = ['TextRASA']
from typing import Callable

from .. import annotations
from ..annotations import manage_docstring, Annotation
from ..base import BaseTextsAggregator
from ..embeddings.rasa import RASA


class TextRASA(BaseTextsAggregator):

    # texts_

    def __init__(self, encoder: Callable, n_iter: int = 100, alpha: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self._rasa = RASA(n_iter, alpha)

    def __getattr__(self, name):
        return getattr(self._rasa, name)

    @manage_docstring
    def fit(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> Annotation(type='TextRASA', title='self'):
        self._rasa.fit(self._encode_data(data), self._encode_true_objects(true_objects))
        return self

    @manage_docstring
    def fit_predict_scores(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> annotations.TASKS_LABEL_SCORES:
        return self._rasa.fit_predict_scores(self._encode_data(data), self._encode_true_objects(true_objects))

    @manage_docstring
    def fit_predict(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> annotations.TASKS_TEXTS:
        rasa_results = self._rasa.fit_predict(self._encode_data(data), self._encode_true_objects(true_objects))
        self.texts_ = rasa_results.reset_index()[['task', 'output']].set_index('task')
        return self.texts_

    def _encode_data(self, data):
        data = data[['task', 'performer', 'output']]
        data['embedding'] = data.output.apply(self.encoder)
        return data

    def _encode_true_objects(self, true_objects):
        return true_objects and true_objects.apply(self.endcoder)
