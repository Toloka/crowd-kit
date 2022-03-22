__all__ = ['TextRASA']
from typing import Callable

from .. import annotations
from ..annotations import manage_docstring, Annotation
from ..base import BaseTextsAggregator
from ..embeddings.rasa import RASA


class TextRASA(BaseTextsAggregator):
    """
    RASA on text embeddings.

    Given a sentence encoder, encodes texts provided by workers and runs the RASA algorithm for embedding
    aggregation.

    Args:
        encoder: A callable that takes a text and returns a NumPy array containing the corresponding embedding.
        n_iter: A number of RASA iterations.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.

    Examples:
        We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
        >>> from crowdkit.datasets import load_dataset
        >>> from crowdkit.aggregation import TextRASA
        >>> from sentence_transformers import SentenceTransformer
        >>> encoder = SentenceTransformer('all-mpnet-base-v2')
        >>> hrrasa = TextRASA(encoder=encoder.encode)
        >>> df, gt = load_dataset('crowdspeech-test-clean')
        >>> df['text'] = df['text'].apply(lambda s: s.lower())
        >>> result = hrrasa.fit_predict(df)
    """

    # texts_

    @property
    def loss_history_(self):
        return self._hrrasa.loss_history_

    def __init__(self, encoder: Callable, n_iter: int = 100, tol: float = 1e-5, alpha: float = 0.05):
        super().__init__()
        self.encoder = encoder
        self._rasa = RASA(n_iter, tol, alpha)

    def __getattr__(self, name):
        return getattr(self._rasa, name)

    @manage_docstring
    def fit(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> Annotation(type='TextRASA', title='self'):
        """
        Fit the model.
        """

        self._rasa.fit(self._encode_data(data), self._encode_true_objects(true_objects))
        return self

    @manage_docstring
    def fit_predict_scores(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> annotations.TASKS_LABEL_SCORES:
        """
        Fit the model and return scores.
        """

        return self._rasa.fit_predict_scores(self._encode_data(data), self._encode_true_objects(true_objects))

    @manage_docstring
    def fit_predict(self, data: annotations.DATA, true_objects: annotations.TASKS_TRUE_LABELS = None) -> annotations.TASKS_TEXTS:
        """
        Fit the model and return aggregated texts.
        """

        rasa_results = self._rasa.fit_predict(self._encode_data(data), self._encode_true_objects(true_objects))
        self.texts_ = rasa_results.reset_index()[['task', 'output']].set_index('task')
        return self.texts_

    def _encode_data(self, data):
        data = data[['task', 'worker', 'output']]
        data['embedding'] = data.output.apply(self.encoder)
        return data

    def _encode_true_objects(self, true_objects):
        return true_objects and true_objects.apply(self.endcoder)
