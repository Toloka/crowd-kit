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
    """
    HRRASA on text embeddings.

    Given a sentence encoder, encodes texts provided by workers and runs the HRRASA algorithm for embedding
    aggregation.

    Args:
        encoder: A callable that takes a text and returns a NumPy array containing the corresponding embedding.
        n_iter: A number of HRRASA iterations.
        lambda_emb: A weight of reliability calculated on embeddigs.
        lambda_out: A weight of reliability calculated on outputs.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.
        calculate_ranks: If true, calculate additional attribute `ranks_`.

    Examples:
        We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
        >>> from crowdkit.datasets import load_dataset
        >>> from crowdkit.aggregation import TextHRRASA
        >>> from sentence_transformers import SentenceTransformer
        >>> encoder = SentenceTransformer('all-mpnet-base-v2')
        >>> hrrasa = TextHRRASA(encoder=encoder.encode)
        >>> df, gt = load_dataset('crowdspeech-test-clean')
        >>> df['text'] = df['text'].apply(lambda s: s.lower())
        >>> result = hrrasa.fit_predict(df)
    """

    # texts_

    @property
    def loss_history_(self):
        return self._hrrasa.loss_history_

    def __init__(
        self,
        encoder: Callable, n_iter: int = 100, tol: float = 1e-5, lambda_emb: float = 0.5, lambda_out: float = 0.5,
        alpha: float = 0.05, calculate_ranks: bool = False, output_similarity: Callable = glue_similarity
    ):
        super().__init__()
        self.encoder = encoder
        self._hrrasa = HRRASA(n_iter, tol, lambda_emb, lambda_out, alpha, calculate_ranks, output_similarity)

    def __getattr__(self, name):
        return getattr(self._hrrasa, name)

    @manage_docstring
    def fit_predict_scores(self, data: DATA, true_objects: TASKS_TRUE_LABELS = None) -> TASKS_LABEL_SCORES:
        """
        Fit the model and return scores.
        """

        return self._hrrasa.fit_predict_scores(self._encode_data(data), self._encode_true_objects(true_objects))

    @manage_docstring
    def fit_predict(self, data: DATA, true_objects: TASKS_TRUE_LABELS = None) -> TASKS_TEXTS:
        """
        Fit the model and return aggregated texts.
        """

        hrrasa_results = self._hrrasa.fit_predict(self._encode_data(data), self._encode_true_objects(true_objects))
        self.texts_ = hrrasa_results.reset_index()[['task', 'output']].set_index('task')
        return self.texts_

    def _encode_data(self, data):
        data = data[['task', 'worker', 'output']]
        data['embedding'] = data.output.apply(self.encoder)
        return data

    def _encode_true_objects(self, true_objects):
        return true_objects and true_objects.apply(self.endcoder)
