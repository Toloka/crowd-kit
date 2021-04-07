from crowdkit.aggregation.base_aggregator import BaseAggregator
from pandas.core.frame import DataFrame
from typing import Any


class BaseEmbeddingAggregator(BaseAggregator):
    """Base class for aggregation algorithms that operate with embeddings of performers answers.

    Attributes:
        aggregated_embeddings_ (Optional[pd.Series]): result of embeddings aggregation for each task.
        golden_embeddings_: (Optional[pd.Series]): embeddings of golden outputs if the golden outputs are provided."""

    def __init__(self, encoder: Any, silent: bool): ...

    def _aggregate_embeddings(self, answers: DataFrame):
        """Calculates weighted average of embeddings for each task."""
        ...

    def _answers_base_checks(self, answers: DataFrame): ...

    def _choose_nearest_output(self, answers, metric='cosine'):
        """Choses nearest performers answer according to aggregated embeddings."""
        ...

    def _distance_from_aggregated(self, answers: DataFrame):
        """Calculates the square of Euclidian distance from aggregated embedding for each answer."""
        ...

    def _get_embeddings(self, answers: DataFrame):
        """Obtaines embeddings for performers answers."""
        ...

    def _get_golden_embeddings(self, answers: DataFrame):
        """Processes embeddings for golden outputs."""
        ...

    def _init_performers_reliabilities(self, answers: DataFrame):
        """Initialize performers reliabilities by ones."""
        ...
