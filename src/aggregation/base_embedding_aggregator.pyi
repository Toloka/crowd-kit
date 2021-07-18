from crowdkit.aggregation.base_aggregator import BaseAggregator
from typing import Any

class BaseEmbeddingAggregator(BaseAggregator):
    """Base class for aggregation algorithms that operate with embeddings of performers answers.

    Attributes:
        aggregated_embeddings_ (Optional[pd.Series]): result of embeddings aggregation for each task.
        golden_embeddings_: (Optional[pd.Series]): embeddings of golden outputs if the golden outputs are provided.
    """

    def __init__(self, encoder: Any, silent: bool): ...
