# BaseEmbeddingAggregator
`crowdkit.aggregation.base_embedding_aggregator.BaseEmbeddingAggregator`

```
BaseEmbeddingAggregator(
    self,
    encoder: Any,
    silent: bool
)
```

Base class for aggregation algorithms that operate with embeddings of performers answers.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`aggregated_embeddings_`|**-**|<p>result of embeddings aggregation for each task.</p>
`golden_embeddings_`|**-**|<p>(Optional[pd.Series]): embeddings of golden outputs if the golden outputs are provided.</p>
