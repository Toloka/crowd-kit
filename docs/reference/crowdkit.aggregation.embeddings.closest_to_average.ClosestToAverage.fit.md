# fit
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/embeddings/closest_to_average.py#L33)

```python
fit(
    self,
    data: DataFrame,
    aggregated_embeddings: Optional[Series] = None,
    true_embeddings: Optional[Series] = None
)
```

Fits the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.</p>
`aggregated_embeddings`|**Optional\[Series\]**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>
`true_embeddings`|**Optional\[Series\]**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  self.

* **Return type:**

  [ClosestToAverage](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.md)
