# fit
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit`

```python
fit(
    self,
    data: DataFrame,
    aggregated_embeddings: Series = None,
    true_embeddings: Series = None
)
```

Fits the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>
`aggregated_embeddings`|**Series**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>
`true_embeddings`|**Series**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  self.

* **Return type:**

  'ClosestToAverage'
