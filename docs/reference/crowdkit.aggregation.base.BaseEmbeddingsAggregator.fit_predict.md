# fit_predict
`crowdkit.aggregation.base.BaseEmbeddingsAggregator.fit_predict`

```python
fit_predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>

* **Returns:**

  Tasks' embeddings and outputs.
A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.

* **Return type:**

  DataFrame
