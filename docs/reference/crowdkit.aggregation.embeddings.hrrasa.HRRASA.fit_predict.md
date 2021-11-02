# fit_predict
`crowdkit.aggregation.embeddings.hrrasa.HRRASA.fit_predict`

```
fit_predict(
    self,
    data: DataFrame,
    true_embeddings: Series = None
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>
`true_embeddings`|**Series**|<p>Tasks&#x27; embeddings A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  Tasks' embeddings and outputs
A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.

* **Return type:**

  DataFrame
