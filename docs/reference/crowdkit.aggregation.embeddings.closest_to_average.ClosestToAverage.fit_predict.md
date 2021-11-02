# fit_predict
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict`

```
fit_predict(
    self,
    data: DataFrame,
    skills: Series = None
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>
`skills`|**Series**|<p>Performers&#x27; skills A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>

* **Returns:**

  Tasks' embeddings and outputs
A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.

* **Return type:**

  DataFrame
