# fit_predict
`crowdkit.aggregation.closest_to_average.ClosestToAverage.fit_predict`

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

  Tasks' most likely true labels
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  DataFrame
