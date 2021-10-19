# fit_predict
`crowdkit.aggregation.rasa.RASA.fit_predict`

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

  Tasks' most likely true labels
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  DataFrame
