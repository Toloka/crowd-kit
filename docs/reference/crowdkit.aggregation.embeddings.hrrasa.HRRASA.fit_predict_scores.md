# fit_predict_scores
`crowdkit.aggregation.embeddings.hrrasa.HRRASA.fit_predict_scores`

```python
fit_predict_scores(
    self,
    data: DataFrame,
    true_embeddings: Series = None
)
```

Fit the model and return scores.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>
`true_embeddings`|**Series**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
