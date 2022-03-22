# fit_predict_scores
`crowdkit.aggregation.embeddings.hrrasa.HRRASA.fit_predict_scores` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/embeddings/hrrasa.py#L162)

```python
fit_predict_scores(
    self,
    data: DataFrame,
    true_embeddings: Optional[Series] = None
)
```

Fit the model and return scores.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.</p>
`true_embeddings`|**Optional\[Series\]**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
