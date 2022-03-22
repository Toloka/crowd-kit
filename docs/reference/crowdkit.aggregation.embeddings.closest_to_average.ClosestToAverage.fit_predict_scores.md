# fit_predict_scores
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict_scores` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/embeddings/closest_to_average.py#L71)

```python
fit_predict_scores(
    self,
    data: DataFrame,
    aggregated_embeddings: Optional[Series] = None
)
```

Fit the model and return the estimated scores.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.</p>
`aggregated_embeddings`|**Optional\[Series\]**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  Tasks' label probability distributions.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the probability of `task`'s true label to be equal to `label`. Each
probability is between 0 and 1, all task's probabilities should sum up to 1

* **Return type:**

  DataFrame
