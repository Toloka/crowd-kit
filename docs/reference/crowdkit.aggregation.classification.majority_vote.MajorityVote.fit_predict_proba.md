# fit_predict_proba
`crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict_proba` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/majority_vote.py#L82)

```python
fit_predict_proba(
    self,
    data: DataFrame,
    skills: Optional[Series] = None
)
```

Fit the model and return probability distributions on labels for each task.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>
`skills`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>

* **Returns:**

  Tasks' label probability distributions.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the probability of `task`'s true label to be equal to `label`. Each
probability is between 0 and 1, all task's probabilities should sum up to 1

* **Return type:**

  DataFrame
