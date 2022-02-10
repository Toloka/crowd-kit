# fit_predict
`crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict`

```python
fit_predict(
    self,
    data: DataFrame,
    skills: Series = None
)
```

Fit the model and return aggregated results.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`skills`|**Series**|<p>Performers&#x27; skills. A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>

* **Returns:**

  Tasks' labels.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  Series
