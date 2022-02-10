# fit
`crowdkit.aggregation.classification.majority_vote.MajorityVote.fit`

```python
fit(
    self,
    data: DataFrame,
    skills: Series = None
)
```

Fit the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`skills`|**Series**|<p>Performers&#x27; skills. A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>

* **Returns:**

  self.

* **Return type:**

  'MajorityVote'
