# fit
`crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit`

```python
fit(
    self,
    data: DataFrame,
    true_labels: Series
)
```

Estimate the performers' skills.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`true_labels`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  self.

* **Return type:**

  'GoldMajorityVote'
