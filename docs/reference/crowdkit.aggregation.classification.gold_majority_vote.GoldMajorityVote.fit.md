# fit
`crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/gold_majority_vote.py#L63)

```python
fit(
    self,
    data: DataFrame,
    true_labels: Series
)
```

Estimate the workers' skills.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>
`true_labels`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  self.

* **Return type:**

  [GoldMajorityVote](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.md)
