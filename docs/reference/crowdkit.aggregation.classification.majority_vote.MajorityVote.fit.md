# fit
`crowdkit.aggregation.classification.majority_vote.MajorityVote.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/majority_vote.py#L62)

```python
fit(
    self,
    data: DataFrame,
    skills: Optional[Series] = None
)
```

Fit the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>
`skills`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>

* **Returns:**

  self.

* **Return type:**

  [MajorityVote](crowdkit.aggregation.classification.majority_vote.MajorityVote.md)
