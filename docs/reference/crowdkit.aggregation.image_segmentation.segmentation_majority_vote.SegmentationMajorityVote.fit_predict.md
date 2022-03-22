# fit_predict
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit_predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/image_segmentation/segmentation_majority_vote.py#L72)

```python
fit_predict(
    self,
    data: DataFrame,
    skills: Optional[Series] = None
)
```

Fit the model and return the aggregated segmentations.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; segmentations. A pandas.DataFrame containing `worker`, `task` and `segmentation` columns&#x27;.</p>
`skills`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>

* **Returns:**

  Tasks' segmentations.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's aggregated segmentation.

* **Return type:**

  Series
