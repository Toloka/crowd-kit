# fit
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/image_segmentation/segmentation_majority_vote.py#L53)

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
`data`|**DataFrame**|<p>Workers&#x27; segmentations. A pandas.DataFrame containing `worker`, `task` and `segmentation` columns&#x27;.</p>
`skills`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>

* **Returns:**

  self.

* **Return type:**

  [SegmentationMajorityVote](crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.md)
