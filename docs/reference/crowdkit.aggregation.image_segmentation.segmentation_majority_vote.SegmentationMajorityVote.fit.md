# fit
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit`

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
`data`|**DataFrame**|<p>Performers&#x27; segmentations. A pandas.DataFrame containing `performer`, `task` and `segmentation` columns&#x27;.</p>
`skills`|**Series**|<p>Performers&#x27; skills. A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>

* **Returns:**

  self.

* **Return type:**

  'SegmentationMajorityVote'
