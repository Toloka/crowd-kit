# fit_predict
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit_predict`

```python
fit_predict(
    self,
    data: DataFrame,
    skills: Series = None
)
```

Fit the model and return the aggregated segmentations.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; segmentations. A pandas.DataFrame containing `performer`, `task` and `segmentation` columns&#x27;.</p>
`skills`|**Series**|<p>Performers&#x27; skills. A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>

* **Returns:**

  Tasks' segmentations.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's aggregated segmentation.

* **Return type:**

  Series
