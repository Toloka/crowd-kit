# fit_predict
`crowdkit.aggregation.image_segmentation.segmentation_rasa.SegmentationRASA.fit_predict`

```python
fit_predict(self, data: DataFrame)
```

Fit the model and return the aggregated segmentations.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; segmentations. A pandas.DataFrame containing `performer`, `task` and `segmentation` columns&#x27;.</p>

* **Returns:**

  Tasks' segmentations.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's aggregated segmentation.

* **Return type:**

  Series
