# fit_predict
`crowdkit.aggregation.base.BaseImageSegmentationAggregator.fit_predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base/__init__.py#L43)

```python
fit_predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; segmentations. A pandas.DataFrame containing `worker`, `task` and `segmentation` columns&#x27;.</p>

* **Returns:**

  Tasks' segmentations.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's aggregated segmentation.

* **Return type:**

  Series
