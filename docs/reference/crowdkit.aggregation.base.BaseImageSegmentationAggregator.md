# BaseImageSegmentationAggregator
`crowdkit.aggregation.base.BaseImageSegmentationAggregator`

```python
BaseImageSegmentationAggregator(self)
```

This is a base class for all image segmentation aggregators

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.base.BaseImageSegmentationAggregator.fit.md)| None
[fit_predict](crowdkit.aggregation.base.BaseImageSegmentationAggregator.fit_predict.md)| None
