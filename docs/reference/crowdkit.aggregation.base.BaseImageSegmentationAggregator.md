# BaseImageSegmentationAggregator
`crowdkit.aggregation.base.BaseImageSegmentationAggregator` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base/__init__.py#L31)

```python
BaseImageSegmentationAggregator(self)
```

This is a base class for all image segmentation aggregators

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.base.BaseImageSegmentationAggregator.fit.md)| None
[fit_predict](crowdkit.aggregation.base.BaseImageSegmentationAggregator.fit_predict.md)| None
