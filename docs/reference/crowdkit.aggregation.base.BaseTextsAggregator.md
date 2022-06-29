# BaseTextsAggregator
`crowdkit.aggregation.base.BaseTextsAggregator` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base/__init__.py#L64)

```python
BaseTextsAggregator(self)
```

This is a base class for all texts aggregators

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`texts_`|**Series**|<p>Tasks&#x27; texts. A pandas.Series indexed by `task` such that `result.loc[task, text]` is the task&#x27;s text.</p>
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.base.BaseTextsAggregator.fit.md)| None
[fit_predict](crowdkit.aggregation.base.BaseTextsAggregator.fit_predict.md)| None
