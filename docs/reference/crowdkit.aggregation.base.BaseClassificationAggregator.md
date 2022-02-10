# BaseClassificationAggregator
`crowdkit.aggregation.base.BaseClassificationAggregator`

```python
BaseClassificationAggregator(self)
```

This is a base class for all classification aggregators

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.base.BaseClassificationAggregator.fit.md)| None
[fit_predict](crowdkit.aggregation.base.BaseClassificationAggregator.fit_predict.md)| None
