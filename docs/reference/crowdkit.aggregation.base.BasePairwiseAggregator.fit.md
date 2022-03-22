# fit
`crowdkit.aggregation.base.BasePairwiseAggregator.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base.py#L87)

```python
fit(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; pairwise comparison results. A pandas.DataFrame containing `worker`, `left`, `right`, and `label` columns&#x27;. For each row `label` must be equal to either `left` column or `right` column.</p>

* **Returns:**

  self.

* **Return type:**

  [BasePairwiseAggregator](crowdkit.aggregation.base.BasePairwiseAggregator.md)
