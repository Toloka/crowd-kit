# fit_predict
`crowdkit.aggregation.base.BasePairwiseAggregator.fit_predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base/__init__.py#L91)

```python
fit_predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; pairwise comparison results. A pandas.DataFrame containing `worker`, `left`, `right`, and `label` columns&#x27;. For each row `label` must be equal to either `left` column or `right` column.</p>

* **Returns:**

  'Labels' scores.
A pandas.Series index by labels and holding corresponding label's scores

* **Return type:**

  Series
