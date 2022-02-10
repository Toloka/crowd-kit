# fit_predict
`crowdkit.aggregation.base.BasePairwiseAggregator.fit_predict`

```python
fit_predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; pairwise comparison results. A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns&#x27;. For each row `label` must be equal to either `left` column or `right` column.</p>

* **Returns:**

  'Labels' scores.
A pandas.Series index by labels and holding corresponding label's scores

* **Return type:**

  Series
