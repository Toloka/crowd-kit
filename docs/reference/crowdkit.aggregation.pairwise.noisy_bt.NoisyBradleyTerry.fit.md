# fit
`crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit`

```python
fit(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; pairwise comparison results. A pandas.DataFrame containing `performer`, `left`, `right`, and `label` columns&#x27;. For each row `label` must be equal to either `left` column or `right` column.</p>

* **Returns:**

  self.

* **Return type:**

  'NoisyBradleyTerry'
