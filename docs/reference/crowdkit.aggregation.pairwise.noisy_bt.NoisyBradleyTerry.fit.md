# fit
`crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/pairwise/noisy_bt.py#L28)

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

  [NoisyBradleyTerry](crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.md)
