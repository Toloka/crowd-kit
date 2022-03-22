# fit
`crowdkit.aggregation.classification.m_msr.MMSR.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/m_msr.py#L88)

```python
fit(self, data: DataFrame)
```

Estimate the workers' skills.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>

* **Returns:**

  self.

* **Return type:**

  [MMSR](crowdkit.aggregation.classification.m_msr.MMSR.md)
