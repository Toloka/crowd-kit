# fit
`crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/dawid_skene.py#L132)

```python
fit(self, data: DataFrame)
```

Fit the model through the EM-algorithm.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>

* **Returns:**

  self.

* **Return type:**

  [DawidSkene](crowdkit.aggregation.classification.dawid_skene.DawidSkene.md)
