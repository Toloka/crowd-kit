# fit
`crowdkit.aggregation.classification.glad.GLAD.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/glad.py#L279)

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

  [GLAD](crowdkit.aggregation.classification.glad.GLAD.md)
