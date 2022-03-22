# fit_predict_score
`crowdkit.aggregation.classification.m_msr.MMSR.fit_predict_score` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/m_msr.py#L123)

```python
fit_predict_score(self, data: DataFrame)
```

Fit the model and return the total sum of weights for each label.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
