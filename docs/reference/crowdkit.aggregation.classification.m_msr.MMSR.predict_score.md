# predict_score
`crowdkit.aggregation.classification.m_msr.MMSR.predict_score`

```python
predict_score(self, data: DataFrame)
```

Return total sum of weights for each label when the model is fitted.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
