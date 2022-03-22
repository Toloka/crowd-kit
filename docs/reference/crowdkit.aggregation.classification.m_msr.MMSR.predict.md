# predict
`crowdkit.aggregation.classification.m_msr.MMSR.predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/m_msr.py#L99)

```python
predict(self, data: DataFrame)
```

Infer the true labels when the model is fitted.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>

* **Returns:**

  Tasks' labels.
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  Series
