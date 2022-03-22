# fit_predict
`crowdkit.aggregation.texts.rover.ROVER.fit_predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/rover.py#L92)

```python
fit_predict(self, data: DataFrame)
```

Fit the model and return the aggregated texts.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; text outputs. A pandas.DataFrame containing `task`, `worker` and `text` columns.</p>

* **Returns:**

  Tasks' texts.
A pandas.Series indexed by `task` such that `result.loc[task, text]`
is the task's text.

* **Return type:**

  Series
