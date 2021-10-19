# fit_predict
`crowdkit.aggregation.rover.ROVER.fit_predict`

```
fit_predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; text outputs A pandas.DataFrame containing `task`, `performer` and `text` columns.</p>

* **Returns:**

  Tasks' label scores
A pandas.DataFrame indexed by `task` such that `result.loc[task, text]`
is the task's text.

* **Return type:**

  DataFrame
