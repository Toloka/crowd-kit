# fit_predict
`crowdkit.aggregation.texts.text_summarization.TextSummarization.fit_predict`

```python
fit_predict(self, data: DataFrame)
```

Run the aggregation and return the aggregated texts.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; text outputs. A pandas.DataFrame containing `task`, `performer` and `text` columns.</p>

* **Returns:**

  Tasks' texts.
A pandas.Series indexed by `task` such that `result.loc[task, text]`
is the task's text.

* **Return type:**

  Series
