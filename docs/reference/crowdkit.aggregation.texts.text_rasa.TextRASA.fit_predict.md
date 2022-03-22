# fit_predict
`crowdkit.aggregation.texts.text_rasa.TextRASA.fit_predict` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/text_rasa.py#L64)

```python
fit_predict(
    self,
    data: DataFrame,
    true_objects: Optional[Series] = None
)
```

Fit the model and return aggregated texts.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs. A pandas.DataFrame containing `task`, `worker` and `output` columns.</p>
`true_objects`|**Optional\[Series\]**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Tasks' texts.
A pandas.Series indexed by `task` such that `result.loc[task, text]`
is the task's text.

* **Return type:**

  Series
