# fit_predict_scores
`crowdkit.aggregation.texts.text_rasa.TextRASA.fit_predict_scores` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/text_rasa.py#L56)

```python
fit_predict_scores(
    self,
    data: DataFrame,
    true_objects: Optional[Series] = None
)
```

Fit the model and return scores.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs. A pandas.DataFrame containing `task`, `worker` and `output` columns.</p>
`true_objects`|**Optional\[Series\]**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
