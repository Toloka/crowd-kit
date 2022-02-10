# fit_predict_scores
`crowdkit.aggregation.texts.text_hrrasa.TextHRRASA.fit_predict_scores`

```python
fit_predict_scores(
    self,
    data: DataFrame,
    true_objects: Series = None
)
```

Fit the model and return scores.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs. A pandas.DataFrame containing `task`, `performer` and `output` columns.</p>
`true_objects`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Tasks' label scores.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the score of `label` for `task`.

* **Return type:**

  DataFrame
