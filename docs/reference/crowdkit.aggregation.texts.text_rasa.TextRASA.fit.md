# fit
`crowdkit.aggregation.texts.text_rasa.TextRASA.fit`

```python
fit(
    self,
    data: DataFrame,
    true_objects: Series = None
)
```

Fit the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs. A pandas.DataFrame containing `task`, `performer` and `output` columns.</p>
`true_objects`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  self.

* **Return type:**

  'TextRASA'
