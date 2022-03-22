# fit
`crowdkit.aggregation.texts.text_rasa.TextRASA.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/texts/text_rasa.py#L47)

```python
fit(
    self,
    data: DataFrame,
    true_objects: Optional[Series] = None
)
```

Fit the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs. A pandas.DataFrame containing `task`, `worker` and `output` columns.</p>
`true_objects`|**Optional\[Series\]**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  self.

* **Return type:**

  [TextRASA](crowdkit.aggregation.texts.text_rasa.TextRASA.md)
