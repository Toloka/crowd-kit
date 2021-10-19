# fit_predict
`crowdkit.aggregation.hrrasa.TextHRRASA.fit_predict`

```
fit_predict(
    self,
    data: DataFrame,
    true_objects: Series = None
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs A pandas.DataFrame containing `task`, `performer` and `output` columns.</p>
`true_objects`|**Series**|<p>Tasks&#x27; ground truth labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Tasks' most likely true labels
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  DataFrame
