# get_accuracy
`crowdkit.aggregation.utils.get_accuracy`

```python
get_accuracy(
    data: DataFrame,
    true_labels: Series,
    by: Optional[str] = None
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`true_labels`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Performers' skills.
A pandas.Series index by performers and holding corresponding performer's skill

* **Return type:**

  Series
