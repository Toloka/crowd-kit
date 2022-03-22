# get_accuracy
`crowdkit.aggregation.utils.get_accuracy` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/utils.py#L84)

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
`data`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>
`true_labels`|**Series**|<p>Tasks&#x27; ground truth labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  workers' skills.
A pandas.Series index by workers and holding corresponding worker's skill

* **Return type:**

  Series
