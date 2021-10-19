# fit_predict
`crowdkit.aggregation.gold_majority_vote.GoldMajorityVote.fit_predict`

```
fit_predict(
    self,
    data: DataFrame,
    true_labels: Series
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`true_labels`|**Series**|<p>Tasks&#x27; ground truth labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s ground truth label.</p>

* **Returns:**

  Tasks' most likely true labels
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  DataFrame
