# predict
`crowdkit.aggregation.gold_majority_vote.GoldMajorityVote.predict`

```
predict(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>

* **Returns:**

  Tasks' most likely true labels
A pandas.Series indexed by `task` such that `labels.loc[task]`
is the tasks's most likely true label.

* **Return type:**

  DataFrame
