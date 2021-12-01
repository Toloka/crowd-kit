# predict_proba
`crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.predict_proba`

```
predict_proba(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>

* **Returns:**

  Tasks' label probability distributions
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the probability of `task`'s true label to be equal to `label`. Each
probability is between 0 and 1, all task's probabilities should sum up to 1

* **Return type:**

  DataFrame