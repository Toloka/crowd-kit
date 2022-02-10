# normalize_rows
`crowdkit.aggregation.utils.normalize_rows`

```python
normalize_rows(scores: DataFrame)
```

Scales values so that every raw sums to 1

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`scores`|**DataFrame**|<p>Tasks&#x27; label scores. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>

* **Returns:**

  Tasks' label probability distributions.
A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
is the probability of `task`'s true label to be equal to `label`. Each
probability is between 0 and 1, all task's probabilities should sum up to 1

* **Return type:**

  DataFrame
