# get_most_probable_labels
`crowdkit.aggregation.utils.get_most_probable_labels` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/utils.py#L51)

```python
get_most_probable_labels(proba: DataFrame)
```

Returns most probable labels

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`proba`|**DataFrame**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
