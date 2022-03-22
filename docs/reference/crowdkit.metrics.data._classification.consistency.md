# consistency
`crowdkit.metrics.data._classification.consistency` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/metrics/data/_classification.py#L38)

```python
consistency(
    answers: DataFrame,
    workers_skills: Optional[Series] = None,
    aggregator: BaseClassificationAggregator = ...,
    by_task: bool = False
)
```

Consistency metric: posterior probability of aggregated label given workers skills


calculated using standard Dawid-Skene model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`answers`|**DataFrame**|<p>A data frame containing `task`, `worker` and `label` columns.</p>
`workers_skills`|**Optional\[Series\]**|<p>workers skills e.g. golden set skills. If not provided, uses aggregator&#x27;s `workers_skills` attribute.</p>
`aggregator`|**[BaseClassificationAggregator](crowdkit.aggregation.base.BaseClassificationAggregator.md)**|<p>aggregation method, default: MajorityVote</p>
`by_task`|**bool**|<p>if set, returns consistencies for every task in provided data frame.</p>

* **Returns:**

  Union[float, pd.Series]

* **Return type:**

  Union\[float, Series\]
