# accuracy_on_aggregates
`crowdkit.metrics.workers.accuracy_on_aggregates.accuracy_on_aggregates` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/metrics/workers/accuracy_on_aggregates.py#L12)

```python
accuracy_on_aggregates(
    answers: DataFrame,
    aggregator: Optional[BaseClassificationAggregator] = ...,
    aggregates: Optional[Series] = None,
    by: Optional[str] = None
)
```

Accuracy on aggregates: a fraction of worker's answers that match the aggregated one.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`answers`|**DataFrame**|<p>a data frame containing `task`, `worker` and `label` columns.</p>
`aggregator`|**Optional\[[BaseClassificationAggregator](crowdkit.aggregation.base.BaseClassificationAggregator.md)\]**|<p>aggregation algorithm. default: MajorityVote</p>
`aggregates`|**Optional\[Series\]**|<p>aggregated answers for provided tasks.</p>
`by`|**Optional\[str\]**|<p>if set, returns accuracies for every worker in provided data frame. Otherwise, returns an average accuracy of all workers.</p>
`Returns`|**-**|<p>Union[float, pd.Series]</p>
