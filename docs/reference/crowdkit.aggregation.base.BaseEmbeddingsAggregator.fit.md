# fit
`crowdkit.aggregation.base.BaseEmbeddingsAggregator.fit` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/base/__init__.py#L55)

```python
fit(self, data: DataFrame)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Workers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `worker`, `output` and `embedding` columns.</p>

* **Returns:**

  self.

* **Return type:**

  [BaseEmbeddingsAggregator](crowdkit.aggregation.base.BaseEmbeddingsAggregator.md)
