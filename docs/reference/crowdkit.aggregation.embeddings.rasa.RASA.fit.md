# fit
`crowdkit.aggregation.embeddings.rasa.RASA.fit`

```python
fit(
    self,
    data: DataFrame,
    true_embeddings: Series = None
)
```

Fit the model.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; outputs with their embeddings. A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.</p>
`true_embeddings`|**Series**|<p>Tasks&#x27; embeddings. A pandas.Series indexed by `task` and holding corresponding embeddings.</p>

* **Returns:**

  self.

* **Return type:**

  'RASA'
