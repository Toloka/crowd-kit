# ClosestToAverage
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage`

```
ClosestToAverage(self, distance: Callable[[ndarray, ndarray], float])
```

Closest to Average - chooses the output with the embedding closest to the average embedding

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`embeddings_and_outputs_`|**DataFrame**|<p>Tasks&#x27; embeddings and outputs A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.</p>
`scores_`|**DataFrame**|<p>Tasks&#x27; label scores A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit.md)| None
[fit_predict](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict.md)| None
[fit_predict_scores](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict_scores.md)| None
