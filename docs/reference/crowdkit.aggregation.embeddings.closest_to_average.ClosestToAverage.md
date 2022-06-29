# ClosestToAverage
`crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/embeddings/closest_to_average.py#L12)

```python
ClosestToAverage(self, distance: Callable[[ndarray, ndarray], float])
```

Closest to Average - chooses the output with the embedding closest to the average embedding.


This method takes a `DataFrame` containing four columns: `task`, `worker`, `output`, and `embedding`.
Here the `embedding` is a vector containing a representation of the `output`. The `output` might be any
type of data such as text, images, NumPy arrays, etc. As the result, the method returns the output which
embedding is the closest one to the average embedding of the task's responses.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`distance`|**Callable\[\[ndarray, ndarray\], float\]**|<p>A callable that takes two NumPy arrays and returns a single `float` number — the distance between these two vectors.</p>
`embeddings_and_outputs_`|**DataFrame**|<p>Tasks&#x27; embeddings and outputs. A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.</p>
`scores_`|**DataFrame**|<p>Tasks&#x27; label scores. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit.md)| Fits the model.
[fit_predict](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict.md)| Fit the model and return the aggregated results.
[fit_predict_scores](crowdkit.aggregation.embeddings.closest_to_average.ClosestToAverage.fit_predict_scores.md)| Fit the model and return the estimated scores.
