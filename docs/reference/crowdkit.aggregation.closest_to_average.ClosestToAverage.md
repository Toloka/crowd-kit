# ClosestToAverage
`crowdkit.aggregation.closest_to_average.ClosestToAverage`

```
ClosestToAverage(self, distance: Callable[[ndarray, ndarray], float])
```

Majority Vote - chooses the correct label for which more performers voted

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`outputs_`|**DataFrame**|<p>Tasks&#x27; most likely true labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`scores_`|**DataFrame**|<p>Tasks&#x27; label scores A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.closest_to_average.ClosestToAverage.fit.md)| None
[fit_predict](crowdkit.aggregation.closest_to_average.ClosestToAverage.fit_predict.md)| None
[fit_predict_scores](crowdkit.aggregation.closest_to_average.ClosestToAverage.fit_predict_scores.md)| None
