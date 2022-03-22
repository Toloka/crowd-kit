# Wawa
`crowdkit.aggregation.classification.wawa.Wawa` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/wawa.py#L12)

```python
Wawa(self)
```

Worker Agreement with Aggregate.


This algorithm does three steps:
1. Calculate the majority vote label
2. Estimate workers' skills as a fraction of responses that are equal to the majority vote
3. Calculate the weigthed majority vote based on skills from the previous step

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>

**Examples:**

```python
from crowdkit.aggregation import Wawa
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
result = Wawa().fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.wawa.Wawa.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.classification.wawa.Wawa.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.wawa.Wawa.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
[predict](crowdkit.aggregation.classification.wawa.Wawa.predict.md)| Infer the true labels when the model is fitted.
[predict_proba](crowdkit.aggregation.classification.wawa.Wawa.predict_proba.md)| Return probability distributions on labels for each task when the model is fitted.
