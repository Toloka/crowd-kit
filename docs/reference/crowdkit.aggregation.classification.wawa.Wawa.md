# Wawa
`crowdkit.aggregation.classification.wawa.Wawa`

```
Wawa(self)
```

Worker Agreement with Aggregate


Calculates the considers the likelihood of coincidence of the performers opinion with the majority
Based on this, for each task, calculates the sum of the agreement of each label
The correct label is the one where the sum of the agreements is greater

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>Performers&#x27; skills A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.wawa.Wawa.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.wawa.Wawa.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.wawa.Wawa.fit_predict_proba.md)| None
[predict](crowdkit.aggregation.classification.wawa.Wawa.predict.md)| None
[predict_proba](crowdkit.aggregation.classification.wawa.Wawa.predict_proba.md)| None
