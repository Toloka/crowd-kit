# DawidSkene
`crowdkit.aggregation.classification.dawid_skene.DawidSkene`

```
DawidSkene(self, n_iter: int)
```

Dawid-Skene aggregation model


A. Philip Dawid and Allan M. Skene. 1979.
Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, 1 (1979), 20–28.

https://doi.org/10.2307/2346806

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`priors_`|**Optional\[Series\]**|<p>A prior label distribution A pandas.Series indexed by labels and holding corresponding label&#x27;s probability of occurrence. Each probability is between 0 and 1, all probabilities should sum up to 1</p>
`errors_`|**Optional\[DataFrame\]**|<p>Performers&#x27; error matrices A pandas.DataFrame indexed by `performer` and `label` with a column for every label_id found in `data` such that `result.loc[performer, observed_label, true_label]` is the probability of `performer` producing an `observed_label` given that a task&#x27;s true label is `true_label`</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit_predict_proba.md)| None
