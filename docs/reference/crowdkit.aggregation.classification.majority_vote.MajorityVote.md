# MajorityVote
`crowdkit.aggregation.classification.majority_vote.MajorityVote`

```
MajorityVote(
    self,
    on_missing_skill: str = 'error',
    default_skill: Optional[float] = None
)
```

Majority Vote - chooses the correct label for which more performers voted

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>Performers&#x27; skills A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`on_missing_skill`|**str**|<p>How to handle assignments done by workers with unknown skill Possible values:<ul><li>&quot;error&quot; — raise an exception if there is at least one assignment done by user with unknown skill;</li><li>&quot;ignore&quot; — drop assignments with unknown skill values during prediction. Raise an exception if there is no  assignments with known skill for any task;</li><li>value — default value will be used if skill is missing.</li></ul></p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict_proba.md)| None
