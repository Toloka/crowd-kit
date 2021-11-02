# GoldMajorityVote
`crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote`

```
GoldMajorityVote(self)
```

Majority Vote when exist golden dataset (ground truth) for some tasks


Calculates the probability of a correct label for each performer based on the golden set
Based on this, for each task, calculates the sum of the probabilities of each label
The correct label is the one where the sum of the probabilities is greater

For Example: You have 10k tasks completed by 3k different performers. And you have 100 tasks where you already
know ground truth labels. First you can call 'fit' to calc percents of correct labels for each performers.
And then call 'predict' to calculate labels for you 10k tasks.

It's necessary that:
1. All performers must done at least one task from golden dataset.
2. All performers in dataset that send to 'predict', existed in answers dataset that was sent to 'fit'

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>Performers&#x27; skills A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit_predict_proba.md)| None
[predict](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.predict.md)| None
[predict_proba](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.predict_proba.md)| None
