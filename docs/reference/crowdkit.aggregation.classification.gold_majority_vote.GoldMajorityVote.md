# GoldMajorityVote
`crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/gold_majority_vote.py#L12)

```python
GoldMajorityVote(self)
```

Majority Vote when exist golden dataset (ground truth) for some tasks.


Calculates the probability of a correct label for each worker based on the golden set.
Based on this, for each task, calculates the sum of the probabilities of each label.
The correct label is the one where the sum of the probabilities is greater.

For Example: You have 10k tasks completed by 3k different workers. And you have 100 tasks where you already
know ground truth labels. First you can call `fit` to calc percents of correct labels for each workers.
And then call `predict` to calculate labels for you 10k tasks.

It's necessary that:
1. All workers must done at least one task from golden dataset.
2. All workers in dataset that send to `predict`, existed in answers dataset that was sent to `fit`.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>

**Examples:**

```python
import pandas as pd
from crowdkit.aggregation import GoldMajorityVote
df = pd.DataFrame(
    [
        ['t1', 'p1', 0],
        ['t1', 'p2', 0],
        ['t1', 'p3', 1],
        ['t2', 'p1', 1],
        ['t2', 'p2', 0],
        ['t2', 'p3', 1],
    ],
    columns=['task', 'worker', 'label']
)
true_labels = pd.Series({'t1': 0})
gold_mv = GoldMajorityVote()
result = gold_mv.fit_predict(df, true_labels)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit.md)| Estimate the workers' skills.
[fit_predict](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
[predict](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.predict.md)| Infer the true labels when the model is fitted.
[predict_proba](crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.predict_proba.md)| Return probability distributions on labels for each task when the model is fitted.
