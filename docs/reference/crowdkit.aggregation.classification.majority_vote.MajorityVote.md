# MajorityVote
`crowdkit.aggregation.classification.majority_vote.MajorityVote` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/majority_vote.py#L12)

```python
MajorityVote(
    self,
    on_missing_skill: str = 'error',
    default_skill: Optional[float] = None
)
```

Majority Vote aggregation algorithm.


Majority vote is a straightforward approach for categorical aggregation: for each task,
it outputs a label which has the largest number of responses. Additionaly, the majority vote
can be used when different weights assigned for workers' votes. In this case, the
resulting label will be the one with the largest sum of weights.


**Note:** in case when two or more labels have the largest number of votes, the resulting
label will be the same for all tasks which have the same set of labels with equal count of votes.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`default_skill`|**Optional\[float\]**|<p>Defualt worker&#x27;s weight value.</p>
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`on_missing_skill`|**str**|<p>How to handle assignments done by workers with unknown skill. Possible values:<ul><li>&quot;error&quot; — raise an exception if there is at least one assignment done by user with unknown skill;</li><li>&quot;ignore&quot; — drop assignments with unknown skill values during prediction. Raise an exception if there is no  assignments with known skill for any task;</li><li>value — default value will be used if skill is missing.</li></ul></p>

**Examples:**

Basic majority voting:
```python
from crowdkit.aggregation import MajorityVote
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
result = MajorityVote().fit_predict(df)
```

Weighted majority vote:
```python
import pandas as pd
from crowdkit.aggregation import MajorityVote
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
skills = pd.Series({'p1': 0.5, 'p2': 0.7, 'p3': 0.4})
result = MajorityVote.fit_predict(df, skills)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.majority_vote.MajorityVote.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
