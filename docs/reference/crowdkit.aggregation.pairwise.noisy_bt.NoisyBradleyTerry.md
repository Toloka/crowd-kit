# NoisyBradleyTerry
`crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/pairwise/noisy_bt.py#L14)

```python
NoisyBradleyTerry(
    self,
    n_iter: int = 100,
    tol: float = 1e-05,
    regularization_ratio: float = 1e-05,
    random_state: int = 0
)
```

Bradley-Terry model for pairwise comparisons with additional parameters.


This model is a modification of the [Bradley-Terry model](crowdkit.aggregation.pairwise.bradley_terry.BradleyTerry.md)
with parameters for workers' skills (reliability) and biases.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`scores_`|**Series**|<p>&#x27;Labels&#x27; scores. A pandas.Series index by labels and holding corresponding label&#x27;s scores</p>
`skills_`|**Series**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`biases_`|**Series**|<p>Predicted biases for each worker. Indicates the probability of a worker to choose the left item.. A series of workers&#x27; biases indexed by workers</p>

**Examples:**

The following example shows how to aggregate results of comparisons **grouped by some column**.
In the example the two questions `q1` and `q2` are used to group the labeled data.
Temporary data structure is created and the model is applied to it.
The results are splitted in two arrays, and each array contains scores for one of the initial groups.

```python
import pandas as pd
from crowdkit.aggregation import NoisyBradleyTerry
data = pd.DataFrame(
    [
        ['q1', 'w1', 'a', 'b', 'a'],
        ['q1', 'w2', 'a', 'b', 'b'],
        ['q1', 'w3', 'a', 'b', 'a'],
        ['q2', 'w1', 'a', 'b', 'b'],
        ['q2', 'w2', 'a', 'b', 'a'],
        ['q2', 'w3', 'a', 'b', 'b'],
    ],
    columns=['question', 'worker', 'left', 'right', 'label']
)
for col in 'left', 'right', 'label':
    data[col] = list(zip(data['question'], data[col]))
result = NoisyBradleyTerry(n_iter=10).fit_predict(data)
result.index = pd.MultiIndex.from_tuples(result.index, names=['question', 'label'])
print(result['q1'])      # Scores for all items in the q1 question
print(result['q2']['b']) # Score for the item b in the q2 question
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit.md)| None
[fit_predict](crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit_predict.md)| None
