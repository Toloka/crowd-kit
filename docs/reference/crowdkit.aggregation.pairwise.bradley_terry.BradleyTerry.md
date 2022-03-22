# BradleyTerry
`crowdkit.aggregation.pairwise.bradley_terry.BradleyTerry` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/pairwise/bradley_terry.py#L16)

```python
BradleyTerry(
    self,
    n_iter: int,
    tol: float = 1e-05
)
```

Bradley-Terry, the classic algorithm for aggregating pairwise comparisons.


This algorithm constructs an items' ranking based on pairwise comparisons. Given
a pair of two items $i$ and $j$, the probability of $i$ to be ranked higher is,
according to the Bradley-Terry's probabilitstic model,
$$
P(i > j) = \frac{p_i}{p_i + p_j}.
$$
Here $\boldsymbol{p}$ is a vector of positive real-valued parameters that the algorithm optimizes. These
optimization process maximizes the log-likelihood of observed comparisons outcomes by the MM-algorithm:
$$
L(\boldsymbol{p}) = \sum_{i=1}^n\sum_{j=1}^n[w_{ij}\ln p_i - w_{ij}\ln (p_i + p_j)],
$$
where $w_{ij}$ denotes the number of comparisons of $i$ and $j$ "won" by $i$.

**Note:** the Bradley-Terry model needs the comparisons graph to be **strongly connected**.

David R. Hunter.
MM algorithms for generalized Bradley-Terry models
*Ann. Statist.*, Vol. 32, 1 (2004): 384–406.

Bradley, R. A. and Terry, M. E.
Rank analysis of incomplete block designs. I. The method of paired comparisons.
*Biometrika*, Vol. 39 (1952): 324–345.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>A number of optimization iterations.</p>
`scores_`|**Series**|<p>&#x27;Labels&#x27; scores. A pandas.Series index by labels and holding corresponding label&#x27;s scores</p>

**Examples:**

The Bradley-Terry model needs the data to be a `DataFrame` containing columns
`left`, `right`, and `label`. `left` and `right` contain identifiers of left and
right items respectfuly, `label` contains identifiers of items that won these
comparisons.

```python
import pandas as pd
from crowdkit.aggregation import BradleyTerry
df = pd.DataFrame(
    [
        ['item1', 'item2', 'item1'],
        ['item2', 'item3', 'item2']
    ],
    columns=['left', 'right', 'label']
)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.pairwise.bradley_terry.BradleyTerry.fit.md)| None
[fit_predict](crowdkit.aggregation.pairwise.bradley_terry.BradleyTerry.fit_predict.md)| None
