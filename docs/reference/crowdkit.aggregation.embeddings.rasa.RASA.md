# RASA
`crowdkit.aggregation.embeddings.rasa.RASA`

```python
RASA(
    self,
    n_iter: int = 100,
    tol: float = 1e-09,
    alpha: float = 0.05
)
```

Reliability Aware Sequence Aggregation.


RASA estimates *global* performers' reliabilities $\beta$ that are initialized by ones.

Next, the algorithm iteratively performs two steps:
1. For each task, estimate the aggregated embedding: $\hat{e}_i = \frac{\sum_k
\beta_k e_i^k}{\sum_k \beta_k}$
2. For each performer, estimate the global reliability: $\beta_k = \frac{\chi^2_{(\alpha/2,
|\mathcal{V}_k|)}}{\sum_i\left(\|e_i^k - \hat{e}_i\|^2\right)}$, where $\mathcal{V}_k$
is a set of tasks completed by the performer $k$

Finally, the aggregated result is the output which embedding is
the closest one to the $\hat{e}_i$.

Jiyi Li.
A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
*Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*,
pages 24–28 Hong Kong, China, November 3, 2019.
<http://doi.org/10.18653/v1/D19-5904>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>A number of iterations.</p>
`alpha`|**float**|<p>Confidence level of chi-squared distribution quantiles in beta parameter formula.</p>
`embeddings_and_outputs_`|**DataFrame**|<p>Tasks&#x27; embeddings and outputs. A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.</p>

**Examples:**

```python
import numpy as np
import pandas as pd
from crowdkit.aggregation import RASA
df = pd.DataFrame(
    [
        ['t1', 'p1', 'a', np.array([1.0, 0.0])],
        ['t1', 'p2', 'a', np.array([1.0, 0.0])],
        ['t1', 'p3', 'b', np.array([0.0, 1.0])]
    ],
    columns=['task', 'performer', 'output', 'embedding']
)
result = RASA().fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.embeddings.rasa.RASA.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.embeddings.rasa.RASA.fit_predict.md)| Fit the model and return aggregated outputs.
[fit_predict_scores](crowdkit.aggregation.embeddings.rasa.RASA.fit_predict_scores.md)| Fit the model and return scores.
