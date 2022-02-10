# DawidSkene
`crowdkit.aggregation.classification.dawid_skene.DawidSkene`

```python
DawidSkene(
    self,
    n_iter: int = 100,
    tol: float = 1e-05
)
```

Dawid-Skene aggregation model.


Probabilistic model that parametrizes performers' level of expertise through confusion matrices.

Let $e^w$ be a performer's confusion (error) matrix of size $K \times K$ in case of $K$ class classification,
$p$ be a vector of prior classes probabilities, $z_j$ be a true task's label, and $y^w_j$ be a performer's
answer for the task $j$. The relationships between these parameters are represented by the following latent
label model.

![Dawid-Skene latent label model](http://tlk.s3.yandex.net/crowd-kit/docs/ds_llm.png)

Here the prior true label probability is
$$
\operatorname{Pr}(z_j = c) = p[c],
$$
and the distribution on the performer's responses given the true label $c$ is represented by the
corresponding column of the error matrix:
$$
\operatorname{Pr}(y_j^w = k | z_j = c) = e^w[k, c].
$$

Parameters $p$ and $e^w$ and latent variables $z$ are optimized through the Expectation-Maximization algorithm.

A. Philip Dawid and Allan M. Skene. Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
*Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28*, 1 (1979), 20–28.

https://doi.org/10.2307/2346806

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>The number of EM iterations.</p>
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`priors_`|**Optional\[Series\]**|<p>A prior label distribution. A pandas.Series indexed by labels and holding corresponding label&#x27;s probability of occurrence. Each probability is between 0 and 1, all probabilities should sum up to 1</p>
`errors_`|**Optional\[DataFrame\]**|<p>Performers&#x27; error matrices. A pandas.DataFrame indexed by `performer` and `label` with a column for every label_id found in `data` such that `result.loc[performer, observed_label, true_label]` is the probability of `performer` producing an `observed_label` given that a task&#x27;s true label is `true_label`</p>

**Examples:**

```python
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
ds = DawidSkene(100)
result = ds.fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit.md)| Fit the model through the EM-algorithm.
[fit_predict](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.dawid_skene.DawidSkene.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
