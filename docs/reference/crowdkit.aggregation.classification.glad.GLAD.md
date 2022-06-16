# GLAD
`crowdkit.aggregation.classification.glad.GLAD` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/glad.py#L31)

```python
GLAD(
    self,
    n_iter: int = 100,
    tol: float = 1e-05,
    silent: bool = True,
    labels_priors: Optional[Series] = None,
    alphas_priors_mean: Optional[Series] = None,
    betas_priors_mean: Optional[Series] = None,
    m_step_max_iter: int = 25,
    m_step_tol: float = 0.01
)
```

Generative model of Labels, Abilities, and Difficulties.


A probabilistic model that parametrizes workers' abilities and tasks' dificulties.
Let's consider a case of $K$ class classification. Let $p$ be a vector of prior class probabilities,
$\alpha_i \in (-\infty, +\infty)$ be a worker's ability parameter, $\beta_j \in (0, +\infty)$ be
an inverse task's difficulty, $z_j$ be a latent variable representing the true task's label, and $y^i_j$
be a worker's response that we observe. The relationships between this variables and parameters according
to GLAD are represented by the following latent label model:

![GLAD latent label model](https://tlk.s3.yandex.net/crowd-kit/docs/glad_llm.png)


The prior probability of $z_j$ being equal to $c$ is
$$
\operatorname{Pr}(z_j = c) = p[c],
$$
the probability distribution of the worker's responses conditioned by the true label value $c$ follows the
single coin Dawid-Skene model where the true label probability is a sigmoid function of the product of
worker's ability and inverse task's difficulty:
$$
\operatorname{Pr}(y^i_j = k | z_j = c) = \begin{cases}a(i, j), & k = c \\ \frac{1 - a(i,j)}{K-1}, & k \neq c\end{cases},
$$
where
$$
a(i,j) = \frac{1}{1 + \exp(-\alpha_i\beta_j)}.
$$

Parameters $p$, $\alpha$, $\beta$ and latent variables $z$ are optimized through the Expectation-Minimization algorithm.


J. Whitehill, P. Ruvolo, T. Wu, J. Bergsma, and J. Movellan.
Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise.
*Proceedings of the 22nd International Conference on Neural Information Processing Systems*, 2009

<https://proceedings.neurips.cc/paper/2009/file/f899139df5e1059396431415e770c6dd-Paper.pdf>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`max_iter`|**-**|<p>Maximum number of EM iterations.</p>
`eps`|**-**|<p>Threshold for convergence criterion.</p>
`silent`|**bool**|<p>If false, show progress bar.</p>
`labels_priors`|**Optional\[Series\]**|<p>Prior label probabilities.</p>
`alphas_priors_mean`|**Optional\[Series\]**|<p>Prior mean value of alpha parameters.</p>
`betas_priors_mean`|**Optional\[Series\]**|<p>Prior mean value of beta parameters.</p>
`m_step_max_iter`|**int**|<p>Maximum number of iterations of conjugate gradient method in M-step.</p>
`m_step_tol`|**float**|<p>Tol parameter of conjugate gradient method in M-step.</p>
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`alphas_`|**Series**|<p>workers&#x27; alpha parameters. A pandas.Series indexed by `worker` that contains estimated alpha parameters.</p>
`betas_`|**Series**|<p>Tasks&#x27; beta parameters. A pandas.Series indexed by `task` that contains estimated beta parameters.</p>

**Examples:**

```python
from crowdkit.aggregation import GLAD
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
glad = GLAD()
result = glad.fit_predict(df)
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.glad.GLAD.fit.md)| Fit the model through the EM-algorithm.
[fit_predict](crowdkit.aggregation.classification.glad.GLAD.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.glad.GLAD.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
