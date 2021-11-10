# GLAD
`crowdkit.aggregation.classification.glad.GLAD`

```
GLAD(
    self,
    max_iter: int = 100,
    eps: float = ...,
    silent: bool = True,
    labels_priors: Optional[Series] = None,
    alphas_priors_mean: Optional[Series] = None,
    betas_priors_mean: Optional[Series] = None,
    m_step_max_iter: int = 25,
    m_step_tol: float = ...
)
```

Generative model of Labels, Abilities, and Difficulties


J. Whitehill, P. Ruvolo, T. Wu, J. Bergsma, and J. Movellan
Whose Vote Should Count More: Optimal Integration of Labels from Labelers of Unknown Expertise.
Proceedings of the 22nd International Conference on Neural Information Processing Systems, 2009

https://proceedings.neurips.cc/paper/2009/file/f899139df5e1059396431415e770c6dd-Paper.pdf

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`probas_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label probability distributions A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the probability of `task`&#x27;s true label to be equal to `label`. Each probability is between 0 and 1, all task&#x27;s probabilities should sum up to 1</p>
`alphas_`|**Series**|<p>Performers&#x27; alpha parameters A pandas.Series indexed by `performer` that containes estimated alpha parameters.</p>
`betas_`|**Series**|<p>Tasks&#x27; beta parameters A pandas.Series indexed by `task` that containes estimated beta parameters.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.glad.GLAD.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.glad.GLAD.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.glad.GLAD.fit_predict_proba.md)| None
