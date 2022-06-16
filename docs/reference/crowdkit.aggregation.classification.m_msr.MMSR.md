# MMSR
`crowdkit.aggregation.classification.m_msr.MMSR` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/m_msr.py#L17)

```python
MMSR(
    self,
    n_iter: int = 10000,
    tol: float = 1e-10,
    random_state: Optional[int] = 0,
    observation_matrix: ndarray = ...,
    covariation_matrix: ndarray = ...,
    n_common_tasks: ndarray = ...,
    n_workers: int = 0,
    n_tasks: int = 0,
    n_labels: int = 0,
    labels_mapping: dict = ...,
    workers_mapping: dict = ...,
    tasks_mapping: dict = ...
)
```

Matrix Mean-Subsequence-Reduced Algorithm.


The M-MSR assumes that workers have different level of expertise and associated
with a vector of "skills" $\boldsymbol{s}$ which entries $s_i$ show the probability
of the worker $i$ to answer correctly to the given task. Having that, we can show that
$$
\mathbb{E}\left[\frac{M}{M-1}\widetilde{C}-\frac{1}{M-1}\boldsymbol{1}\boldsymbol{1}^T\right]
 = \boldsymbol{s}\boldsymbol{s}^T,
$$
where $M$ is the total number of classes, $\widetilde{C}$ is a covariation matrix between
workers, and $\boldsymbol{1}\boldsymbol{1}^T$ is the all-ones matrix which has the same
size as $\widetilde{C}$.


So, the problem of recovering the skills vector $\boldsymbol{s}$ becomes equivalent to the
rank-one matrix completion problem. The M-MSR algorithm is an iterative algorithm for *rubust*
rank-one matrix completion, so its result is an estimator of the vector $\boldsymbol{s}$.
Then, the aggregation is the weighted majority vote with weights equal to
$\log \frac{(M-1)s_i}{1-s_i}$.

Matrix Mean-Subsequence-Reduced Algorithm. Qianqian Ma and Alex Olshevsky.
Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion.
*34th Conference on Neural Information Processing Systems (NeurIPS 2020)*

<https://arxiv.org/abs/2010.12181>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>The maximum number of iterations of the M-MSR algorithm.</p>
`eps`|**-**|<p>Convergence threshold.</p>
`random_state`|**Optional\[int\]**|<p>Seed number for the random initialization.</p>
`labels_`|**Optional\[Series\]**|<p>Tasks&#x27; labels. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
`skills_`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`scores_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label scores. A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>

**Examples:**

```python
from crowdkit.aggregation import MMSR
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
mmsr = MMSR()
result = mmsr.fit_predict(df)
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.m_msr.MMSR.fit.md)| Estimate the workers' skills.
[fit_predict](crowdkit.aggregation.classification.m_msr.MMSR.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_score](crowdkit.aggregation.classification.m_msr.MMSR.fit_predict_score.md)| Fit the model and return the total sum of weights for each label.
[predict](crowdkit.aggregation.classification.m_msr.MMSR.predict.md)| Infer the true labels when the model is fitted.
[predict_score](crowdkit.aggregation.classification.m_msr.MMSR.predict_score.md)| Return total sum of weights for each label when the model is fitted.
