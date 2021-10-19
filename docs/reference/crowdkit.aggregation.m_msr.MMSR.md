# MMSR
`crowdkit.aggregation.m_msr.MMSR`

```
MMSR(
    self,
    n_iter: int = 10000,
    eps: float = ...,
    random_state: Optional[int] = 0,
    observation_matrix: ndarray = ...,
    covariation_matrix: ndarray = ...,
    n_common_tasks: ndarray = ...,
    n_performers: int = 0,
    n_tasks: int = 0,
    n_labels: int = 0,
    labels_mapping: dict = ...,
    performers_mapping: dict = ...,
    tasks_mapping: dict = ...
)
```

Matrix Mean-Subsequence-Reduced Algorithm


Qianqian Ma and Alex Olshevsky. 2020.
Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion
34th Conference on Neural Information Processing Systems (NeurIPS 2020)
[https://arxiv.org/abs/2010.12181](https://arxiv.org/abs/2010.12181)

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`skills_`|**Optional\[Series\]**|<p>Performers&#x27; skills A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>
`scores_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; label scores A pandas.DataFrame indexed by `task` such that `result.loc[task, label]` is the score of `label` for `task`.</p>
`labels_`|**Optional\[DataFrame\]**|<p>Tasks&#x27; most likely true labels A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s most likely true label.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.m_msr.MMSR.fit.md)| None
[fit_predict](crowdkit.aggregation.m_msr.MMSR.fit_predict.md)| None
[fit_predict_score](crowdkit.aggregation.m_msr.MMSR.fit_predict_score.md)| None
[predict](crowdkit.aggregation.m_msr.MMSR.predict.md)| None
[predict_score](crowdkit.aggregation.m_msr.MMSR.predict_score.md)| None
