# NoisyBradleyTerry
`crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/pairwise/noisy_bt.py#L14)

```python
NoisyBradleyTerry(
    self,
    n_iter: int = 100,
    tol: float = 1e-05,
    random_state: int = 0
)
```

A modification of Bradley-Terry with parameters for workers' skills and


their biases.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`scores_`|**Series**|<p>&#x27;Labels&#x27; scores. A pandas.Series index by labels and holding corresponding label&#x27;s scores</p>
`skills_`|**Series**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>
`biases_`|**Series**|<p>Predicted biases for each worker. Indicates the probability of a worker to choose the left item.. A series of workers&#x27; biases indexed by workers</p>
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit.md)| None
[fit_predict](crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.fit_predict.md)| None
