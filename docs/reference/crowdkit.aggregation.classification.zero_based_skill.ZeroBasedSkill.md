# ZeroBasedSkill
`crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/classification/zero_based_skill.py#L13)

```python
ZeroBasedSkill(
    self,
    n_iter: int = 100,
    lr_init: float = 1.0,
    lr_steps_to_reduce: int = 20,
    lr_reduce_factor: float = 0.5,
    eps: float = 1e-05
)
```

The Zero-Based Skill aggregation model.


Performs weighted majority voting on tasks. After processing a pool of tasks,
re-estimates workers' skills through a gradient descend step of optimization
of the mean squared error of current skills and the fraction of responses that
are equal to the aggregated labels.

Repeats this process until labels do not change or the number of iterations exceeds.

It's necessary that all workers in a dataset that send to 'predict' existed in answers
the dataset that was sent to 'fit'.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>A number of iterations to perform.</p>
`lr_init`|**float**|<p>A starting learning rate.</p>
`lr_steps_to_reduce`|**int**|<p>A number of steps necessary to decrease the learning rate.</p>
`lr_reduce_factor`|**float**|<p>A factor that the learning rate will be multiplied by every `lr_steps_to_reduce` steps.</p>
`eps`|**float**|<p>A convergence threshold.</p>

**Examples:**

```python
from crowdkit.aggregation import ZeroBasedSkill
from crowdkit.datasets import load_dataset
df, gt = load_dataset('relevance-2')
result = ZeroBasedSkill().fit_predict(df)
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit_predict.md)| Fit the model and return aggregated results.
[fit_predict_proba](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit_predict_proba.md)| Fit the model and return probability distributions on labels for each task.
[predict](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.predict.md)| Infer the true labels when the model is fitted.
[predict_proba](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.predict_proba.md)| Return probability distributions on labels for each task when the model is fitted.
