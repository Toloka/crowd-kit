# ZeroBasedSkill
`crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill`

```
ZeroBasedSkill(
    self,
    n_iter: int = 100,
    lr_init: float = ...,
    lr_steps_to_reduce: int = 20,
    lr_reduce_factor: float = ...,
    eps: float = ...
)
```

The Zero-Based Skill aggregation model


Performs weighted majority voting on tasks. After processing a pool of tasks,
re-estimates performers' skills according to the correctness of their answers.
Repeats this process until labels do not change or the number of iterations exceeds.

It's necessary that all performers in a dataset that send to 'predict' existed in answers
the dataset that was sent to 'fit'.

## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit.md)| None
[fit_predict](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit_predict.md)| None
[fit_predict_proba](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.fit_predict_proba.md)| None
[predict](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.predict.md)| None
[predict_proba](crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.predict_proba.md)| None
