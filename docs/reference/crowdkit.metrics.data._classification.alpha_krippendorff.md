# alpha_krippendorff
`crowdkit.metrics.data._classification.alpha_krippendorff` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/metrics/data/_classification.py#L196)

```python
alpha_krippendorff(answers: DataFrame, distance: Callable[[Hashable, Hashable], float] = binary_distance)
```

Inter-annotator agreement coefficient (Krippendorff 1980).


Amount that annotators agreed on label assignments beyond what is expected by chance.
The value of alpha should be interpreted as follows.
    alpha >= 0.8 indicates a reliable annotation,
    alpha >= 0.667 allows making tentative conclusions only,
    while the lower values suggest the unreliable annotation.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`answers`|**DataFrame**|<p>A data frame containing `task`, `worker` and `label` columns.</p>
`distance`|**Callable\[\[Hashable, Hashable\], float\]**|<p>Distance metric, that takes two arguments, and returns a value between 0.0 and 1.0 By default: binary_distance (0.0 for equal labels 1.0 otherwise).</p>

* **Returns:**

  Float value.

* **Return type:**

  float

**Examples:**

Consistent answers.

```python
alpha_krippendorff(pd.DataFrame.from_records([
    {'task': 'X', 'worker': 'A', 'label': 'Yes'},
    {'task': 'X', 'worker': 'B', 'label': 'Yes'},
    {'task': 'Y', 'worker': 'A', 'label': 'No'},
    {'task': 'Y', 'worker': 'B', 'label': 'No'},
]))
```

Partially inconsistent answers.

```python
alpha_krippendorff(pd.DataFrame.from_records([
    {'task': 'X', 'worker': 'A', 'label': 'Yes'},
    {'task': 'X', 'worker': 'B', 'label': 'Yes'},
    {'task': 'Y', 'worker': 'A', 'label': 'No'},
    {'task': 'Y', 'worker': 'B', 'label': 'No'},
    {'task': 'Z', 'worker': 'A', 'label': 'Yes'},
    {'task': 'Z', 'worker': 'B', 'label': 'No'},
]))
```
