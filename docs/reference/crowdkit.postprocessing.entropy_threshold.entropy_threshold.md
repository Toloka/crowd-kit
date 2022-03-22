# entropy_threshold
`crowdkit.postprocessing.entropy_threshold.entropy_threshold` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/postprocessing/entropy_threshold.py#L11)

```python
entropy_threshold(
    answers: DataFrame,
    workers_skills: Optional[Series] = None,
    percentile: int = 10,
    min_answers: int = 2
)
```

Entropy thresholding postprocessing: filters out all answers by workers,


whos' entropy (uncertanity) of answers is below specified percentile.

This heuristic detects answers of workers that answer the same way too often, e.g. when "speed-running" by only
clicking one button.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`answers`|**DataFrame**|<p>Workers&#x27; labeling results. A pandas.DataFrame containing `task`, `worker` and `label` columns.</p>
`workers_skills`|**Optional\[Series\]**|<p>workers&#x27; skills. A pandas.Series index by workers and holding corresponding worker&#x27;s skill</p>

* **Returns:**

  pd.DataFrame

* **Return type:**

  

**Examples:**

Fraudent worker always answers the same and gets filtered out.

```python
answers = pd.DataFrame.from_records(
    [
        {'task': '1', 'performer': 'A', 'label': frozenset(['dog'])},
        {'task': '1', 'performer': 'B', 'label': frozenset(['cat'])},
        {'task': '2', 'performer': 'A', 'label': frozenset(['cat'])},
        {'task': '2', 'performer': 'B', 'label': frozenset(['cat'])},
        {'task': '3', 'performer': 'A', 'label': frozenset(['dog'])},
        {'task': '3', 'performer': 'B', 'label': frozenset(['cat'])},
    ]
)
entropy_threshold(answers)
```
