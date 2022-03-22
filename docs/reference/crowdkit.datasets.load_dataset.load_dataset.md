# load_dataset
`crowdkit.datasets.load_dataset.load_dataset` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/datasets/load_dataset.py#L11)

```python
load_dataset(dataset: str, data_dir: Optional[str] = None)
```

Downloads a dataset from remote and loads it into Pandas objects.


If a dataset is already downloaded, loads it from cache.

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`dataset`|**str**|<p>str, a dataset name</p>
`data_dir`|**Optional\[str\]**|<p>Optional[str] Path to folder where to store downloaded dataset. If `None`, `~/crowdkit_data` is used. `default=None`. Alternatively, it can be set by the &#x27;CROWDKIT_DATA&#x27; environment variable.</p>

* **Returns:**

  Tuple[pd.DataFrame, pd.Series], a tuple of workers answers and ground truth labels.

* **Return type:**

  Tuple\[DataFrame, Series\]
