# manage_data
`crowdkit.aggregation.utils.manage_data`

```python
manage_data(
    data: DataFrame,
    weights: Optional[Series] = None,
    skills: Series = None
)
```

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`data`|**DataFrame**|<p>Performers&#x27; labeling results. A pandas.DataFrame containing `task`, `performer` and `label` columns.</p>
`skills`|**Series**|<p>Performers&#x27; skills. A pandas.Series index by performers and holding corresponding performer&#x27;s skill</p>
