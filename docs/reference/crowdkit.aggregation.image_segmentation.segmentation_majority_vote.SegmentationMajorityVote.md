# SegmentationMajorityVote
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/image_segmentation/segmentation_majority_vote.py#L13)

```python
SegmentationMajorityVote(
    self,
    on_missing_skill: str = 'error',
    default_skill: Optional[float] = None
)
```

Segmentation Majority Vote - chooses a pixel if more than half of workers voted.


This method implements a straightforward approach to the image segmentations aggregation:
it assumes that if pixel is not inside in the worker's segmentation, this vote counts
as 0, otherwise, as 1. Next, the `SegmentationEM` aggregates these categorical values
for each pixel by the Majority Vote.

The method also supports weighted majority voting if `skills` were provided to `fit` method.

Doris Jung-Lin Lee. 2018.
Quality Evaluation Methods for Crowdsourced Image Segmentation
<https://ilpubs.stanford.edu:8090/1161/1/main.pdf>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`default_skill`|**Optional\[float\]**|<p>A default skill value for missing skills.</p>
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
`on_missing_skill`|**str**|<p>How to handle assignments done by workers with unknown skill. Possible values:<ul><li>&quot;error&quot; — raise an exception if there is at least one assignment done by user with unknown skill;</li><li>&quot;ignore&quot; — drop assignments with unknown skill values during prediction. Raise an exception if there is no  assignments with known skill for any task;</li><li>value — default value will be used if skill is missing.</li></ul></p>

**Examples:**

```python
import numpy as np
import pandas as pd
from crowdkit.aggregation import SegmentationMajorityVote
df = pd.DataFrame(
    [
        ['t1', 'p1', np.array([[1, 0], [1, 1]])],
        ['t1', 'p2', np.array([[0, 1], [1, 1]])],
        ['t1', 'p3', np.array([[0, 1], [1, 1]])]
    ],
    columns=['task', 'worker', 'segmentation']
)
result = SegmentationMajorityVote().fit_predict(df)
```
## Methods Summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit_predict.md)| Fit the model and return the aggregated segmentations.
