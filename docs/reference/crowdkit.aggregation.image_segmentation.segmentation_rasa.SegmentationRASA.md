# SegmentationRASA
`crowdkit.aggregation.image_segmentation.segmentation_rasa.SegmentationRASA`

```python
SegmentationRASA(
    self,
    n_iter: int = 10,
    tol: float = 1e-05
)
```

Segmentation RASA - chooses a pixel if sum of weighted votes of each performers' more than 0.5.


Algorithm works iteratively, at each step, the performers are reweighted in proportion to their distances
to the current answer estimation. The distance is considered as $1 - IOU$. Modification of the RASA method
for texts.

Jiyi Li.
A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
*Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*,
pages 24–28 Hong Kong, China, November 3, 2019.
<http://doi.org/10.18653/v1/D19-5904>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>A number of iterations.</p>
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>

**Examples:**

```python
import numpy as np
import pandas as pd
from crowdkit.aggregation import SegmentationRASA
df = pd.DataFrame(
    [
        ['t1', 'p1', np.array([[1, 0], [1, 1]])],
        ['t1', 'p2', np.array([[0, 1], [1, 1]])],
        ['t1', 'p3', np.array([[0, 1], [1, 1]])]
    ],
    columns=['task', 'performer', 'segmentation']
)
result = SegmentationRASA().fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.image_segmentation.segmentation_rasa.SegmentationRASA.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.image_segmentation.segmentation_rasa.SegmentationRASA.fit_predict.md)| Fit the model and return the aggregated segmentations.
