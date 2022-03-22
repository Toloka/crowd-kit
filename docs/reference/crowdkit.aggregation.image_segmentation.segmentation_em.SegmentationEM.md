# SegmentationEM
`crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM` | [Source code](https://github.com/Toloka/crowd-kit/blob/v1.0.0/crowdkit/aggregation/image_segmentation/segmentation_em.py#L12)

```python
SegmentationEM(
    self,
    n_iter: int = 10,
    tol: float = 1e-05
)
```

The EM algorithm for the image segmentation task.


This method performs a categorical aggregation task for each pixel: should
it be included to the resulting aggregate or no. This task is solved by
the single coin Dawid-Skene algorithm. Each worker has a latent parameter
"skill" that shows the probability of this worker to answer correctly.
Skills and true pixels' labels are optimized by the Expectation-Maximization
algorithm.


Doris Jung-Lin Lee. 2018.
Quality Evaluation Methods for Crowdsourced Image Segmentation
<http://ilpubs.stanford.edu:8090/1161/1/main.pdf>

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`n_iter`|**int**|<p>A number of EM iterations.</p>
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations. A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>

**Examples:**

```python
import numpy as np
import pandas as pd
from crowdkit.aggregation import SegmentationEM
df = pd.DataFrame(
    [
        ['t1', 'p1', np.array([[1, 0], [1, 1]])],
        ['t1', 'p2', np.array([[0, 1], [1, 1]])],
        ['t1', 'p3', np.array([[0, 1], [1, 1]])]
    ],
    columns=['task', 'worker', 'segmentation']
)
result = SegmentationEM().fit_predict(df)
```
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM.fit.md)| Fit the model.
[fit_predict](crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM.fit_predict.md)| Fit the model and return the aggregated segmentations.
