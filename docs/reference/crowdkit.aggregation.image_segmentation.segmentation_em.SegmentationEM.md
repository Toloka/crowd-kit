# SegmentationEM
`crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM`

```
SegmentationEM(self, n_iter: int = 10)
```

The EM algorithm for the image segmentation task.


For each task, EM algorithm is performed to classify the image pixels.
Doris Jung-Lin Lee. 2018.
Quality Evaluation Methods for Crowdsourced Image Segmentation
http://ilpubs.stanford.edu:8090/1161/1/main.pdf

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM.fit.md)| None
[fit_predict](crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM.fit_predict.md)| None
