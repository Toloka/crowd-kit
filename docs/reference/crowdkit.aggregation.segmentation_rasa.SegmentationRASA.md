# SegmentationRASA
`crowdkit.aggregation.segmentation_rasa.SegmentationRASA`

```
SegmentationRASA(self, n_iter: int = 10)
```

Segmentation RASA - chooses a pixel if sum of weighted votes of each performers' more than 0.5.


Algorithm works iteratively, at each step, the performers are reweighted in proportion to their distances
to the current answer estimation. The distance is considered as 1-iou. Modification of the RASA method for texts.

Jiyi Li. 2019.
A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP,
pages 24–28 Hong Kong, China, November 3, 2019.
[http://doi.org/10.18653/v1/D19-5904](http://doi.org/10.18653/v1/D19-5904)

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**ndarray**|<p>Tasks&#x27; segmentations A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.segmentation_rasa.SegmentationRASA.fit.md)| None
[fit_predict](crowdkit.aggregation.segmentation_rasa.SegmentationRASA.fit_predict.md)| None
