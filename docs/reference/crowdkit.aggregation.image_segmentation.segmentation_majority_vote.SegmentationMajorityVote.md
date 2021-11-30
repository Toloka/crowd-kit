# SegmentationMajorityVote
`crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote`

```
SegmentationMajorityVote(
    self,
    on_missing_skill: str = 'error',
    default_skill: Optional[float] = None
)
```

Majority Vote - chooses a pixel if more than half of performers voted


Doris Jung-Lin Lee. 2018.
Quality Evaluation Methods for Crowdsourced Image Segmentation
http://ilpubs.stanford.edu:8090/1161/1/main.pdf

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**Series**|<p>Tasks&#x27; segmentations A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
`on_missing_skill`|**str**|<p>How to handle assignments done by workers with unknown skill Possible values:<ul><li>&quot;error&quot; — raise an exception if there is at least one assignment done by user with unknown skill;</li><li>&quot;ignore&quot; — drop assignments with unknown skill values during prediction. Raise an exception if there is no  assignments with known skill for any task;</li><li>value — default value will be used if skill is missing.</li></ul></p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit.md)| None
[fit_predict](crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.fit_predict.md)| None
