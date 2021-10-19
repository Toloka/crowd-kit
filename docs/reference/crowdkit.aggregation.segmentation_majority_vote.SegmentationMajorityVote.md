# SegmentationMajorityVote
`crowdkit.aggregation.segmentation_majority_vote.SegmentationMajorityVote`

Majority Vote - chooses a pixel if more than half of performers voted


Doris Jung-Lin Lee. 2018.
Quality Evaluation Methods for Crowdsourced Image Segmentation
[http://ilpubs.stanford.edu:8090/1161/1/main.pdf](http://ilpubs.stanford.edu:8090/1161/1/main.pdf)

## Parameters Description

| Parameters | Type | Description |
| :----------| :----| :-----------|
`segmentations_`|**ndarray**|<p>Tasks&#x27; segmentations A pandas.Series indexed by `task` such that `labels.loc[task]` is the tasks&#x27;s aggregated segmentation.</p>
## Methods summary

| Method | Description |
| :------| :-----------|
[fit](crowdkit.aggregation.segmentation_majority_vote.SegmentationMajorityVote.fit.md)| None
[fit_predict](crowdkit.aggregation.segmentation_majority_vote.SegmentationMajorityVote.fit_predict.md)| None
