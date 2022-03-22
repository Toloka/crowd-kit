1.0.0
-------------------
**Not a backward-compatible change**:
* **Replaced all mentions of "performer" with "worker". This change is not backward compatible because parameters names and DataFrame/Series columns are also affected.**

Improvements:
* `GoldMajorityVote` `true_labels` argument now supports multiple ground truth values for a single task.
* Added `tol` optimization parameter as a tolerance stopping criteria for iterative methods with a variable number of steps.
* Python 3.10 support added.
* Enhanced aggregation methods descriptions.

0.0.9
-------------------
* Added TextSummarization aggregation
* Added new datasets
* Added `entropy_threshold` method
* Added names for pd.Series which are available after `fit`
* Added `on_missing_skill` and `default_skill` params for models that use skills

0.0.8
-------------------
* Added GLAD aggregation
* Fixed https://github.com/Toloka/crowd-kit/issues/6
* Fixed https://github.com/Toloka/crowd-kit/issues/3


0.0.7
-------------------
* Added segmentation EM
* Added ROVER
* Fixed HRRASA and refactored TextRASA and TextHRRASA


0.0.6
-------------------
* Added realization of inter-annotator agreement coefficient (Krippendorff 1980):
  * `alpha_krippendorff`
* Added usage examples.


0.0.5
-------------------
* Added aggregations for image segmentation problem:
  * `SegmentationMajorityVote`
  * `SegmentationRASA`
