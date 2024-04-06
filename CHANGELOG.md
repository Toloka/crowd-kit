1.3.0.post0
-----------

- Article reference update: <https://doi.org/10.21105/joss.06227>

1.3.0
-----

* Added [NetEaseCrowd](https://github.com/fuxiAIlab/NetEaseCrowd-Dataset) dataset
* Fixed bugs in RASA #71, relevance #74, MV #81, GLAD #83
* Switched to MkDocs
* Major typing improvements
* Added more example notebooks
* Set up PyPI Trusted Publishing

1.2.1
-----

* Bug fixes in RASA, HRRASA, and documentation

1.2.0
-----

* Crowd-Kit Learning subpackage introducing implementations of deep learning from crowds methods: CoNAL and CrowdLayer
* Added Multi-Binary aggregation

1.1.0
-----
* New aggregation methods: One-Coin Dawid Skene, MACE, and KOS
* Fixed bugs in Dawid-Skene implementation
* Improved maintainability by removing stub files
* Switched to `setup.cfg` from `setup.py`

1.0.0
-----

* **Breaking change.** Replaced all mentions of `performer` with `worker`; this affects parameter names and `DataFrame` and `Series` entries
* `GoldMajorityVote` `true_labels` argument now supports multiple ground truth values for a single task
* Added `tol` optimization parameter as a tolerance stopping criteria for iterative methods with a variable number of steps
* Python 3.10 support added
* Enhanced aggregation methods descriptions

0.0.9
-----

* Added `TextSummarization` aggregation
* Added new datasets
* Added `entropy_threshold` method
* Added names for pd.Series which are available after `fit`
* Added `on_missing_skill` and `default_skill` params for models that use skills

0.0.8
-----

* Added GLAD aggregation
* Fixed #3 and #6

0.0.7
-----

* Added segmentation EM
* Added ROVER
* Fixed HRRASA and refactored TextRASA and TextHRRASA

0.0.6
-----

* Added realization of inter-annotator agreement coefficient (Krippendorff 1980): `alpha_krippendorff`
* Added usage examples

0.0.5
-----

* Added aggregations for image segmentation problem: `SegmentationMajorityVote` and `SegmentationRASA`
