# Crowd-Kit: Crowdsourcing Aggregation, Metrics and Datasets

[![GitHub Tests][github_tests_badge]][github_tests_link]
[![GitHub Tests][github_coverage_badge]][github_coverage_link]

[github_tests_badge]: https://github.com/Toloka/crowdlib/workflows/Tests/badge.svg?branch=main
[github_tests_link]: https://github.com/Toloka/crowdlib/actions?query=workflow:Tests
[github_coverage_badge]: https://codecov.io/gh/Toloka/crowd-kit/branch/main/graph/badge.svg
[github_coverage_link]: https://codecov.io/gh/Toloka/crowd-kit


`crowd-kit` is a powerful Python library that provides an implementation of commonly used aggregation methods for crowdsourced annotation, metrics, and datasets. We strive to implement functionality that eases working with crowdsourced data. Currently, the module contains:
* Implementations of commonly used aggregation methods for the following types of responses: **categorical**, **text**, **image segmentations** and **pair-wise comparisons**.
* A set of metrics
* Datasets for categorical aggregation

*The module is currently in a heavy development state, and interfaces are subject to change.*

Install
--------------
Installing Crowd-Kit is as easy as `pip install crowd-kit`


Getting Started
--------------
This example shows how to use Crowd-Kit for categorical aggregation using the Dawid-Skene algorithm.

First, let's do all the necessary imports.
````python
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset

import pandas as pd
````

Then, you need to read your annotations into Pandas DataFrame with columns `task`, `performer`, `label`. Alternatively, you can download an example dataset.

````python
crowd_annotations = pd.read_csv('your_dataset.csv')  # Should contatin columns ['task', 'performer', 'label']

# Or download an example dataset
# crowd_annotations, ground_truth = load_dataset('relevance-2')
````

After that, you can aggregate performers' responses as easily as fitting an `sklearn` model.

````python
aggregated_labels = DawidSkene(n_iter=100).fit_predict(crowd_annotations)
````

Implemented Aggregation Methods
--------------
### Categorical
| Method        | Status        |
| ------------- |:-------------:|
| Majority Vote | ✅            |
| [Dawid-Skene](https://doi.org/10.2307/2346806)   | ✅      |
| Gold Majority Vote | ✅      |
| [M-MSR](https://arxiv.org/abs/2010.12181) | ✅      |
| Wawa | ✅      |
| Zero-Based Skill | ✅      |
| GLAD | ❌      |
| BCC | ❌      |

### Text
| Method        | Status        |
| ------------- |:-------------:|
| [RASA](https://doi.org/10.1145/3397271.3401239) | ✅            |
| [HRRASA](https://doi.org/10.1145/3397271.3401239)   | ✅      |
| [ROVER](https://ieeexplore.ieee.org/document/659110) | ❌      |

### Image Segmentations
| Method        | Status        |
| ------------- |:-------------:|
| Segmentation MV | ✅            |
| Segmentation RASA   | ❌      |
| Segmentation EM | ❌      |

### Pair-Wise Comparisons
| Method        | Status        |
| ------------- |:-------------:|
| Bradley-Terry | ✅            |
| Noisy  Bradley-Terry  | ✅      |

Questions and bug reports
--------------
For reporting bugs please use the [Toloka/bugreport](https://github.com/Toloka/crowdlib/issues) page.


License
-------
© YANDEX LLC, 2020-2021. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
