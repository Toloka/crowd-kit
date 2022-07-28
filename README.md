# Crowd-Kit: Computational Quality Control for Crowdsourcing

[![Crowd-Kit](https://tlk.s3.yandex.net/crowd-kit/Crowd-Kit-GitHub.png)](https://github.com/Toloka/crowd-kit)

[![GitHub Tests][github_tests_badge]][github_tests_link]
[![Codecov][codecov_badge]][codecov_link]
[![Documentation][docs_badge]][docs_link]

[github_tests_badge]: https://github.com/Toloka/crowdlib/workflows/Tests/badge.svg?branch=main
[github_tests_link]: https://github.com/Toloka/crowdlib/actions?query=workflow:Tests
[codecov_badge]: https://codecov.io/gh/Toloka/crowd-kit/branch/main/graph/badge.svg
[codecov_link]: https://codecov.io/gh/Toloka/crowd-kit
[docs_badge]: https://img.shields.io/badge/docs-toloka.ai-1E2126
[docs_link]: https://toloka.ai/en/docs/crowd-kit/

**Crowd-Kit** is a powerful Python library that implements commonly-used aggregation methods for crowdsourced annotation and offers the relevant metrics and datasets. We strive to implement functionality that simplifies working with crowdsourced data.

Currently, Crowd-Kit contains:

* implementations of commonly-used aggregation methods for categorical, pairwise, textual, and segmentation responses
* metrics of uncertainty, consistency, and agreement with aggregate
* loaders for popular crowdsourced datasets

## Installing

Installing Crowd-Kit is as easy as `pip install crowd-kit`.

Those who are interested in contributing to Crowd-Kit can use [Pipenv](https://pipenv.pypa.io/) to install the library with its dependencies: `pipenv install --dev`. We use [pytest](https://pytest.org/) for testing.

## Getting Started

This example shows how to use Crowd-Kit for categorical aggregation using the classical Dawid-Skene algorithm.

First, let us do all the necessary imports.

````python
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset

import pandas as pd
````

Then, you need to read your annotations into Pandas DataFrame with columns `task`, `worker`, `label`. Alternatively, you can download an example dataset.

````python
df = pd.read_csv('results.csv')  # should contain columns: task, worker, label
# df, ground_truth = load_dataset('relevance-2')  # or download an example dataset
````

Then you can aggregate the worker responses as easily as in scikit-learn:

````python
aggregated_labels = DawidSkene(n_iter=100).fit_predict(df)
````

[More usage examples](https://github.com/Toloka/crowd-kit/tree/main/examples)

## Implemented Aggregation Methods

Below is the list of currently implemented methods, including the already available (✅) and in progress (🟡).

### Categorical Responses

|Method|Status|
|-|:-:|
|Majority Vote|✅|
|One-coin Dawid-Skene|✅|
|[Dawid-Skene](https://doi.org/10.2307/2346806)|✅|
|Gold Majority Vote|✅|
|[M-MSR](https://proceedings.neurips.cc/paper/2020/hash/f86890095c957e9b949d11d15f0d0cd5-Abstract.html)|✅|
|Wawa|✅|
|Zero-Based Skill|✅|
|[GLAD](https://papers.nips.cc/paper/3644-whose-vote-should-count-more-optimal-integration-of-labels-from-labelers-of-unknown-expertise.pdf)|✅|
|[KOS](https://papers.nips.cc/paper/2011/file/c667d53acd899a97a85de0c201ba99be-Paper.pdf)|✅|
|[MACE](https://aclanthology.org/N13-1132.pdf)|✅|
|BCC|🟡|

### Textual Responses

|Method|Status|
|-|:-:|
|[RASA](https://doi.org/10.18653/v1/D19-5904)|✅|
|[HRRASA](https://doi.org/10.1145/3397271.3401239)|✅|
|[ROVER](https://ieeexplore.ieee.org/document/659110)|✅|
|[Language Model-Based](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/65ded5353c5ee48d0b7d48c591b8f430-Abstract-round1.html)|✅|

### Image Segmentation

|Method|Status|
|-|:-:|
|Segmentation MV|✅|
|Segmentation RASA|✅|
|Segmentation EM|✅|

### Pairwise Comparisons

|Method|Status|
|-|:-:|
|[Bradley-Terry](https://doi.org/10.2307/2334029)|✅|
|Noisy Bradley-Terry|✅|

## Citation

* Ustalov D., Pavlichenko N., Losev V., Giliazev I., and Tulin E. [A General-Purpose Crowdsourcing Computational Quality Control Toolkit for Python](https://www.humancomputation.com/assets/wips_demos/HCOMP_2021_paper_85.pdf). *The Ninth AAAI Conference on Human Computation and Crowdsourcing: Works-in-Progress and Demonstration Track.* HCOMP 2021. 2021. arXiv: [2109.08584 [cs.HC]](https://arxiv.org/abs/2109.08584).

```(bibtex)
@inproceedings{HCOMP2021/CrowdKit,
  author    = {Ustalov, Dmitry and Pavlichenko, Nikita and Losev, Vladimir and Giliazev, Iulian and Tulin, Evgeny},
  title     = {{A General-Purpose Crowdsourcing Computational Quality Control Toolkit for Python}},
  year      = {2021},
  booktitle = {The Ninth AAAI Conference on Human Computation and Crowdsourcing: Works-in-Progress and Demonstration Track},
  series    = {HCOMP~2021},
  eprint    = {2109.08584},
  eprinttype = {arxiv},
  eprintclass = {cs.HC},
  url       = {https://www.humancomputation.com/assets/wips_demos/HCOMP_2021_paper_85.pdf},
  language  = {english},
}
```

## Questions and Bug Reports

* For reporting bugs please use the [Toloka/bugreport](https://github.com/Toloka/crowdlib/issues) page.
* Join our English-speaking [slack community](https://toloka.ai/community) for both tech and abstract questions.

## License

© YANDEX LLC, 2020-2022. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
