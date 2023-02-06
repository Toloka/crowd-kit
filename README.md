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

* implementations of commonly-used aggregation methods for categorical, pairwise, textual, and segmentation responses;
* metrics of uncertainty, consistency, and agreement with aggregate;
* loaders for popular crowdsourced datasets.

Also, the `learning` subpackage contains PyTorch implementations of deep learning from crowds methods and advanced aggregation algorithms.

## Installing

To install Crowd-Kit, run the following command: `pip install crowd-kit`. If you also want to use the `learning` subpackage, type `pip instal crowd-kit[learning]`.

If you are interested in contributing to Crowd-Kit, use [Pipenv](https://pipenv.pypa.io/en/latest/) to install the library with its dependencies: `pipenv install --dev`. We use [pytest](https://docs.pytest.org/en/7.1.x/) for testing.

## Getting Started

This example shows how to use Crowd-Kit for categorical aggregation using the classical Dawid-Skene algorithm.

First, let us do all the necessary imports.

````python
from crowdkit.aggregation import DawidSkene
from crowdkit.datasets import load_dataset

import pandas as pd
````

Then, you need to read your annotations into Pandas DataFrame with columns `task`, `worker`, `label`. Alternatively, you can download an example dataset:

````python
df = pd.read_csv('results.csv')  # should contain columns: task, worker, label
# df, ground_truth = load_dataset('relevance-2')  # or download an example dataset
````

Then, you can aggregate the workers' responses using the `fit_predict` method from the **scikit-learn** library:

````python
aggregated_labels = DawidSkene(n_iter=100).fit_predict(df)
````

[More usage examples](https://github.com/Toloka/crowd-kit/tree/main/examples)

## Implemented Aggregation Methods

Below is the list of currently implemented methods, including the already available (✅) and in progress (🟡).

### Categorical Responses

| Method | Status |
| ------------- | :-------------: |
| [Majority Vote](reference/crowdkit.aggregation.classification.majority_vote.MajorityVote.md) | ✅ |
| [One-coin Dawid-Skene](reference/crowdkit.aggregation.classification.dawid_skene.OneCoinDawidSkene.md) | ✅ |
| [Dawid-Skene](reference/crowdkit.aggregation.classification.dawid_skene.DawidSkene.md) | ✅ |
| [Gold Majority Vote](reference/crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.md) | ✅ |
| [M-MSR](reference/crowdkit.aggregation.classification.m_msr.MMSR.md) | ✅ |
| [Wawa](reference/crowdkit.aggregation.classification.wawa.Wawa.md) | ✅ |
| [Zero-Based Skill](reference/crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.md) | ✅ |
| [GLAD](reference/crowdkit.aggregation.classification.glad.GLAD.md) | ✅ |
| [KOS](reference/crowdkit.aggregation.classification.kos.KOS.md) | ✅ |
| [MACE](reference/crowdkit.aggregation.classification.mace.MACE.md) | ✅ |
| BCC | 🟡 |

### Multi-Label Responses

|Method|Status|
|-|:-:|
|[Binary Relevance](reference/crowdkit.aggregation.multilabel.binary_relevance.BinaryRelevance.md)|✅|

### Textual Responses

| Method | Status |
| ------------- | :-------------: |
| [RASA](reference/crowdkit.aggregation.embeddings.rasa.RASA.md) | ✅ |
| [HRRASA](reference/crowdkit.aggregation.embeddings.hrrasa.HRRASA.md) | ✅ |
| [ROVER](reference/crowdkit.aggregation.texts.rover.ROVER.md) | ✅ |

### Image Segmentation

| Method | Status |
| ------------------ | :------------------: |
| [Segmentation MV](reference/crowdkit.aggregation.image_segmentation.segmentation_majority_vote.SegmentationMajorityVote.md) | ✅ |
| [Segmentation RASA](reference/crowdkit.aggregation.image_segmentation.segmentation_rasa.SegmentationRASA.md) | ✅ |
| [Segmentation EM](reference/crowdkit.aggregation.image_segmentation.segmentation_em.SegmentationEM.md) | ✅ |

### Pairwise Comparisons

| Method | Status |
| -------------- | :---------------------: |
| [Bradley-Terry](reference/crowdkit.aggregation.pairwise.bradley_terry.BradleyTerry.md) | ✅ |
| [Noisy Bradley-Terry](reference/crowdkit.aggregation.pairwise.noisy_bt.NoisyBradleyTerry.md) | ✅ |

### Learning from Crowds

|Method|Status|
|-|:-:|
|[CrowdLayer](reference/crowdkit.learning.crowd_layer.CrowdLayer.md)|✅|
|[CoNAL](reference/crowdkit.learning.conal.Conal.md)|✅|

## Citation

* Ustalov D., Pavlichenko N., Losev V., Giliazev I., and Tulin E. [A General-Purpose Crowdsourcing Computational Quality Control Toolkit for Python](https://www.humancomputation.com/2021/assets/wips_demos/HCOMP_2021_paper_85.pdf). *The Ninth AAAI Conference on Human Computation and Crowdsourcing: Works-in-Progress and Demonstration Track.* HCOMP 2021. 2021. arXiv: [2109.08584 [cs.HC]](https://arxiv.org/abs/2109.08584).

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
  url       = {https://www.humancomputation.com/2021/assets/wips_demos/HCOMP_2021_paper_85.pdf},
  language  = {english},
}
```

## Questions and Bug Reports

* To report a bug, post an issue on the [Toloka/bugreport](https://github.com/Toloka/crowdlib/issues) page.
* To find answers to common questions or start a new discussion, join our English-speaking [Slack community](https://toloka.ai/community).

## License

&copy; Crowd-Kit team authors, 2020&ndash;2023. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
