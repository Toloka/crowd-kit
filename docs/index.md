# Crowd-Kit: Computational Quality Control for Crowdsourcing

**Crowd-Kit** is a powerful Python library that implements commonly-used aggregation methods for crowdsourced annotation and offers the relevant metrics and datasets. We strive to implement functionality that simplifies working with crowdsourced data.

Currently, Crowd-Kit contains:

* implementations of commonly-used aggregation methods for categorical, pairwise, textual, and segmentation responses
* metrics of uncertainty, consistency, and agreement with aggregate
* loaders for popular crowdsourced datasets

## Installing

Installing Crowd-Kit is as easy as `pip install crowd-kit`

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

| Method | Status |
| ------------- | :-------------: |
| [Majority Vote](reference/crowdkit.aggregation.classification.majority_vote.MajorityVote.md) | ✅ |
| [Dawid-Skene](reference/crowdkit.aggregation.classification.dawid_skene.DawidSkene.md) | ✅ |
| [Gold Majority Vote](reference/crowdkit.aggregation.classification.gold_majority_vote.GoldMajorityVote.md) | ✅ |
| [M-MSR](reference/crowdkit.aggregation.classification.m_msr.MMSR.md) | ✅ |
| [Wawa](reference/crowdkit.aggregation.classification.wawa.Wawa.md) | ✅ |
| [Zero-Based Skill](reference/crowdkit.aggregation.classification.zero_based_skill.ZeroBasedSkill.md) | ✅ |
| [GLAD](reference/crowdkit.aggregation.classification.glad.GLAD.md) | ✅ |
| BCC | 🟡 |

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

## Questions and Bug Reports

* For reporting bugs please use the [Toloka/bugreport](https://github.com/Toloka/crowdlib/issues) page.
* Join our English-speaking [slack community](https://toloka.ai/community) for both tech and abstract questions.

## Source Code

* [Crowd-Kit on GitHub](https://github.com/Toloka/crowd-kit)

## Toloka Global Community

Stay informed about updates to the platform and open-source libraries — connect with the Toloka team in our [Global Community](https://join.slack.com/t/tolokacommunity/shared_invite/zt-sxr745fr-dvfZffzvQTwNXOE0gEqysg) for announcements, discussions, and invites to events.

## License

© YANDEX LLC, 2020-2022. Licensed under the Apache License, Version 2.0. See LICENSE file for more details.
