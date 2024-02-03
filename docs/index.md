# Home

**Crowd-Kit** is a powerful Python library that implements commonly-used aggregation methods for crowdsourced annotation and offers the relevant metrics and datasets. We strive to implement functionality that simplifies working with crowdsourced data.

[![Crowd-Kit](https://tlk.s3.yandex.net/crowd-kit/Crowd-Kit-GitHub.png)](https://github.com/Toloka/crowd-kit)

Currently, Crowd-Kit contains:

* implementations of commonly-used aggregation methods for categorical, pairwise, textual, and segmentation responses;
* implementations of deep learning from crowds methods and advanced aggregation algorithms in PyTorch;
* metrics of uncertainty, consistency, and agreement with aggregate;
* loaders for popular crowdsourced datasets.

## Installing

To install Crowd-Kit, run the following command: `pip install crowd-kit`. If you also want to use the `learning` subpackage, type `pip install crowd-kit[learning]`.

## Getting Started

Crowd-Kit's API resembles the one of scikit-learn. We recommend checking out our examples at <https://github.com/Toloka/crowd-kit/tree/main/examples>.

## Citation

* Ustalov D., Pavlichenko N., Tseitlin B. [Learning from Crowds with Crowd-Kit](https://arxiv.org/abs/2109.08584). 2023. arXiv: [2109.08584 [cs.HC]](https://arxiv.org/abs/2109.08584).

```bibtex
@misc{CrowdKit,
  author    = {Ustalov, Dmitry and Pavlichenko, Nikita and Tseitlin, Boris},
  title     = {{Learning from Crowds with Crowd-Kit}},
  year      = {2023},
  publisher = {arXiv},
  eprint    = {2109.08584},
  eprinttype = {arxiv},
  eprintclass = {cs.HC},
  url       = {https://arxiv.org/abs/2109.08584},
  language  = {english},
}
```
