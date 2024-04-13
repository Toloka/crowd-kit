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

* Ustalov D., Pavlichenko N., Tseitlin B. (2024). [Learning from Crowds with Crowd-Kit](https://doi.org/10.21105/joss.06227). Journal of Open Source Software, 9(96), 6227

```bibtex
@article{CrowdKit,
  author    = {Ustalov, Dmitry and Pavlichenko, Nikita and Tseitlin, Boris},
  title     = {{Learning from Crowds with Crowd-Kit}},
  year      = {2024},
  journal   = {Journal of Open Source Software},
  volume    = {9},
  number    = {96},
  pages     = {6227},
  publisher = {The Open Journal},
  doi       = {10.21105/joss.06227},
  issn      = {2475-9066},
  eprint    = {2109.08584},
  eprinttype = {arxiv},
  eprintclass = {cs.HC},
  language  = {english},
}
```
