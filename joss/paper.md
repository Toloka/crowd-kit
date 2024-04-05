---
title: 'Learning from Crowds with Crowd-Kit'
tags:
  - Python
  - crowdsourcing
  - data labeling
  - answer aggregation
  - truth inference
  - learning from crowds
  - machine learning
  - quality control
  - data quality
authors:
  - name: Dmitry Ustalov
    orcid: 0000-0002-9979-2188
    affiliation: 1
    corresponding: true
  - name: Nikita Pavlichenko
    orcid: 0000-0002-7330-393X
    affiliation: 2
  - name: Boris Tseitlin
    orcid: 0000-0001-8553-4260
    affiliation: 3
affiliations:
 - name: JetBrains, Serbia
   index: 1
 - name: JetBrains, Germany
   index: 2
 - name: Planet Farms, Portugal
   index: 3
date: 24 September 2023
bibliography: paper.bib
---

# Summary

This paper presents Crowd-Kit, a general-purpose computational quality control toolkit for crowdsourcing. Crowd-Kit provides efficient and convenient implementations of popular quality control algorithms in Python, including methods for truth inference, deep learning from crowds, and data quality estimation. Our toolkit supports multiple modalities of answers and provides dataset loaders and example notebooks for faster prototyping. We extensively evaluated our toolkit on several datasets of different natures, enabling benchmarking computational quality control methods in a uniform, systematic, and reproducible way using the same codebase. We release our code and data under the Apache License 2.0 at <https://github.com/Toloka/crowd-kit>.

# Statement of need

A traditional approach to quality control in crowdsourcing builds upon various organizational means, such as careful task design, decomposition, and preparing golden tasks [@Zhdanovskaya:23]. These techniques yield the best results when accompanied by computational methods that leverage worker-task-label relationships and their statistical properties.

Many studies in crowdsourcing simplify complex tasks via multi-classification or post-acceptance steps, as discussed in a pivotal paper by @Bernstein:10. Meanwhile, researchers in natural language processing and computer vision develop specialized techniques. However, existing toolkits like SQUARE [@Sheshadri:13], CEKA [@Zhang:15], Truth Inference [@Zheng:17], spark-crowd [@Rodrigo:19] require additional effort for integration into applications, popular data science libraries and frameworks.

We propose addressing this challenge with **Crowd-Kit**, an open-source Python toolkit for computational quality control in crowdsourcing. Crowd-Kit implements popular quality control methods, providing a standardized platform for reliable experimentation and application. We extensively evaluate the Crowd-Kit library to establish a basis for comparisons. *In all the experiments in this paper, we used our implementations of the corresponding methods.*

# Design

Our fundamental goal of Crowd-Kit development was to bridge the gap between crowdsourcing research and vivid data science ecosystem of NumPy, SciPy, pandas [@McKinney:10], and scikit-learn [@Pedregosa:11]. We implemented Crowd-Kit in Python and employed the highly optimized data structures and algorithms available in these libraries, maintaining compatibility with the application programming interface (API) of scikit-learn and data frames/series of pandas. Even for a user not familiar with crowdsourcing but familiar with scientific computing and data analysis in Python, the basic API usage would be straightforward:

```python
# df is a DataFrame with labeled data in form of (task, label, worker)
# gt is a Series with ground truth per task
df, gt = load_dataset('relevance-2')  # binary relevance sample dataset

# run the Dawid-Skene categorical aggregation method
agg_ds = DawidSkene(n_iter=10).fit_predict(df)  # same format as gt
```

We implemented all the methods in Crowd-Kit from scratch in Python. Although unlike spark-crowd [@Rodrigo:19], our library did not provide a means for running on a distributed computational cluster, it leveraged efficient implementations of numerical algorithms in underlying libraries widely used in the research community. In addition to categorical aggregation methods, Crowd-Kit offers non-categorical aggregation methods, dataset loaders, and annotation quality estimators.

# Maintenance and governance

Crowd-Kit is not bound to any specific crowdsourcing platform, allowing analyzing data from any crowdsourcing marketplace (as soon as one can download the labeled data from that platform). Crowd-Kit is an open-source library working under most operating systems and available under the Apache license 2.0 both on GitHub and Python Package Index (PyPI).[^1] All code of Crowd-Kit has strict type annotations for additional safety and clarity. By the time of submission, our library had a test coverage of 93%.

[^1]: <https://github.com/Toloka/crowd-kit> & <https://pypi.org/project/crowd-kit/>

We built Crowd-Kit on top of the established open-source frameworks and best practices. We widely use the continuous integration facilities via GitHub Actions for two purposes. First, every patch (*commit* in git terminology) invokes unit testing and coverage, type checking, linting, documentation and packaging dry run. Second, every release is automatically submitted to PyPI directly from GitHub Actions via the trusted publishing mechanism to avoid potential side effects on the individual developer machines. Besides commit checks, every code change (*pull request* on GitHub) goes through a code review by the Crowd-Kit developers. We accept bug reports via GitHub Issues.

# Functionality

Crowd-Kit implements a selection of popular methods for answer aggregation and learning from crowds, dataset loaders, and annotation quality characteristics.

## Aggregating and learning with Crowd-Kit

Crowd-Kit features aggregation methods suitable for most kinds of crowdsourced responses, including categorical, pairwise, sequential, and image segmentation answers (see the summary in \autoref{tab:methods}).

Methods for *categorical aggregation*, which are the most widespread in practice, assume that there is only one correct objective label per task and aim at recovering a latent true label from the observed noisy data. Some of these methods, such as Dawid-Skene and GLAD, also estimate latent parameters --- aka skills --- of the workers. Where the task design does not meet the latent label assumption, Crowd-Kit offers methods for aggregation *pairwise comparisons*, which are essential for subjective opinion gathering. Also, Crowd-Kit provides specialized methods for aggregating *sequences* (such as texts) and *image segmentation*. All these aggregation methods are implemented purely using NumPy, SciPy, pandas, and scikit-learn without any deep learning framework. Last but not least, Crowd-Kit offers methods for *deep learning from crowds* that learn an end-to-end machine learning model from raw responses submitted by the workers without the use of aggregation, which are available as ready-to-use modules for PyTorch [@Paszke:19].

One can easily add a new aggregation method to Crowd-Kit. For example, without the loss of generality, to create a new categorical aggregator, one should extend the base class `BaseClassificationAggregator` and implement two methods, `fit()` and `fit_predict()`, filling the instance variable `labels_` with the aggregated labels.[^2] Also, to add a new method for learning from crowds, one has to create a subclass from `torch.nn.Module` and implement the `forward()` method.[^3]

: Summary of the implemented methods in Crowd-Kit.\label{tab:methods}

| **Aggregation** | **Methods**                                               |
|-----------------|-----------------------------------------------------------|
| Categorical     | Majority Vote, Wawa [@Wawa], @Dawid:79,                   |
|                 | GLAD [@Whitehill:09], MACE [@Hovy:13],                    |
|                 | @Karger:14, M-MSR [@Ma:20]                                |
| Pairwise        | @Bradley:52, noisyBT [@Bugakova:19]                       |
| Sequence        | ROVER [@Fiscus:97], RASA and HRRASA [@Li:20],             |
|                 | Language Model [@Pavlichenko:21:crowdspeech]              |
| Segmentation    | Majority Vote, Expectation-Maximization [@JungLinLee:18], |
|                 | RASA and HRRASA [@Li:20]                                  |
| Learning        | CrowdLayer [@Rodrigues:18], CoNAL [@Chu:21]               |

## Dataset loaders

Crowd-Kit offers convenient dataset loaders for some popular or demonstrative datasets (see \autoref{tab:datasets}), allowing downloading them from the Internet in a ready-to-use form with a single line of code. It is possible to add new datasets in a declarative way and, if necessary, add the corresponding code to load the data as pandas data frames and series.

: Summary of the datasets provided by Crowd-Kit.\label{tab:datasets}

| **Task**     | **Datasets** |
|--------------|---------------------------------------------------------------|
| Categorical  | Toloka Relevance 2 and 5, TREC Relevance [@Buckley:10]        |
| Pairwise     | IMDB-WIKI-SbS [@Pavlichenko:21:sbs]                           |
| Sequence     | CrowdWSA [-@Li:19], CrowdSpeech [@Pavlichenko:21:crowdspeech] |
| Image        | Common Objects in Context [@Lin:14]                           |

[^2]: See the implementation of Majority Vote at <https://github.com/Toloka/crowd-kit/blob/main/crowdkit/aggregation/classification/majority_vote.py> as an example of an aggregation method.

[^3]: See the implementation of CrowdLayer at <https://github.com/Toloka/crowd-kit/blob/main/crowdkit/learning/crowd_layer.py> as an example of a method for deep learning from crowds.

## Annotation quality estimators

Crowd-Kit allows one to apply commonly-used techniques to evaluate data and annotation quality, providing a unified pandas-compatible API to compute $\alpha$ [@Krippendorff:18], annotation uncertainty [@Malinin:19], agreement with aggregate [@Wawa], Dawid-Skene posterior probability, etc.

# Evaluation

We extensively evaluate Crowd-Kit methods for answer aggregation and learning from crowds. When possible, we compare with other authors; either way, we show how the currently implemented methods perform on well-known datasets with noisy crowdsourced data, indicating the correctness of our implementations.

## Evaluation of aggregation methods

**Categorical.** To ensure the correctness of our implementations, we compared the observed aggregation quality with the already available implementations by @Zheng:17 and @Rodrigo:19. \autoref{tab:categorical} shows evaluation results, indicating a similar level of quality as them: *D_Product*, *D_PosSent*, *S_Rel*, and *S_Adult* are real-world datasets from @Zheng:17, and *binary1* and *binary2* are synthetic datasets from @Rodrigo:19. Our implementation of M-MSR could not process the D_Product dataset in a reasonable time, KOS can be applied to binary datasets only, and none of our implementations handled *binary3* and *binary4* synthetic datasets, which require a distributed computing cluster.

: Comparison of the implemented categorical aggregation methods (accuracy is used).\label{tab:categorical}

| **Method**  | **D_Product** | **D_PosSent** | **S_Rel** | **S_Adult** | **binary1** | **binary2** |
| ------------|--------------:|--------------:|----------:|------------:|------------:|------------:|
| MV          |    $0.897$    |    $0.932$    |  $0.536$  |   $0.763$   |   $0.931$   |   $0.936$   |
| Wawa        |    $0.897$    |    $0.951$    |  $0.557$  |   $0.766$   |   $0.981$   |   $0.983$   |
| DS          |    $0.940$    |    $0.960$    |  $0.615$  |   $0.748$   |   $0.994$   |   $0.994$   |
| GLAD        |    $0.928$    |    $0.948$    |  $0.511$  |   $0.760$   |   $0.994$   |   $0.994$   |
| KOS         |    $0.895$    |    $0.933$    |    ---    |     ---     |   $0.993$   |   $0.994$   |
| MACE        |    $0.929$    |    $0.950$    |  $0.501$  |   $0.763$   |   $0.995$   |   $0.995$   |
| M-MSR       |      ---      |    $0.937$    |  $0.425$  |   $0.751$   |   $0.994$   |   $0.994$   |

**Pairwise.** \autoref{tab:pairwise} shows the comparison of the *Bradley-Terry* and *noisyBT* methods implemented in Crowd-Kit to the random baseline on the graded readability dataset by @Chen:13 and a larger people age dataset by @Pavlichenko:21:sbs.

: Comparison of implemented pairwise aggregation methods (Spearman's $\rho$ is used).\label{tab:pairwise}

| **Method**     | **@Chen:13** | **IMDB-WIKI-SBS** |
|----------------|-------------:|------------------:|
| Bradley-Terry  |    $0.246$   |      $0.737$      |
| noisyBT        |    $0.238$   |      $0.744$      |
| Random         |   $-0.013$   |     $-0.001$      |

**Sequence.** We used two datasets, CrowdWSA [@Li:19] and CrowdSpeech [@Pavlichenko:21:crowdspeech]. As the typical application for sequence aggregation in crowdsourcing is audio transcription, we used the word error rate as the quality criterion [@Fiscus:97] in \autoref{tab:sequence}.

: Comparison of implemented sequence aggregation methods (average word error rate is used).\label{tab:sequence}

| **Dataset**  | **Version** | **ROVER** | **RASA** | **HRRASA** |
|--------------|:-----------:|----------:|---------:|-----------:|
| CrowdWSA     |     J1      |  $0.612$  | $0.659$  |  $0.676$   |
|              |     T1      |  $0.514$  | $0.483$  |  $0.500$   |
|              |     T2      |  $0.524$  | $0.498$  |  $0.520$   |
| CrowdSpeech  |  dev-clean  |  $0.676$  | $0.750$  |  $0.745$   |
|              |  dev-other  |  $0.132$  | $0.142$  |  $0.142$   |
|              | test-clean  |  $0.729$  | $0.860$  |  $0.859$   |
|              | test-other  |  $0.134$  | $0.157$  |  $0.157$   |

**Segmentation.** We annotated on the Toloka crowdsourcing platform a sample of 2,000 images from the MS COCO [@Lin:14] dataset consisting of four object labels. For each image, nine workers submitted segmentations. The dataset is available in Crowd-Kit as `mscoco_small`. In total, we received 18,000 responses. \autoref{tab:segmentation} shows the comparison of the methods on the above-described dataset using the *intersection over union* (IoU) criterion.

: Comparison of implemented image aggregation algorithms (IoU is used).\label{tab:segmentation}

| **Dataset**  | **MV**  | **EM**  | **RASA** |
|--------------|--------:|--------:|---------:|
| MS COCO      | $0.839$ | $0.861$ | $0.849$  |

## Evaluation of methods for learning from crowds

To demonstrate the impact of learning on raw annotator labels compared to answer aggregation in crowdsourcing, we compared the implemented methods for learning from crowds with the two classical aggregation algorithms, Majority Vote (MV) and Dawid-Skene (DS). We picked the two most common machine learning tasks for which ground truth datasets are available: text classification and image classification. For text classification, we used the IMDB Movie Reviews dataset [@Maas:11], and for image classification, we chose CIFAR-10 [@Krizhevsky:09]. In each dataset, each object was annotated by three different annotators; 100 objects were used as golden tasks.

We compared how different methods for learning from crowds impact test accuracy. We picked two different backbone networks for text classification, LSTM [@Hochreiter:97] and RoBERTa [@Liu:19], and one backbone network for image classification, VGG-16 [@Simonyan:15]. Then, we trained each backbone in three scenarios: use the fully connected layer after the backbone without taking into account any specifics of crowdsourcing (Base), CrowdLayer method by @Rodrigues:18, and CoNAL method by @Chu:21. \autoref{tab:learning} shows the evaluation results.

: Comparison of different methods for deep learning from crowds with traditional answer aggregation methods (test set accuracy is used).\label{tab:learning}

| **Dataset**  | **Backbone** | **CoNAL** | **CrowdLayer** | **Base** | **DS**  | **MV**  |
|--------------|:------------:|----------:|---------------:|---------:|--------:|--------:|
| IMDb         |     LSTM     |  $0.844$  |    $0.825$     | $0.835$  | $0.841$ | $0.819$ |
| IMDb         |   RoBERTa    |  $0.932$  |    $0.928$     | $0.927$  | $0.932$ | $0.927$ |
| CIFAR-10     |    VGG-16    |  $0.825$  |    $0.863$     | $0.882$  | $0.877$ | $0.865$ |

Our experiment shows the feasibility of training a deep learning model directly from the raw annotated data, skipping trivial aggregation methods like MV. However, specialized methods like CoNAL and CrowdLayer or non-trivial aggregation methods like DS can significantly enhance prediction accuracy. It is crucial to make a well-informed model selection to achieve optimal results. We believe that Crowd-Kit can seamlessly integrate these methods into machine learning pipelines that utilize crowdsourced data with reliability and ease.

# Conclusion

Our experience running Crowd-Kit in production for processing crowdsourced data at Toloka shows that it successfully handles industry-scale datasets without needing a large compute cluster. We believe that the availability of computational quality control techniques in a standardized way would open new venues for reliable improvement of the crowdsourcing quality beyond the traditional well-known methods and pipelines.

# Acknowledgements

The work was done while the authors were with Yandex. We are grateful to Enrique G. Rodrigo for sharing the spark-crowd evaluation dataset. We want to thank Daniil Fedulov, Iulian Giliazev, Artem Grigorev, Daniil Likhobaba, Vladimir Losev, Stepan Nosov, Alisa Smirnova, Aleksey Sukhorosov, and Evgeny Tulin for their contributions to the library. We received no external funding.

# References
