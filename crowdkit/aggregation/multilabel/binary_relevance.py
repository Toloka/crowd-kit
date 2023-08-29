__all__ = ['BinaryRelevance']

from typing import Dict, List, Union, Any

import attr
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from ..classification import MajorityVote

from ..base import BaseClassificationAggregator
from ..utils import clone_aggregator


@attr.s
class BinaryRelevance(BaseClassificationAggregator):
    r"""The **Binary Relevance** algorithm is a simple aggregation algorithm for the multi-label classification.

    Binary Relevance is a straightforward approach for the multi-label classification aggregation:
    each label is represented as a class in the binary classification problem and aggregated separately using
    aggregation algorithms for classification (e.g., Majority Vote or Dawid-Skene).

    The multi-label training set $D$ is specified in the following way:
    $$
    D = {(x^i, y^i) | 1 <= i <= m},
    $$
    where $m$ is a number of multi-label training examples.
    For each multi-label training example $(x^i, y^i)$, $x^i \in X$ is a d-dimensional feature vector $[x_1^i, x_2^i, ..., x_d^i]^T$
    and $y^i \in {-1, +1}^q$ is a q-bits binary vector $[y_1^i, y_2^i, ..., y_q^i]^T$ where $y_j^i$ is a relevant (irrelevant) label
    for $x^i$ ($1 <= j <= q$ where $q$ is a number of class labels).

    For each class label $λ_j$, Binary Relevance derives a binary training set $D_j$ from the original
    multi-label training set $D$ in the following way:
    $$
    D_j = {(x^i, y_j^i) | 1 <= i <= m}.
    $$
    In other words, each multi-label training example $(x^i, y^i)$ is transformed into a binary training example
    based on its relevancy to $λ_j$.

    {% note info %}

    If this method is used for the single-label classification, the output of the Binary Relevance method may differ
    from the output of the basic aggregator used for its intended purpose since each class generates a binary
    classification task, and therefore it is considered separately. For example, some objects may not have labels.

    {% endnote %}

    M-L. Zhang, Y-K. Li, X-Y. Liu, X. Geng. Binary Relevance for Multi-Label Learning: An Overview.
    *Frontiers of Computer Science. Vol. 12*, 2 (2018), 191-202.

    <http://palm.seu.edu.cn/zhangml/files/FCS'17.pdf>

    Args:
        base_aggregator: The aggregator instance that will be used for each binary classification. All class parameters
         will be copied, except for the results of the previous fit.

    Examples:
        >>> import pandas as pd
        >>> from crowdkit.aggregation import BinaryRelevance, DawidSkene
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'w1', ['house', 'tree']],
        >>>         ['t1', 'w2', ['house']],
        >>>         ['t1', 'w3', ['house', 'tree', 'grass']],
        >>>         ['t2', 'w1', ['car']],
        >>>         ['t2', 'w2', ['car', 'human']],
        >>>         ['t2', 'w3', ['train']]
        >>>     ]
        >>> )
        >>> df.columns = ['task', 'worker', 'label']
        >>> result = BinaryRelevance(DawidSkene(n_iter=10)).fit_predict(df)

    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): The task labels.
            The `pandas.Series` data is indexed by `task` so that `labels.loc[task]` is a list of the task aggregated labels.

        aggregators_ (dict[str, BaseClassificationAggregator]): The label aggregators matched to the classes.
            It is represented as a dictionary that matches the aggregators to the classes.
            The key is a class found in the source data,
            and the value is an aggregator used for this class.
            The set of keys is all the classes that are used in the input data.
    """
    base_aggregator: BaseClassificationAggregator = attr.ib(
        # validator=attr.validators.instance_of(BaseClassificationAggregator),
        default=MajorityVote())
    aggregators_: Dict[str, BaseClassificationAggregator] = dict()

    @base_aggregator.validator
    def _any_name_except_a_name_of_an_attribute(self, attribute: Any, value: Any) -> None:
        assert issubclass(value.__class__, BaseClassificationAggregator), \
            "Aggregator argument should be a classification aggregator"

    def fit(self, data: pd.DataFrame) -> 'BinaryRelevance':
        """Fits the model to the training data.

        Args:
            data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.
                The `label` column should contain a list of labels (e.g., ['tree', 'house', 'car']).

        Returns:
            BinaryRelevance: self.
        """

        data = data[['task', 'worker', 'label']]
        mlb = MultiLabelBinarizer()
        binarized_labels = mlb.fit_transform(data['label'])
        task_to_labels: Dict[Union[str, float], List[Union[str, float]]] = dict()

        for i, label in enumerate(mlb.classes_):
            single_label_df = data[['task', 'worker']]
            single_label_df['label'] = binarized_labels[:, i]

            label_aggregator = clone_aggregator(self.base_aggregator)
            label_aggregator.fit_predict(single_label_df)
            self.aggregators_[label] = label_aggregator
            if label_aggregator.labels_ is not None:  # for mypy correct work
                for task, label_value in label_aggregator.labels_.items():
                    if task not in task_to_labels:
                        task_to_labels[task] = list()
                    if label_value:
                        task_to_labels[task].append(label)
        if not task_to_labels:
            self.labels_ = pd.Series(task_to_labels, dtype=float)
        else:
            self.labels_ = pd.Series(task_to_labels)
        if len(self.labels_):
            self.labels_.index.name = 'task'
        return self

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fits the model to the training data and returns the aggregated results.

         Args:
             data (DataFrame): The training dataset of workers' labeling results
                which is represented as the `pandas.DataFrame` data containing `task`, `worker`, and `label` columns.

         Returns:
             Series: Task labels.
                 The `pandas.Series` data is indexed by `task` so that `labels.loc[task]`
                 is a list of the task aggregated labels.
         """
        return self.fit(data).labels_
