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
    r"""Simple aggregation algorithm for multi-label classification.

    Binary Relevance is a straightforward approach for multi-label classification aggregation:
    each label is treated as a class in binary classification problem and aggregated separately using
    aggregation algorithms for classification, e.g. Majority Vote or Dawid Skene.

    {% note info %}

    If this method is used for single-label classification, the output of the BinaryRelevance method may differ
    from the output of the basic aggregator used for its intended purpose, since each class generates a binary
    classification task, and therefore it is considered separately. For example, some objects may not have labels.

    {% endnote %}

    Args:
        base_aggregator: Aggregator instance that will be used for each binary classification. All class parameters
         will be copied, except for the results of previous fit.

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
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks' aggregated labels.

        aggregators_ (dict[str, BaseClassificationAggregator]): Labels' aggregators matched to classes.
            A dictionary that matches aggregators to classes.
            The key is the class found in the source data,
            and the value is the aggregator used for this class.
            The set of keys is all the classes that are in the input data.
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
        """Fit the aggregators.

        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
                'label' column should contain list of labels, e.g. ['tree', 'house', 'car']

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
                for task, label_value in label_aggregator.labels_.iteritems():
                    if task not in task_to_labels:
                        task_to_labels[task] = list()
                    if label_value:
                        task_to_labels[task].append(label)
        self.labels_ = pd.Series(task_to_labels)
        if len(self.labels_):
            self.labels_.index.name = 'task'
        return self

    def fit_predict(self, data: pd.DataFrame) -> pd.Series:
        """Fit the model and return aggregated results.

         Args:
             data (DataFrame): Workers' labeling results.
                 A pandas.DataFrame containing `task`, `worker` and `label` columns.

         Returns:
             Series: Tasks' labels.
                 A pandas.Series indexed by `task` such that `labels.loc[task]`
                 is a list with the task's aggregated labels.
         """
        return self.fit(data).labels_
