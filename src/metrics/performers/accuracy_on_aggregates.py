from typing import Optional, Union

import pandas as pd
from crowdkit.aggregation import MajorityVote
from crowdkit.aggregation.base_aggregator import BaseAggregator
from crowdkit.aggregation.utils import get_accuracy


def accuracy_on_aggregates(answers: pd.DataFrame,
                           aggregator: Optional[BaseAggregator] = MajorityVote(),
                           aggregates: Optional[pd.Series] = None,
                           by: Optional[str] = None) -> Union[float, pd.Series]:
    """
    Accuracy on aggregates: a fraction of worker's answers that match the aggregated one.
    Args:
            answers (pandas.DataFrame): a data frame containing `task`, `performer` and `label` columns.
            aggregator (aggregation.BaseAggregator): aggregation algorithm. default: MajorityVote
            aggregates (Optional[pandas.Series]): aggregated answers for provided tasks.
            by_performer (bool): if set, returns accuracies for every performer in provided data frame. Otherwise,
                returns an average accuracy of all performers.
            answer_column: column in the data frame that contanes performers answers.
            comapre_function (Optional[Callable[[Any, Any], float]]): function that compares performer's answer with
                the golden answer. If `None`, uses `==` operator.

        Returns:
            Union[float, pd.Series]
    """
    if aggregates is None and aggregator is None:
        raise AssertionError('One of aggregator or aggregates should be not None')
    if aggregates is None:
        aggregates = aggregator.fit_predict(answers)
    return get_accuracy(answers, aggregates, by=by)
