from typing import Any, Callable, Optional, Union
import pandas as pd

from crowdkit.aggregation.base_aggregator import BaseAggregator
from crowdkit.aggregation import MajorityVote
from .golden_set_accuracy import golden_set_accuracy


def accuracy_on_aggregates(answers: pd.DataFrame,
                           aggregator: Optional[BaseAggregator] = MajorityVote(),
                           aggregates: Optional[pd.Series] = None,
                           by_performer: bool = False,
                           answer_column: Any = 'label',
                           comapre_function: Optional[Callable[[Any, Any], float]] = None) -> Union[float, pd.Series]:
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
        aggregates = aggregator.fit_predict(answers).set_index('task')[answer_column]
    return golden_set_accuracy(answers, aggregates, by_performer, answer_column, comapre_function)
