__all__ = [
    'accuracy_on_aggregates',
]
import crowdkit.aggregation.base
import pandas
import typing


def accuracy_on_aggregates(
    answers: pandas.DataFrame,
    aggregator: typing.Optional[crowdkit.aggregation.base.BaseClassificationAggregator] = ...,
    aggregates: typing.Optional[pandas.Series] = None,
    by: typing.Optional[str] = None
) -> typing.Union[float, pandas.Series]:
    """Accuracy on aggregates: a fraction of worker's answers that match the aggregated one.
    Args:
        answers: a data frame containing `task`, `worker` and `label` columns.
        aggregator: aggregation algorithm. default: MajorityVote
        aggregates: aggregated answers for provided tasks.
        by: if set, returns accuracies for every worker in provided data frame. Otherwise,
            returns an average accuracy of all workers.

        Returns:
            Union[float, pd.Series]
    """
    ...
