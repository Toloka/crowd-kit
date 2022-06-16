__all__ = [
    'consistency',
    'uncertainty',
    'alpha_krippendorff',
]
import crowdkit.aggregation.base
import nltk.metrics.distance
import pandas
import typing


def consistency(
    answers: pandas.DataFrame,
    workers_skills: typing.Optional[pandas.Series] = None,
    aggregator: crowdkit.aggregation.base.BaseClassificationAggregator = ...,
    by_task: bool = False
) -> typing.Union[float, pandas.Series]:
    """Consistency metric: posterior probability of aggregated label given workers skills
    calculated using standard Dawid-Skene model.
    Args:
        answers (pandas.DataFrame): A data frame containing `task`, `worker` and `label` columns.
        workers_skills (Optional[pandas.Series]): workers skills e.g. golden set skills. If not provided,
            uses aggregator's `workers_skills` attribute.
        aggregator (aggregation.base.BaseClassificationAggregator): aggregation method, default: MajorityVote
        by_task (bool): if set, returns consistencies for every task in provided data frame.

    Returns:
        Union[float, pd.Series]
    """
    ...


def uncertainty(
    answers: pandas.DataFrame,
    workers_skills: typing.Optional[pandas.Series] = None,
    aggregator: typing.Optional[crowdkit.aggregation.base.BaseClassificationAggregator] = None,
    compute_by: str = 'task',
    aggregate: bool = True
) -> typing.Union[float, pandas.Series]:
    """Label uncertainty metric: entropy of labels probability distribution.
    Computed as Shannon's Entropy with label probabilities computed either for tasks or workers:
    $$H(L) = -\sum_{label_i \in L} p(label_i) \cdot \log(p(label_i))$$
    Args:
        answers: A data frame containing `task`, `worker` and `label` columns.
        workers_skills: workers skills e.g. golden set skills. If not provided,
            but aggregator provided, uses aggregator's `workers_skills` attribute.
            Otherwise assumes equal skills for workers.
        aggregator: aggregation method to obtain
            worker skills if not provided.
        compute_by: what to compute uncertainty for. If 'task', compute uncertainty of answers per task.
            If 'worker', compute uncertainty for each worker.
        aggregate: If true, return the mean uncertainty, otherwise return uncertainties for each task or worker.

    Returns:
        Union[float, pd.Series]

    Examples:
        Mean task uncertainty minimal, as all answers to task are same.

        >>> uncertainty(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        >>> ]))
        0.0

        Mean task uncertainty maximal, as all answers to task are different.

        >>> uncertainty(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'No'},
        >>>     {'task': 'X', 'worker': 'C', 'label': 'Maybe'},
        >>> ]))
        1.0986122886681096

        Uncertainty by task without averaging.

        >>> uncertainty(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'No'},
        >>>     {'task': 'Y', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'Y', 'worker': 'B', 'label': 'Yes'},
        >>> ]),
        >>> workers_skills=pd.Series([1, 1], index=['A', 'B']),
        >>> compute_by="task", aggregate=False)
        task
        X    0.693147
        Y    0.000000
        dtype: float64

        Uncertainty by worker

        >>> uncertainty(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'No'},
        >>>     {'task': 'Y', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'Y', 'worker': 'B', 'label': 'Yes'},
        >>> ]),
        >>> workers_skills=pd.Series([1, 1], index=['A', 'B']),
        >>> compute_by="worker", aggregate=False)
        worker
        A    0.000000
        B    0.693147
        dtype: float64
    Args:
        answers (DataFrame): Workers' labeling results.
            A pandas.DataFrame containing `task`, `worker` and `label` columns.
        workers_skills (typing.Optional[pandas.core.series.Series]): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
    """
    ...


def alpha_krippendorff(answers: pandas.DataFrame, distance: typing.Callable[[typing.Hashable, typing.Hashable], float] = nltk.metrics.distance.binary_distance) -> float:
    """Inter-annotator agreement coefficient (Krippendorff 1980).

    Amount that annotators agreed on label assignments beyond what is expected by chance.
    The value of alpha should be interpreted as follows.
        alpha >= 0.8 indicates a reliable annotation,
        alpha >= 0.667 allows making tentative conclusions only,
        while the lower values suggest the unreliable annotation.

    Args:
        answers: A data frame containing `task`, `worker` and `label` columns.
        distance: Distance metric, that takes two arguments,
            and returns a value between 0.0 and 1.0
            By default: binary_distance (0.0 for equal labels 1.0 otherwise).

    Returns:
        Float value.

    Examples:
        Consistent answers.

        >>> alpha_krippendorff(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        >>>     {'task': 'Y', 'worker': 'A', 'label': 'No'},
        >>>     {'task': 'Y', 'worker': 'B', 'label': 'No'},
        >>> ]))
        1.0

        Partially inconsistent answers.

        >>> alpha_krippendorff(pd.DataFrame.from_records([
        >>>     {'task': 'X', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'X', 'worker': 'B', 'label': 'Yes'},
        >>>     {'task': 'Y', 'worker': 'A', 'label': 'No'},
        >>>     {'task': 'Y', 'worker': 'B', 'label': 'No'},
        >>>     {'task': 'Z', 'worker': 'A', 'label': 'Yes'},
        >>>     {'task': 'Z', 'worker': 'B', 'label': 'No'},
        >>> ]))
        0.4444444444444444
    """
    ...
