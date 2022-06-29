__all__ = [
    'MMSR',
]
import crowdkit.aggregation.base
import numpy
import pandas
import typing


class MMSR(crowdkit.aggregation.base.BaseClassificationAggregator):
    """Matrix Mean-Subsequence-Reduced Algorithm.

    The M-MSR assumes that workers have different level of expertise and associated
    with a vector of "skills" $\boldsymbol{s}$ which entries $s_i$ show the probability
    of the worker $i$ to answer correctly to the given task. Having that, we can show that
    $$
    \mathbb{E}\left[\frac{M}{M-1}\widetilde{C}-\frac{1}{M-1}\boldsymbol{1}\boldsymbol{1}^T\right]
     = \boldsymbol{s}\boldsymbol{s}^T,
    $$
    where $M$ is the total number of classes, $\widetilde{C}$ is a covariation matrix between
    workers, and $\boldsymbol{1}\boldsymbol{1}^T$ is the all-ones matrix which has the same
    size as $\widetilde{C}$.


    So, the problem of recovering the skills vector $\boldsymbol{s}$ becomes equivalent to the
    rank-one matrix completion problem. The M-MSR algorithm is an iterative algorithm for *rubust*
    rank-one matrix completion, so its result is an estimator of the vector $\boldsymbol{s}$.
    Then, the aggregation is the weighted majority vote with weights equal to
    $\log \frac{(M-1)s_i}{1-s_i}$.

    Matrix Mean-Subsequence-Reduced Algorithm. Qianqian Ma and Alex Olshevsky.
    Adversarial Crowdsourcing Through Robust Rank-One Matrix Completion.
    *34th Conference on Neural Information Processing Systems (NeurIPS 2020)*

    <https://arxiv.org/abs/2010.12181>

    Args:
        n_iter: The maximum number of iterations of the M-MSR algorithm.
        eps: Convergence threshold.
        random_state: Seed number for the random initialization.

    Examples:
        >>> from crowdkit.aggregation import MMSR
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> mmsr = MMSR()
        >>> result = mmsr.fit_predict(df)
    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        skills_ (typing.Optional[pandas.core.series.Series]): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
        scores_ (typing.Optional[pandas.core.frame.DataFrame]): Tasks' label scores.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the score of `label` for `task`.
    """

    def fit(self, data: pandas.DataFrame) -> 'MMSR':
        """Estimate the workers' skills.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            MMSR: self.
        """
        ...

    def predict(self, data: pandas.DataFrame) -> pandas.Series:
        """Infer the true labels when the model is fitted.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def predict_score(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Return total sum of weights for each label when the model is fitted.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label scores.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """
        ...

    def fit_predict(self, data: pandas.DataFrame) -> pandas.Series:
        """Fit the model and return aggregated results.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def fit_predict_score(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Fit the model and return the total sum of weights for each label.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label scores.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """
        ...

    def __init__(
        self,
        n_iter: int = 10000,
        tol: float = 1e-10,
        random_state: typing.Optional[int] = 0,
        observation_matrix: numpy.ndarray = ...,
        covariation_matrix: numpy.ndarray = ...,
        n_common_tasks: numpy.ndarray = ...,
        n_workers: int = 0,
        n_tasks: int = 0,
        n_labels: int = 0,
        labels_mapping: dict = ...,
        workers_mapping: dict = ...,
        tasks_mapping: dict = ...
    ) -> None:
        """Method generated by attrs for class MMSR.
        """
        ...

    labels_: typing.Optional[pandas.Series]
    n_iter: int
    tol: float
    random_state: typing.Optional[int]
    _observation_matrix: numpy.ndarray
    _covariation_matrix: numpy.ndarray
    _n_common_tasks: numpy.ndarray
    _n_workers: int
    _n_tasks: int
    _n_labels: int
    _labels_mapping: dict
    _workers_mapping: dict
    _tasks_mapping: dict
    skills_: typing.Optional[pandas.Series]
    scores_: typing.Optional[pandas.DataFrame]
    loss_history_: typing.List[float]
