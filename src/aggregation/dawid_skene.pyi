from crowdkit.aggregation.base_aggregator import BaseAggregator
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from typing import ClassVar, Optional


class DawidSkene(BaseAggregator):
    """Dawid-Skene aggregation model
    A. Philip Dawid and Allan M. Skene. 1979.
    Maximum Likelihood Estimation of Observer Error-Rates Using the EM Algorithm.
    Journal of the Royal Statistical Society. Series C (Applied Statistics), Vol. 28, 1 (1979), 20–28.

    https://doi.org/10.2307/2346806
    Attributes:
        probas (typing.ClassVar[typing.Optional[pandas.core.frame.DataFrame]]): Estimated label probabilities
            A frame indexed by `task` and a column for every label id found
            in `data` such that `result.loc[task, label]` is the probability of `task`'s
            true label to be equal to `label`.

        priors (typing.ClassVar[typing.Optional[pandas.core.series.Series]]): A prior label distribution
            A series of labels' probabilities indexed by labels
        task_labels (typing.ClassVar[typing.Optional[pandas.core.frame.DataFrame]]): Estimated labels
            A pandas.DataFrame indexed by `task` with a single column `label` containing
            `tasks`'s most probable label for last fitted data, or None otherwise.

        errors (typing.ClassVar[typing.Optional[pandas.core.frame.DataFrame]]): Performers' error matrices
            A pandas.DataFrame indexed by `performer` and `label` with a column for every
            label_id found in `data` such that `result.loc[performer, observed_label, true_label]`
            is the probability of `performer` producing an `observed_label` given that a task's
            true label is `true_label`"""

    tasks_labels: ClassVar[Optional[DataFrame]]
    probas: ClassVar[Optional[DataFrame]]
    performers_skills: ClassVar[Optional[Series]]
    priors: ClassVar[Optional[Series]]
    task_labels: ClassVar[Optional[DataFrame]]
    errors: ClassVar[Optional[DataFrame]]

    def __init__(self, n_iter: int):
        """Args:
            n_iter: Number of iterations to perform"""
        ...

    @staticmethod
    def _e_step(data: DataFrame, priors: DataFrame, errors: DataFrame) -> DataFrame:
        """Perform E-step of Dawid-Skene algorithm.

        Given performer's answers, labels' prior probabilities and performer's performer's
        errors probabilities matrix estimates tasks' true labels probabilities.
        Args:
            data (DataFrame): Input data
                A pandas.DataFrame containing `task`, `performer` and `label` columns
            priors (DataFrame): Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`.

            errors (DataFrame): Performers' error matrices
                A pandas.DataFrame indexed by `performer` and `label` with a column for every
                label_id found in `data` such that `result.loc[performer, observed_label, true_label]`
                is the probability of `performer` producing an `observed_label` given that a task's
                true label is `true_label`

        Returns:
            DataFrame: Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`."""
        ...

    @staticmethod
    def _m_step(data: DataFrame, probas: DataFrame) -> DataFrame:
        """Perform M-step of Dawid-Skene algorithm.

        Given performers' answers and tasks' true labels probabilities estimates
        performer's errors probabilities matrix.
        Args:
            data (DataFrame): Input data
                A pandas.DataFrame containing `task`, `performer` and `label` columns
            probas (DataFrame): Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`.

        Returns:
            DataFrame: Performers' error matrices
                A pandas.DataFrame indexed by `performer` and `label` with a column for every
                label_id found in `data` such that `result.loc[performer, observed_label, true_label]`
                is the probability of `performer` producing an `observed_label` given that a task's
                true label is `true_label`"""
        ...

    def fit(self, data: DataFrame) -> 'DawidSkene':
        """Args:
            data (DataFrame): Input data
                A pandas.DataFrame containing `task`, `performer` and `label` columns
        Returns:
            DawidSkene: self"""
        ...

    def fit_predict(self, data: DataFrame) -> DataFrame:
        """Args:
            data (DataFrame): Input data
                A pandas.DataFrame containing `task`, `performer` and `label` columns
        Returns:
            DataFrame: Estimated labels
                A pandas.DataFrame indexed by `task` with a single column `label` containing
                `tasks`'s most probable label for last fitted data, or None otherwise."""
        ...

    def fit_predict_proba(self, data: DataFrame) -> DataFrame:
        """Args:
            data (DataFrame): Input data
                A pandas.DataFrame containing `task`, `performer` and `label` columns
        Returns:
            DataFrame: Estimated label probabilities
                A frame indexed by `task` and a column for every label id found
                in `data` such that `result.loc[task, label]` is the probability of `task`'s
                true label to be equal to `label`."""
        ...
