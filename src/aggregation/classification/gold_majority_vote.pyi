__all__ = [
    'GoldMajorityVote',
]
import crowdkit.aggregation.base
import pandas.core.frame
import pandas.core.series
import typing


class GoldMajorityVote(crowdkit.aggregation.base.BaseClassificationAggregator):
    """Majority Vote when exist golden dataset (ground truth) for some tasks

    Calculates the probability of a correct label for each performer based on the golden set
    Based on this, for each task, calculates the sum of the probabilities of each label
    The correct label is the one where the sum of the probabilities is greater

    For Example: You have 10k tasks completed by 3k different performers. And you have 100 tasks where you already
    know ground truth labels. First you can call 'fit' to calc percents of correct labels for each performers.
    And then call 'predict' to calculate labels for you 10k tasks.

    It's necessary that:
    1. All performers must done at least one task from golden dataset.
    2. All performers in dataset that send to 'predict', existed in answers dataset that was sent to 'fit'
    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        skills_ (typing.Optional[pandas.core.series.Series]): Performers' skills
            A pandas.Series index by performers and holding corresponding performer's skill
        probas_ (typing.Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """

    def fit(
        self,
        data: pandas.core.frame.DataFrame,
        true_labels: pandas.core.series.Series
    ) -> 'GoldMajorityVote':
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
            true_labels (Series): Tasks' ground truth labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            GoldMajorityVote: self
        """
        ...

    def predict(self, data: pandas.core.frame.DataFrame) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            Series: Tasks' labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def predict_proba(self, data: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def fit_predict(
        self,
        data: pandas.core.frame.DataFrame,
        true_labels: pandas.core.series.Series
    ) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
            true_labels (Series): Tasks' ground truth labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: Tasks' labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def fit_predict_proba(
        self,
        data: pandas.core.frame.DataFrame,
        true_labels: pandas.core.series.Series
    ) -> pandas.core.frame.DataFrame:
        """Args:
            data (DataFrame): Performers' labeling results
                A pandas.DataFrame containing `task`, `performer` and `label` columns.
            true_labels (Series): Tasks' ground truth labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            DataFrame: Tasks' label probability distributions
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def __init__(self) -> None:
        """Method generated by attrs for class GoldMajorityVote.
        """
        ...

    labels_: typing.Optional[pandas.core.series.Series]
    skills_: typing.Optional[pandas.core.series.Series]
    probas_: typing.Optional[pandas.core.frame.DataFrame]
