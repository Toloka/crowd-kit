__all__ = [
    'GoldMajorityVote',
]
import crowdkit.aggregation.base
import pandas
import typing


class GoldMajorityVote(crowdkit.aggregation.base.BaseClassificationAggregator):
    """Majority Vote when exist golden dataset (ground truth) for some tasks.

    Calculates the probability of a correct label for each worker based on the golden set.
    Based on this, for each task, calculates the sum of the probabilities of each label.
    The correct label is the one where the sum of the probabilities is greater.

    For Example: You have 10k tasks completed by 3k different workers. And you have 100 tasks where you already
    know ground truth labels. First you can call `fit` to calc percents of correct labels for each workers.
    And then call `predict` to calculate labels for you 10k tasks.

    It's necessary that:
    1. All workers must done at least one task from golden dataset.
    2. All workers in dataset that send to `predict`, existed in answers dataset that was sent to `fit`.

    Examples:
        >>> import pandas as pd
        >>> from crowdkit.aggregation import GoldMajorityVote
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', 0],
        >>>         ['t1', 'p2', 0],
        >>>         ['t1', 'p3', 1],
        >>>         ['t2', 'p1', 1],
        >>>         ['t2', 'p2', 0],
        >>>         ['t2', 'p3', 1],
        >>>     ],
        >>>     columns=['task', 'worker', 'label']
        >>> )
        >>> true_labels = pd.Series({'t1': 0})
        >>> gold_mv = GoldMajorityVote()
        >>> result = gold_mv.fit_predict(df, true_labels)
    Attributes:
        labels_ (typing.Optional[pandas.core.series.Series]): Tasks' labels.
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's most likely true label.

        skills_ (typing.Optional[pandas.core.series.Series]): workers' skills.
            A pandas.Series index by workers and holding corresponding worker's skill
        probas_ (typing.Optional[pandas.core.frame.DataFrame]): Tasks' label probability distributions.
            A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
            is the probability of `task`'s true label to be equal to `label`. Each
            probability is between 0 and 1, all task's probabilities should sum up to 1
    """

    def fit(
        self,
        data: pandas.DataFrame,
        true_labels: pandas.Series
    ) -> 'GoldMajorityVote':
        """Estimate the workers' skills.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            true_labels (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            GoldMajorityVote: self.
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

    def predict_proba(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Return probability distributions on labels for each task when the model is fitted.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def fit_predict(
        self,
        data: pandas.DataFrame,
        true_labels: pandas.Series
    ) -> pandas.Series:
        """Fit the model and return aggregated results.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            true_labels (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: Tasks' labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's most likely true label.
        """
        ...

    def fit_predict_proba(
        self,
        data: pandas.DataFrame,
        true_labels: pandas.Series
    ) -> pandas.DataFrame:
        """Fit the model and return probability distributions on labels for each task.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
            true_labels (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            DataFrame: Tasks' label probability distributions.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the probability of `task`'s true label to be equal to `label`. Each
                probability is between 0 and 1, all task's probabilities should sum up to 1
        """
        ...

    def __init__(self) -> None:
        """Method generated by attrs for class GoldMajorityVote.
        """
        ...

    labels_: typing.Optional[pandas.Series]
    skills_: typing.Optional[pandas.Series]
    probas_: typing.Optional[pandas.DataFrame]
