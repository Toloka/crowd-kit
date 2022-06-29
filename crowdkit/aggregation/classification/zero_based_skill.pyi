__all__ = [
    'ZeroBasedSkill',
]
import crowdkit.aggregation.base
import pandas
import typing


class ZeroBasedSkill(crowdkit.aggregation.base.BaseClassificationAggregator):
    """The Zero-Based Skill aggregation model.

    Performs weighted majority voting on tasks. After processing a pool of tasks,
    re-estimates workers' skills through a gradient descend step of optimization
    of the mean squared error of current skills and the fraction of responses that
    are equal to the aggregated labels.

    Repeats this process until labels do not change or the number of iterations exceeds.

    It's necessary that all workers in a dataset that send to 'predict' existed in answers
    the dataset that was sent to 'fit'.

    Args:
        n_iter: A number of iterations to perform.
        lr_init: A starting learning rate.
        lr_steps_to_reduce: A number of steps necessary to decrease the learning rate.
        lr_reduce_factor: A factor that the learning rate will be multiplied by every `lr_steps_to_reduce` steps.
        eps: A convergence threshold.

    Examples:
        >>> from crowdkit.aggregation import ZeroBasedSkill
        >>> from crowdkit.datasets import load_dataset
        >>> df, gt = load_dataset('relevance-2')
        >>> result = ZeroBasedSkill().fit_predict(df)
    """

    def fit(self, data: pandas.DataFrame) -> 'ZeroBasedSkill':
        """Fit the model.
        Args:
            data (DataFrame): Workers' labeling results.
                A pandas.DataFrame containing `task`, `worker` and `label` columns.
        Returns:
            ZeroBasedSkill: self.
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

    def fit_predict_proba(self, data: pandas.DataFrame) -> pandas.DataFrame:
        """Fit the model and return probability distributions on labels for each task.
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

    def __init__(
        self,
        n_iter: int = 100,
        lr_init: float = 1.0,
        lr_steps_to_reduce: int = 20,
        lr_reduce_factor: float = 0.5,
        eps: float = 1e-05
    ) -> None:
        """Method generated by attrs for class ZeroBasedSkill.
        """
        ...

    labels_: typing.Optional[pandas.Series]
    n_iter: int
    lr_init: float
    lr_steps_to_reduce: int
    lr_reduce_factor: float
    eps: float
    skills_: typing.Optional[pandas.Series]
    probas_: typing.Optional[pandas.DataFrame]
