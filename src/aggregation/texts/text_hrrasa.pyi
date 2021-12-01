__all__ = [
    'TextHRRASA',
]
import crowdkit.aggregation.base
import pandas.core.frame
import pandas.core.series
import typing


class TextHRRASA(crowdkit.aggregation.base.BaseTextsAggregator):
    def __init__(
        self,
        encoder: typing.Callable,
        n_iter: int = 100,
        lambda_emb: float = ...,
        lambda_out: float = ...,
        alpha: float = ...,
        calculate_ranks: bool = False,
        output_similarity: typing.Callable = ...
    ): ...

    def fit_predict_scores(
        self,
        data: pandas.core.frame.DataFrame,
        true_objects: pandas.core.series.Series = None
    ) -> pandas.core.frame.DataFrame:
        """Args:
            data (DataFrame): Performers' outputs
                A pandas.DataFrame containing `task`, `performer` and `output` columns.
            true_objects (Series): Tasks' ground truth labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            DataFrame: Tasks' label scores
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """
        ...

    def fit_predict(
        self,
        data: pandas.core.frame.DataFrame,
        true_objects: pandas.core.series.Series = None
    ) -> pandas.core.series.Series:
        """Args:
            data (DataFrame): Performers' outputs
                A pandas.DataFrame containing `task`, `performer` and `output` columns.
            true_objects (Series): Tasks' ground truth labels
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: Tasks' texts
                A pandas.Series indexed by `task` such that `result.loc[task, text]`
                is the task's text.
        """
        ...

    texts_: pandas.core.series.Series