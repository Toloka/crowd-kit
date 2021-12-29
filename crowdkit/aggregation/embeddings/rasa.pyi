__all__ = [
    'RASA',
]
import crowdkit.aggregation.base
import pandas
import typing


class RASA(crowdkit.aggregation.base.BaseEmbeddingsAggregator):
    """Reliability Aware Sequence Aggregation.

    RASA estimates *global* performers' reliabilities $\beta$ that are initialized by ones.

    Next, the algorithm iteratively performs two steps:
    1. For each task, estimate the aggregated embedding: $\hat{e}_i = \frac{\sum_k
    \beta_k e_i^k}{\sum_k \beta_k}$
    2. For each performer, estimate the global reliability: $\beta_k = \frac{\chi^2_{(\alpha/2,
    |\mathcal{V}_k|)}}{\sum_i\left(\|e_i^k - \hat{e}_i\|^2\right)}$, where $\mathcal{V}_k$
    is a set of tasks completed by the performer $k$

    Finally, the aggregated result is the output which embedding is
    the closest one to the $\hat{e}_i$.

    Jiyi Li.
    A Dataset of Crowdsourced Word Sequences: Collections and Answer Aggregation for Ground Truth Creation.
    *Proceedings of the First Workshop on Aggregating and Analysing Crowdsourced Annotations for NLP*,
    pages 24–28 Hong Kong, China, November 3, 2019.
    http://doi.org/10.18653/v1/D19-5904

    Args:
        n_iter: A number of iterations.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.

    Examples:
        >>> import numpy as np
        >>> import pandas as pd
        >>> from crowdkit.aggregation import RASA
        >>> df = pd.DataFrame(
        >>>     [
        >>>         ['t1', 'p1', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p2', 'a', np.array([1.0, 0.0])],
        >>>         ['t1', 'p3', 'b', np.array([0.0, 1.0])]
        >>>     ],
        >>>     columns=['task', 'performer', 'output', 'embedding']
        >>> )
        >>> result = RASA().fit_predict(df)
    Attributes:
        embeddings_and_outputs_ (DataFrame): Tasks' embeddings and outputs.
            A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
    """

    def fit(
        self,
        data: pandas.DataFrame,
        true_embeddings: pandas.Series = None
    ) -> 'RASA':
        """Fit the model.
        Args:
            data (DataFrame): Performers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.
        Returns:
            RASA: self.
        """
        ...

    def fit_predict_scores(
        self,
        data: pandas.DataFrame,
        true_embeddings: pandas.Series = None
    ) -> pandas.DataFrame:
        """Fit the model and return scores.
        Args:
            data (DataFrame): Performers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.
        Returns:
            DataFrame: Tasks' label scores.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """
        ...

    def fit_predict(
        self,
        data: pandas.DataFrame,
        true_embeddings: pandas.Series = None
    ) -> pandas.DataFrame:
        """Fit the model and return aggregated outputs.
        Args:
            data (DataFrame): Performers' outputs with their embeddings.
                A pandas.DataFrame containing `task`, `performer`, `output` and `embedding` columns.
            true_embeddings (Series): Tasks' embeddings.
                A pandas.Series indexed by `task` and holding corresponding embeddings.
        Returns:
            DataFrame: Tasks' embeddings and outputs.
                A pandas.DataFrame indexed by `task` with `embedding` and `output` columns.
        """
        ...

    def __init__(
        self,
        n_iter: int = 100,
        tol: float = 1e-09,
        alpha: float = 0.05
    ) -> None:
        """Method generated by attrs for class RASA.
        """
        ...

    embeddings_and_outputs_: pandas.DataFrame
    n_iter: int
    tol: float
    alpha: float
    loss_history_: typing.List[float]
