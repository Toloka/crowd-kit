__all__ = [
    'TextRASA',
]
import crowdkit.aggregation.base
import pandas
import typing


class TextRASA(crowdkit.aggregation.base.BaseTextsAggregator):
    """RASA on text embeddings.

    Given a sentence encoder, encodes texts provided by workers and runs the RASA algorithm for embedding
    aggregation.

    Args:
        encoder: A callable that takes a text and returns a NumPy array containing the corresponding embedding.
        n_iter: A number of RASA iterations.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.

    Examples:
        We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
        >>> from crowdkit.datasets import load_dataset
        >>> from crowdkit.aggregation import TextRASA
        >>> from sentence_transformers import SentenceTransformer
        >>> encoder = SentenceTransformer('all-mpnet-base-v2')
        >>> hrrasa = TextRASA(encoder=encoder.encode)
        >>> df, gt = load_dataset('crowdspeech-test-clean')
        >>> df['text'] = df['text'].apply(lambda s: s.lower())
        >>> result = hrrasa.fit_predict(df)
    """

    def __init__(
        self,
        encoder: typing.Callable,
        n_iter: int = 100,
        tol: float = 1e-05,
        alpha: float = 0.05
    ): ...

    def fit(
        self,
        data: pandas.DataFrame,
        true_objects: pandas.Series = None
    ) -> 'TextRASA':
        """Fit the model.
        Args:
            data (DataFrame): Workers' outputs.
                A pandas.DataFrame containing `task`, `worker` and `output` columns.
            true_objects (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            TextRASA: self.
        """
        ...

    def fit_predict_scores(
        self,
        data: pandas.DataFrame,
        true_objects: pandas.Series = None
    ) -> pandas.DataFrame:
        """Fit the model and return scores.
        Args:
            data (DataFrame): Workers' outputs.
                A pandas.DataFrame containing `task`, `worker` and `output` columns.
            true_objects (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            DataFrame: Tasks' label scores.
                A pandas.DataFrame indexed by `task` such that `result.loc[task, label]`
                is the score of `label` for `task`.
        """
        ...

    def fit_predict(
        self,
        data: pandas.DataFrame,
        true_objects: pandas.Series = None
    ) -> pandas.Series:
        """Fit the model and return aggregated texts.
        Args:
            data (DataFrame): Workers' outputs.
                A pandas.DataFrame containing `task`, `worker` and `output` columns.
            true_objects (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: Tasks' texts.
                A pandas.Series indexed by `task` such that `result.loc[task, text]`
                is the task's text.
        """
        ...

    texts_: pandas.Series