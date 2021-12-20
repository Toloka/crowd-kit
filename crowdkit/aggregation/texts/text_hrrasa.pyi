__all__ = [
    'TextHRRASA',
]
import crowdkit.aggregation.base
import crowdkit.aggregation.embeddings.hrrasa
import pandas
import pandas.core.series
import typing


class TextHRRASA(crowdkit.aggregation.base.BaseTextsAggregator):
    """HRRASA on text embeddings.

    Given a sentence encoder, encodes texts provided by performers and runs the HRRASA algorithm for embedding
    aggregation.

    Args:
        encoder: A callable that takes a text and returns a NumPy array containing the corresponding embedding.
        n_iter: A number of HRRASA iterations.
        lambda_emb: A weight of reliability calculated on embeddigs.
        lambda_out: A weight of reliability calculated on outputs.
        alpha: Confidence level of chi-squared distribution quantiles in beta parameter formula.
        calculate_ranks: If true, calculate additional attribute `ranks_`.

    Examples:
        We suggest to use sentence encoders provided by [Sentence Transformers](https://www.sbert.net).
        >>> from crowdkit.datasets import load_dataset
        >>> from crowdkit.aggregation import TextHRRASA
        >>> from sentence_transformers import SentenceTransformer
        >>> encoder = SentenceTransformer('all-mpnet-base-v2')
        >>> hrrasa = TextHRRASA(encoder=encoder.encode)
        >>> df, gt = load_dataset('crowdspeech-test-clean')
        >>> df['text'] = df['text'].apply(lambda s: s.lower())
        >>> result = hrrasa.fit_predict(df)
    """

    def __init__(
        self,
        encoder: typing.Callable,
        n_iter: int = 100,
        lambda_emb: float = ...,
        lambda_out: float = ...,
        alpha: float = ...,
        calculate_ranks: bool = False,
        output_similarity: typing.Callable = crowdkit.aggregation.embeddings.hrrasa.glue_similarity
    ): ...

    def fit_predict_scores(
        self,
        data: pandas.DataFrame,
        true_objects: pandas.core.series.Series = None
    ) -> pandas.DataFrame:
        """Fit the model and return scores.
        Args:
            data (DataFrame): Performers' outputs.
                A pandas.DataFrame containing `task`, `performer` and `output` columns.
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
        true_objects: pandas.core.series.Series = None
    ) -> pandas.core.series.Series:
        """Fit the model and return aggregated texts.
        Args:
            data (DataFrame): Performers' outputs.
                A pandas.DataFrame containing `task`, `performer` and `output` columns.
            true_objects (Series): Tasks' ground truth labels.
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's ground truth label.

        Returns:
            Series: Tasks' texts.
                A pandas.Series indexed by `task` such that `result.loc[task, text]`
                is the task's text.
        """
        ...

    texts_: pandas.core.series.Series
