__all__ = [
    'SegmentationMajorityVote',
]
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from toloka.client.base_aggregator import BaseAggregator

class SegmentationMajorityVote(BaseAggregator):
    """Majority Vote - chooses a pixel if more than half of performers voted

    Doris Jung-Lin Lee. 2018.
    Quality Evaluation Methods for Crowdsourced Image Segmentation
    http://ilpubs.stanford.edu:8090/1161/1/main.pdf
    Attributes:
        segmentations_ (ndarray): Tasks' segmentations
            A pandas.Series indexed by `task` such that `labels.loc[task]`
            is the tasks's aggregated segmentation.
    """

    def fit(
        self,
        data: DataFrame,
        skills: Series = None
    ) -> 'SegmentationMajorityVote':
        """Args:
            data (DataFrame): Performers' segmentations
                A pandas.DataFrame containing `performer`, `task` and `segmentation` columns'.

            skills (Series): Performers' skills
                A pandas.Series index by performers and holding corresponding performer's skill
        Returns:
            SegmentationMajorityVote: self
        """
        ...

    def fit_predict(
        self,
        data: DataFrame,
        skills: Series = None
    ) -> ndarray:
        """Args:
            data (DataFrame): Performers' segmentations
                A pandas.DataFrame containing `performer`, `task` and `segmentation` columns'.

            skills (Series): Performers' skills
                A pandas.Series index by performers and holding corresponding performer's skill
        Returns:
            ndarray: Tasks' segmentations
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                is the tasks's aggregated segmentation.
        """
        ...

    segmentations_: ndarray
