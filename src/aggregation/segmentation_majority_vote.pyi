from crowdkit.aggregation.base_aggregator import BaseAggregator
from numpy import ndarray
from pandas.core.frame import DataFrame
from pandas.core.series import Series


class SegmentationMajorityVote(BaseAggregator):
    """Majority Vote - chooses a pixel if more than half of performers voted

    Doris Jung-Lin Lee. 2018.
    Quality Evaluation Methods for Crowdsourced Image Segmentation
    http://ilpubs.stanford.edu:8090/1161/1/main.pdf
    Attributes:
        segmentations_ (ndarray): Tasks' segmentations
            A pandas.Series indexed by `task` such that `labels.loc[task]`
                    is the tasks's aggregated segmentation."""

    @staticmethod
    def _segmentation_majority_vote(segmentations: Series) -> ndarray:
        """Calculates pixel majority vote for single segmentation
        Args:
            segmentations (Series): Single task segmentations
                A pandas.Series containing segmentations for a single task.

        Returns:
            ndarray: Segmentation
                A numpy 2d ndarray, which is a bitmask of segmentation."""
        ...

    def fit(self, data: DataFrame) -> 'SegmentationMajorityVote':
        """Args:
            data (DataFrame): Performers' outputs with images
                A pandas.DataFrame containing `performer`, `task`, `image` and `output` columns'.
                For each row `image` must be ndarray.

        Returns:
            SegmentationMajorityVote: self"""
        ...

    def fit_predict(self, data: DataFrame) -> ndarray:
        """Args:
            data (DataFrame): Performers' outputs with images
                A pandas.DataFrame containing `performer`, `task`, `image` and `output` columns'.
                For each row `image` must be ndarray.

        Returns:
            ndarray: Tasks' segmentations
                A pandas.Series indexed by `task` such that `labels.loc[task]`
                        is the tasks's aggregated segmentation."""
        ...

    segmentations_: ndarray
