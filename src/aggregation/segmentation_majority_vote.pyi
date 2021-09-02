__all__ = [
    'SegmentationMajorityVote',
]
import crowdkit.aggregation.base_aggregator
import numpy
import pandas.core.frame
import pandas.core.series


class SegmentationMajorityVote(crowdkit.aggregation.base_aggregator.BaseAggregator):
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
        data: pandas.core.frame.DataFrame,
        skills: pandas.core.series.Series = None
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
        data: pandas.core.frame.DataFrame,
        skills: pandas.core.series.Series = None
    ) -> numpy.ndarray:
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

    segmentations_: numpy.ndarray
