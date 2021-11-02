__all__ = [
    'SegmentationEM',
    'SegmentationRASA',
    'SegmentationMajorityVote',
]
from crowdkit.aggregation.image_segmentation.segmentation_em import SegmentationEM
from crowdkit.aggregation.image_segmentation.segmentation_majority_vote import SegmentationMajorityVote
from crowdkit.aggregation.image_segmentation.segmentation_rasa import SegmentationRASA
