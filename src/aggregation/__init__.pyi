__all__ = [
    'base',
    'BradleyTerry',
    'ClosestToAverage',
    'DawidSkene',
    'GLAD',
    'GoldMajorityVote',
    'HRRASA',
    'MMSR',
    'MajorityVote',
    'NoisyBradleyTerry',
    'RASA',
    'ROVER',
    'SegmentationEM',
    'SegmentationMajorityVote',
    'SegmentationRASA',
    'TextHRRASA',
    'TextRASA',
    'TextSummarization',
    'Wawa',
    'ZeroBasedSkill',
]
from crowdkit.aggregation import base
from crowdkit.aggregation.classification.dawid_skene import DawidSkene
from crowdkit.aggregation.classification.glad import GLAD
from crowdkit.aggregation.classification.gold_majority_vote import GoldMajorityVote
from crowdkit.aggregation.classification.m_msr import MMSR
from crowdkit.aggregation.classification.majority_vote import MajorityVote
from crowdkit.aggregation.classification.wawa import Wawa
from crowdkit.aggregation.classification.zero_based_skill import ZeroBasedSkill
from crowdkit.aggregation.embeddings.closest_to_average import ClosestToAverage
from crowdkit.aggregation.embeddings.hrrasa import HRRASA
from crowdkit.aggregation.embeddings.rasa import RASA
from crowdkit.aggregation.image_segmentation.segmentation_em import SegmentationEM
from crowdkit.aggregation.image_segmentation.segmentation_majority_vote import SegmentationMajorityVote
from crowdkit.aggregation.image_segmentation.segmentation_rasa import SegmentationRASA
from crowdkit.aggregation.pairwise.bradley_terry import BradleyTerry
from crowdkit.aggregation.pairwise.noisy_bt import NoisyBradleyTerry
from crowdkit.aggregation.texts.rover import ROVER
from crowdkit.aggregation.texts.text_hrrasa import TextHRRASA
from crowdkit.aggregation.texts.text_rasa import TextRASA
from crowdkit.aggregation.texts.text_summarization import TextSummarization
