from .aggregation.classification import (
    DawidSkene,
    OneCoinDawidSkene,
    GLAD,
    GoldMajorityVote,
    KOS,
    MMSR,
    MACE,
    MajorityVote,
    Wawa,
    ZeroBasedSkill
)
from .aggregation.embeddings import ClosestToAverage, HRRASA, RASA
from .aggregation.image_segmentation import (
    SegmentationEM,
    SegmentationMajorityVote,
    SegmentationRASA
)
from .aggregation.multilabel import BinaryRelevance
from .aggregation.pairwise import BradleyTerry, NoisyBradleyTerry
from .aggregation.texts import ROVER, TextHRRASA, TextRASA
from .aggregation.utils import (
    get_accuracy,
    get_most_probable_labels,
    manage_data,
    normalize_rows,
    add_skills_to_data
)

from .datasets import get_datasets_list, load_dataset
from .learning import (
    CoNAL,
    CrowdLayer,
    TextSummarization
)

from .metrics.data import (
    alpha_krippendorff,
    consistency,
    uncertainty
)
from .metrics.workers import accuracy_on_aggregates

from .postprocessing import entropy_threshold

__all__ = [
    'DawidSkene',
    'OneCoinDawidSkene',
    'GLAD',
    'GoldMajorityVote',
    'KOS',
    'MMSR',
    'MACE',
    'MajorityVote',
    'Wawa',
    'ZeroBasedSkill',
    'ClosestToAverage',
    'HRRASA',
    'RASA',
    'SegmentationEM',
    'SegmentationMajorityVote',
    'SegmentationRASA',
    'BinaryRelevance',
    'BradleyTerry',
    'NoisyBradleyTerry',
    'ROVER',
    'TextHRRASA',
    'TextRASA',
    'get_accuracy',
    'get_most_probable_labels',
    'manage_data',
    'normalize_rows',
    'add_skills_to_data',
    'get_datasets_list',
    'load_dataset',
    'CoNAL',
    'CrowdLayer',
    'TextSummarization',
    'alpha_krippendorff',
    'consistency',
    'uncertainty',
    'accuracy_on_aggregates',
    'entropy_threshold'
    # Add other functions/classes as needed
]
