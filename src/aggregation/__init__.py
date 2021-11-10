from . import base
from .classification import (
    DawidSkene,
    GLAD,
    GoldMajorityVote,
    MMSR,
    MajorityVote,
    Wawa,
    ZeroBasedSkill
)
from .embeddings import (
    ClosestToAverage,
    HRRASA,
    RASA,
)
from .image_segmentation import (
    SegmentationEM,
    SegmentationRASA,
    SegmentationMajorityVote
)
from .texts import (
    TextRASA,
    TextHRRASA,
    ROVER
)
from .pairwise import (
    BradleyTerry,
    NoisyBradleyTerry
)

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


def is_arcadia():
    try:
        import __res
        return __res == __res
    except ImportError:
        return False


if not is_arcadia():
    from .texts.text_summarization import TextSummarization
