from . import base
from .classification import (
    DawidSkene,
    OneCoinDawidSkene,
    GLAD,
    GoldMajorityVote,
    MMSR,
    MajorityVote,
    Wawa,
    ZeroBasedSkill,
    MultiBinary
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
from .pairwise import (
    BradleyTerry,
    NoisyBradleyTerry
)
from .texts import (
    TextRASA,
    TextHRRASA,
    ROVER
)

__all__ = [
    'base',

    'BradleyTerry',
    'ClosestToAverage',
    'DawidSkene',
    'OneCoinDawidSkene',
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
    'MultiBinary'
]


def is_arcadia():
    try:
        import __res
        return __res == __res
    except ImportError:
        return False


if not is_arcadia():
    from .texts.text_summarization import TextSummarization
