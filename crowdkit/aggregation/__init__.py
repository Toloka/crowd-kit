from typing import cast

from . import base
from .classification import (
    DawidSkene,
    GLAD,
    GoldMajorityVote,
    KOS,
    MACE,
    MajorityVote,
    MMSR,
    OneCoinDawidSkene,
    Wawa,
    ZeroBasedSkill,
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
    'KOS',
    'MACE',
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


def is_arcadia() -> bool:
    try:
        import __res
        return cast(bool, __res == __res)
    except ImportError:
        return False


if not is_arcadia():
    from .texts.text_summarization import TextSummarization
