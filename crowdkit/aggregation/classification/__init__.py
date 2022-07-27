__all__ = [
    'DawidSkene',
    'GLAD',
    'GoldMajorityVote',
    'MACE',
    'MajorityVote',
    'MMSR',
    'OneCoinDawidSkene',
    'Wawa',
    'ZeroBasedSkill',
    'BinaryRelevance'
]

from .dawid_skene import DawidSkene, OneCoinDawidSkene
from .glad import GLAD
from .gold_majority_vote import GoldMajorityVote
from .mace import MACE
from .majority_vote import MajorityVote
from .m_msr import MMSR
from .wawa import Wawa
from .zero_based_skill import ZeroBasedSkill
from .binary_relevance import BinaryRelevance
