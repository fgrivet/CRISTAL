"""
CRISTAL Regularizers Subpackage \
Contains various regularization classes, including Vu, Vu_C, Constant and Comb regularizers.
"""

from enum import Enum

from .base import BaseRegularizer
from .comb import CombRegularizer
from .constant import ConstantRegularizer
from .vu import VuCRegularizer, VuRegularizer


class IMPLEMENTED_REGULARIZERS(Enum):
    """
    The implemented regularization classes.
    """

    COMB = CombRegularizer
    CONSTANT = ConstantRegularizer
    VU = VuRegularizer
    VU_C = VuCRegularizer


__all__ = [
    "BaseRegularizer",
    "CombRegularizer",
    "ConstantRegularizer",
    "IMPLEMENTED_REGULARIZERS",
    "VuRegularizer",
    "VuCRegularizer",
]
