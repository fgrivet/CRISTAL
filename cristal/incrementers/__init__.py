"""
CRISTAL Incrementers Subpackage \
Contains various incrementer classes for updating moments matrices, including Woodbury, Sherman-Morrison, and inverse methods.
"""

from enum import Enum

from .base import BaseIncrementer
from .inverse import InverseIncrementer
from .sherman import ShermanIncrementer
from .woodbury import WoodburyIncrementer


class IMPLEMENTED_INCREMENTERS(Enum):
    """
    The implemented incrementer classes.
    """

    INVERSE = InverseIncrementer
    SHERMAN = ShermanIncrementer
    WOODBURY = WoodburyIncrementer


__all__ = ["BaseIncrementer", "IMPLEMENTED_INCREMENTERS", "InverseIncrementer", "ShermanIncrementer", "WoodburyIncrementer"]
