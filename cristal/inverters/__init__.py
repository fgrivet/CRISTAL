"""
CRISTAL Inverters Subpackage \
Contains various inverter classes for matrix inversion, including standard, pseudo, and Cholesky-based methods.
"""

from enum import Enum

from .base import BaseInverter
from .lapack import FPDInverter
from .scipy import InvInverter, PDInverter, PseudoInverter


class IMPLEMENTED_INVERTERS(Enum):
    """
    The implemented inverter classes.
    """

    FPD = FPDInverter
    INV = InvInverter
    PSEUDO = PseudoInverter
    PD = PDInverter


__all__ = [
    "BaseInverter",
    "FPDInverter",
    "IMPLEMENTED_INVERTERS",
    "InvInverter",
    "PDInverter",
    "PseudoInverter",
]
