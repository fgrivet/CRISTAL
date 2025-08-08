"""
CRISTAL Polynomials Subpackage \
Contains various polynomial classes, including Monomials, Chebyshev and Legendre polynomials.
"""

from enum import Enum

from .base import BasePolynomialFamily, MultivariatePolynomialBasis
from .chebyshev import ChebyshevT1Family, ChebyshevT2Family, ChebyshevUFamily
from .legendre import LegendreFamily
from .monomials import MonomialsFamily


class IMPLEMENTED_POLYNOMIALS(Enum):
    """
    The implemented polynomial classes.
    """

    MONOMIALS = MonomialsFamily
    CHEBYSHEV_T1 = ChebyshevT1Family
    CHEBYSHEV_T2 = ChebyshevT2Family
    CHEBYSHEV_U = ChebyshevUFamily
    LEGENDRE = LegendreFamily


__all__ = [
    "BasePolynomialFamily",
    "ChebyshevT1Family",
    "ChebyshevT2Family",
    "ChebyshevUFamily",
    "IMPLEMENTED_POLYNOMIALS",
    "LegendreFamily",
    "MonomialsFamily",
    "MultivariatePolynomialBasis",
]
