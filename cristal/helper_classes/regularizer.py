"""
Regularization methods for score functions.
"""

from enum import Enum
from math import comb

from cristal.helper_classes.base import BaseRegularizer

__all__ = [
    "IMPLEMENTED_REGULARIZATION_OPTIONS",
    "VuRegularizer",
    "VuCRegularizer",
    "CombRegularizer",
    "ConstantRegularizer",
]


class VuRegularizer(BaseRegularizer):
    """
    Regularizer for the Vu regularization method : :math:`reg = n^{3d/2}`.

    See https://arxiv.org/abs/1910.14458 for more details.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the Vu regularization factor."""
        return n ** (3 * d / 2)


class VuCRegularizer(BaseRegularizer):
    """
    Regularizer for the Vu_C regularization method : :math:`reg = \\frac{n^{3d/2}}{C}`.

    See https://arxiv.org/abs/1910.14458 for more details.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the Vu_C regularization factor."""
        return (n ** (3 * d / 2)) / C


class CombRegularizer(BaseRegularizer):
    """
    Regularizer for the combinatorial regularization method : :math:`reg = \\begin{pmatrix} d + n \\\\ d \\end{pmatrix}`.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the combinatorial regularization factor."""
        if isinstance(n, float):
            n = int(n)
        return comb(d + n, d)


class ConstantRegularizer(BaseRegularizer):
    """
    Regularizer that applies a constant factor C : : :math:`reg = C`.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the constant regularization factor."""
        return C


class IMPLEMENTED_REGULARIZATION_OPTIONS(Enum):
    """
    The implemented regularization classes.
    """

    VU = VuRegularizer
    VU_C = VuCRegularizer
    COMB = CombRegularizer
    CONSTANT = ConstantRegularizer
