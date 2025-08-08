"""
Class for constant regularization method.
"""

from .base import BaseRegularizer


class ConstantRegularizer(BaseRegularizer):
    """
    Regularizer that applies a constant factor C : : :math:`reg = C`.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the constant regularization factor."""
        return C
