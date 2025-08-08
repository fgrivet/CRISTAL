"""
Class for combinatorial regularization method.
"""

from math import comb
from .base import BaseRegularizer


class CombRegularizer(BaseRegularizer):
    """
    Regularizer for the combinatorial regularization method : :math:`reg = \\begin{pmatrix} d + n \\\\ d \\end{pmatrix}`.
    """

    @staticmethod
    def compute_value(n: int | float, d: int, C: float | int) -> float:
        """Compute the combinatorial regularization factor."""
        if isinstance(n, float):
            n = int(n)
        return comb(d + n, d)
