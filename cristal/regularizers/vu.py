"""
Class for Vu regularization method. with (:class:`VuCRegularizer`) and without (:class:`VuRegularizer`) constant factor C. \
See https://arxiv.org/abs/1910.14458 for more details.
"""

from .base import BaseRegularizer


class VuRegularizer(BaseRegularizer):
    """
    Regularizer for the Vu regularization method : :math:`reg = n^{3d/2}`. \
    See https://arxiv.org/abs/1910.14458 for more details.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the Vu regularization factor."""
        return n ** (3 * d / 2)


class VuCRegularizer(BaseRegularizer):
    """
    Regularizer for the Vu_C regularization method : :math:`reg = \\frac{n^{3d/2}}{C}`. \
    See https://arxiv.org/abs/1910.14458 for more details.
    """

    @staticmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the Vu_C regularization factor."""
        return (n ** (3 * d / 2)) / C
