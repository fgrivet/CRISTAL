"""
Base class for regularization methods.
"""

from abc import ABC, abstractmethod


class BaseRegularizer(ABC):
    """
    Base class for regularization methods.
    """

    @staticmethod
    @abstractmethod
    def compute_value(n: int | float, d: int, C: float | int) -> float:
        """Compute the regularization value.

        Parameters
        ----------
        n : int
            The polynomial basis maximum degree.
        d : int
            The dimension of the input data.
        C : float | int
            A constant used in the regularization computation.

        Returns
        -------
        float
            The computed regularization value.
        """
