"""
Base class for matrix inversion methods.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseInverter(ABC):
    """
    Base class for matrix inversion methods.
    """

    @staticmethod
    @abstractmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Invert the given matrix.

        Parameters
        ----------
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The inverse of the input matrix.
        """
