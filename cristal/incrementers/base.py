"""
Base class for moments matrix incrementers.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from ..inverters.base import BaseInverter

if TYPE_CHECKING:
    from ..moments_matrix.moments_matrix import MomentsMatrix


class BaseIncrementer(ABC):
    """
    Base class for moments matrix incrementers.
    """

    update_moments_matrix: bool = False  #: Whether the method updates the moments matrix during incrementing

    @staticmethod
    @abstractmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_class: type[BaseInverter], sym: bool = True):
        """Increment the inverse moments matrix (and possibly the moments matrix).

        Parameters
        ----------
        mm : MomentsMatrix
            The moments matrix to increment.
        x : np.ndarray (N', d)
            The input data.
        n : int
            The number of points integrated in the moments matrix.
        inv_class : type[BaseInverter]
            The inversion class to use.
        sym : bool, optional
            Whether to consider the matrix as symmetric, by default True
        """
