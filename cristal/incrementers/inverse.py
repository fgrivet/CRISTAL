"""
Class for incrementing moments matrix by updating the moments matrix and then computing the inverse.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..inverters.base import BaseInverter
from ..polynomials.base import MultivariatePolynomialBasis
from .base import BaseIncrementer

if TYPE_CHECKING:
    from ..moments_matrix.moments_matrix import MomentsMatrix


class InverseIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the inverse method.
    This method updates the moments matrix and then computes the inverse of the updated moments matrix directly.
    """

    update_moments_matrix = True

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_class: type[BaseInverter], sym: bool = True):
        """Increment the moments matrix using the inverse method.

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
            Not used in this implementation, by default True
        """
        N = x.shape[0]
        # Revert the mean of the moments matrix
        moments_matrix = n * mm.moments_matrix  # type: ignore

        # Compute the design matrix to add
        V = MultivariatePolynomialBasis.make_design_matrix(x, mm.multidegree_combinations, mm.polynomial_class)  # type: ignore

        # Update the moments matrix
        moments_matrix += V.T @ V
        # Compute the mean of the updated moments matrix
        moments_matrix /= n + N
        mm.moments_matrix = moments_matrix

        # Compute the inverse of the updated moments matrix
        mm.inverse_moments_matrix = inv_class.invert(moments_matrix)
