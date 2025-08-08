"""
Class for incrementing moments matrix using the Sherman-Morrison formula iteratively.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..inverters.base import BaseInverter
from .base import BaseIncrementer

if TYPE_CHECKING:
    from ..moments_matrix.moments_matrix import MomentsMatrix


class ShermanIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the Sherman-Morrison formula.
    This method updates the inverse moments matrix iteratively and not modifies the moments matrix.
    """

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_class: type[BaseInverter] | None = None, sym: bool = True):
        """Increment the moments matrix using the Sherman-Morrison formula iteratively.

        Sherman-Morrison formula:

        .. math::
            (A + uv^T)^{-1} = A^{-1} - \\frac{(A^{-1} u v^T A^{-1})}{(1 + v^T A^{-1} u)}

        .. math::
            \\text{Here } u = v^T

        Parameters
        ----------
        mm : MomentsMatrix
            The moments matrix to increment.
        x : np.ndarray (N', d)
            The input data.
        n : int
            The number of points integrated in the moments matrix.
        inv_class : type[BaseInverter] | None, optional
            Not used in this implementation, by default None
        sym : bool, optional
            Whether to consider the matrix as symmetric, by default True
        """
        # Revert the mean of the moments matrix
        inv_moments_matrix = mm.inverse_moments_matrix / n  # type: ignore

        # Apply the Sherman-Morrison formula iteratively
        for xx in x:
            # Compute the vector v for the current point
            v = mm.polynomial_class.func(xx, mm.multidegree_combinations)  # type: ignore
            # Compute the left-hand side of the Sherman-Morrison formula
            left = inv_moments_matrix @ v
            # Compute the denominator
            denom = v.T @ left
            # Divide the left-hand side by (1 + denom)
            # Reduce the division cost from O(N^2) to O(N) by using the fact that left is a vector and numerator is a matrix
            left_div = left / (1 + denom)
            if sym is True:
                # If the matrix is symmetric, we can use the fact that the right-hand side of the numerator is the transpose of its left-hand side
                inv_moments_matrix -= left_div @ left.T
            else:
                # Otherwise, we need to compute the right-hand side of the numerator
                droite = v.T @ inv_moments_matrix
                # And multiply the left-hand side already divided by the right-hand side
                inv_moments_matrix -= left_div @ droite
        # Compute the mean of the updated inverse moments matrix
        mm.inverse_moments_matrix = (n + x.shape[0]) * inv_moments_matrix
