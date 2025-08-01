"""
Incrementers for moments matrix using various methods.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from cristal.helper_classes.base import BaseIncrementer
from cristal.helper_classes.polynomial_basis import PolynomialsBasisGenerator

if TYPE_CHECKING:
    from .moment_matrix import MomentsMatrix

__all__ = [
    "IMPLEMENTED_INCREMENTATERS_OPTIONS",
    "InverseIncrementer",
    "ShermanIncrementer",
    "WoodburyIncrementer",
]


class InverseIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the inverse method.
    This method updates the moments matrix and then computes the inverse of the updated moments matrix directly.
    """

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_opt: Callable[[np.ndarray], np.ndarray], sym: bool = True):
        """Increment the moments matrix using the inverse method.

        Parameters
        ----------
        mm : MomentsMatrix
            The moments matrix to increment.
        x : np.ndarray
            The input data.
        n : int
            The number of points integrated in the moments matrix.
        inv_opt : Callable[[np.ndarray], np.ndarray]
            The inversion method to use.
        sym : bool, optional
            Not used in this implementation, by default True
        """
        N = x.shape[0]
        # Revert the mean of the moments matrix
        moments_matrix = n * mm.moments_matrix  # type: ignore

        # Compute the design matrix to add
        V = PolynomialsBasisGenerator.make_design_matrix(x, mm.monomials_matrix, mm.polynomial_class)  # type: ignore

        # Update the moments matrix
        moments_matrix += V.T @ V
        # Compute the mean of the updated moments matrix
        moments_matrix /= n + N
        mm.moments_matrix = moments_matrix

        # Compute the inverse of the updated moments matrix
        mm.inverse_moments_matrix = inv_opt(moments_matrix)


class ShermanIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the Sherman-Morrison formula.
    This method updates the inverse moments matrix iteratively and not modifies the moments matrix.
    """

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_opt: Callable[[np.ndarray], np.ndarray] | None = None, sym: bool = True):
        """Increment the moments matrix using the Sherman-Morrison formula iteratively.

        Sherman-Morrison formula: \\
        (A + uv^T)^-1 = A^-1 - (A^-1 u v^T A^-1) / (1 + v^T A^-1 u) \\
        Here u = v^T

        Parameters
        ----------
        mm : MomentsMatrix
            The moments matrix to increment.
        x : np.ndarray
            The input data.
        n : int
            The number of points integrated in the moments matrix.
        inv_opt : Callable[[np.ndarray], np.ndarray] | None, optional
            Not used in this implementation, by default None
        sym : bool, optional
            Whether to consider the matrix as symmetric, by default True
        """
        # Revert the mean of the moments matrix
        inv_moments_matrix = mm.inverse_moments_matrix / n  # type: ignore

        # Apply the Sherman-Morrison formula iteratively
        for xx in x:
            # Compute the vector v for the current point
            v = mm.polynomial_class(xx, mm.monomials_matrix)
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


class WoodburyIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the Woodbury matrix identity.
    This method updates the inverse moments matrix and not modifies the moments matrix.
    """

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_opt: Callable[[np.ndarray], np.ndarray], sym: bool = True):
        """Increment the moments matrix using the Woodbury matrix identity.
        
        Woodbury matrix identity: \\
        (A + UCV)^-1 = A^-1 - A^-1 U (C^-1 + V A^-1 U)^-1 V A^-1 \\
        Here C = I and U = V^T = [v_1, v_2, ..., v_N] \\
        
        Parameters
        ----------
        mm : MomentsMatrix
            The moments matrix to increment.
        x : np.ndarray
            The input data.
        n : int
            The number of points integrated in the moments matrix.
        inv_opt : Callable[[np.ndarray], np.ndarray]
            The inversion method to use.
        sym : bool, optional
            Whether to consider the matrix as symmetric, by default True
        """
        N = x.shape[0]

        # Revert the mean of the moments matrix
        inv_moments_matrix = mm.inverse_moments_matrix / n  # type: ignore

        # Compute the design matrix to add
        V = PolynomialsBasisGenerator.make_design_matrix(x, mm.monomials_matrix, mm.polynomial_class)  # type: ignore

        # Define the identity matrix C
        C = np.eye(N)

        # Compute the product V @ A^-1
        V_A_inv = V @ inv_moments_matrix
        # Compute the sum C + V @ A^-1 @ U
        sum_ = C + V_A_inv @ V.T
        # Compute the inverse of the sum
        if N == 1:
            sum_inv = 1 / sum_
        else:
            sum_inv = inv_opt(sum_)
        # Compute A^-1 @ U @ (C^-1 + V @ A^-1 @ U)^-1
        if sym:
            # A is symmetric and U = V^T so A^-1 @ U = (V @ A^-1)^T
            A_inv_U_sum_inv = V_A_inv.T @ sum_inv
        else:
            A_inv_U = inv_moments_matrix @ V.T
            A_inv_U_sum_inv = A_inv_U @ sum_inv
        # Compute the product A^-1 @ U @ (C^-1 + V @ A^-1 @ U)^-1 @ V^T @ A^-1
        prod = A_inv_U_sum_inv @ V_A_inv

        inv_moments_matrix -= prod

        # Compute the mean of the updated inverse moments matrix
        mm.inverse_moments_matrix = (n + N) * inv_moments_matrix


IMPLEMENTED_INCREMENTATERS_OPTIONS: dict[str, type[BaseIncrementer]] = {
    "inverse": InverseIncrementer,
    "sherman": ShermanIncrementer,
    "woodbury": WoodburyIncrementer,
}
