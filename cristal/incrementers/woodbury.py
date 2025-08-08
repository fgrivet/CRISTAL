"""
Class for incrementing moments matrix using the Woodbury matrix identity.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..inverters.base import BaseInverter
from ..polynomials.base import MultivariatePolynomialBasis
from .base import BaseIncrementer

if TYPE_CHECKING:
    from ..moments_matrix.moments_matrix import MomentsMatrix


class WoodburyIncrementer(BaseIncrementer):
    """
    Incrementer for moments matrix using the Woodbury matrix identity.
    This method updates the inverse moments matrix and not modifies the moments matrix.
    """

    @staticmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_class: type[BaseInverter], sym: bool = True):
        """Increment the moments matrix using the Woodbury matrix identity.

        Woodbury matrix identity:

        .. math::
            (A + UCV)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}

        .. math::
            \\text{Here } C = I \\text{ and } U = V^T = \\begin{bmatrix} v_1, v_2, \\cdots, v_{N'} \\end{bmatrix}

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
        N = x.shape[0]

        # Revert the mean of the moments matrix
        inv_moments_matrix = mm.inverse_moments_matrix / n  # type: ignore

        # Compute the design matrix to add
        V = MultivariatePolynomialBasis.make_design_matrix(x, mm.multidegree_combinations, mm.polynomial_class)  # type: ignore

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
            sum_inv = inv_class.invert(sum_)
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
