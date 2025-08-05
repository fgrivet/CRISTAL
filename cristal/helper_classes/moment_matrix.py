"""
MomentsMatrix class for computing and storing moments matrix and its inverse.
"""

import copy
import logging
from typing import Any, Self

import numpy as np

from cristal.helper_classes.base import BaseIncrementer, BaseInverter, BasePolynomialBasis
from cristal.helper_classes.incrementers import IMPLEMENTED_INCREMENTERS_OPTIONS
from cristal.helper_classes.inversion import IMPLEMENTED_INVERSION_OPTIONS
from cristal.helper_classes.polynomial_basis import IMPLEMENTED_POLYNOMIAL_BASIS, PolynomialsBasisGenerator
from cristal.utils.type_checking import check_types, positive_integer

logger = logging.getLogger("CRISTAL")

__all__ = [
    "MomentsMatrix",
]


class MomentsMatrix:
    """Class for computing and storing moments matrix and its inverse.

    Attributes
    ----------
    n : int
        The degree of the polynomial basis.
    polynomial_basis : str
        The polynomial basis to use.
    inv_opt : str
        The inversion option to use.
    incr_opt : str
        The incrementation option to use.
    polynomial_func : Callable(np.ndarray, np.ndarray) -> np.ndarray
        The function to apply the polynomial combinations.
    inv_class : BaseInverter
        The class to use for inverting the moments matrix.
    incr_class : BaseIncrementer
        The class to use for incrementing the inverse moments matrix.
    monomials_matrix : np.ndarray | None
        The monomials matrix used for computing the moments matrix. None if not fitted.
    moments_matrix : np.ndarray | None
        The moments matrix. None if not fitted.
    inverse_moments_matrix : np.ndarray | None
        The inverse moments matrix. None if not fitted.
    N : int
        The number of points integrated in the moments matrix.
    """

    @check_types(
        {
            "n": positive_integer,
        }
    )
    def __init__(
        self,
        n: int,
        polynomial_basis: IMPLEMENTED_POLYNOMIAL_BASIS = IMPLEMENTED_POLYNOMIAL_BASIS.MONOMIALS,
        inv_opt: IMPLEMENTED_INVERSION_OPTIONS = IMPLEMENTED_INVERSION_OPTIONS.INV,
        incr_opt: IMPLEMENTED_INCREMENTERS_OPTIONS = IMPLEMENTED_INCREMENTERS_OPTIONS.INVERSE,
    ):
        """Initialize the MomentsMatrix.

        Parameters
        ----------
        n : int
            The degree of the polynomial basis.
        polynomial_basis : IMPLEMENTED_POLYNOMIAL_BASIS, optional
            The polynomial basis to use, by default IMPLEMENTED_POLYNOMIAL_BASIS.MONOMIALS
        inv_opt : IMPLEMENTED_INVERSION_OPTIONS, optional
            The inversion option to use, by default IMPLEMENTED_INVERSION_OPTIONS.INV
        incr_opt : IMPLEMENTED_INCREMENTERS_OPTIONS, optional
            The incrementation option to use, by default IMPLEMENTED_INCREMENTERS_OPTIONS.INVERSE
        """
        # Initialize the parameters
        self.n = n
        self.polynomial_basis = polynomial_basis
        self.inv_opt = inv_opt
        self.incr_opt = incr_opt
        # Initialize the functions
        self.polynomial_class: type[BasePolynomialBasis] = self.polynomial_basis.value
        self.inv_class: type[BaseInverter] = self.inv_opt.value
        self.incr_class: type[BaseIncrementer] = self.incr_opt.value
        # Initialize the moments matrix and its inverse (these variables will be set during the fit method)
        self.monomials_matrix = None
        self.moments_matrix = None
        self.inverse_moments_matrix = None
        self.N = 0  # Number of points integrated in the moments matrix

    def fit(self, x: np.ndarray) -> Self:
        """Construct the moments matrix and its inverse from the input data.

        Parameters
        ----------
        x : np.ndarray (N, d)
            The input data.

        Returns
        -------
        Self
            The fitted MomentsMatrix instance.
        """
        self.N = x.shape[0]
        # Generate the monomials based on the degree n and the number of features in x
        monomials_matrix = PolynomialsBasisGenerator.generate_combinations(self.n, x.shape[1])
        self.monomials_matrix = monomials_matrix
        # Compute the design matrix
        X = PolynomialsBasisGenerator.make_design_matrix(x, monomials_matrix, self.polynomial_class)
        # Compute the moments matrix and its inverse
        moments_matrix: np.ndarray = np.dot(X.T, X) / self.N
        self.moments_matrix = moments_matrix
        self.inverse_moments_matrix = self.inv_class.invert(moments_matrix)
        return self

    def score_samples(self, x: np.ndarray) -> np.ndarray:
        """Compute the score for each sample in the input data.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data.

        Returns
        -------
        np.ndarray (L,)
            The scores for each sample.
        """
        if not self.is_fitted():
            raise ValueError("MomentsMatrix is not fitted. Call fit() before score_samples().")
        # Compute the design matrix for the input data
        v_matrix = PolynomialsBasisGenerator.make_design_matrix(x, self.monomials_matrix, self.polynomial_class)  # type: ignore
        # Compute the scores using the moments matrix and its inverse
        # The score is computed as v(xx) @ M^-1 @ v(xx)^T for each sample xx in x
        temp = np.dot(v_matrix, self.inverse_moments_matrix)  # type: ignore
        scores = np.sum(temp * v_matrix, axis=1)
        return scores

    def update(self, x: np.ndarray, sym: bool = True) -> "MomentsMatrix":
        """Update the inverse of the moments matrix with new data.

        Parameters
        ----------
        x : np.ndarray (N', d)
            The new input data.
        sym : bool, optional
            Whether the inverse of the moments matrix should be considered as symmetric, by default True

        Returns
        -------
        MomentsMatrix
            The updated MomentsMatrix instance.
        """
        self.incr_class.increment(self, x, self.N, self.inv_class, sym=sym)
        self.N += x.shape[0]
        if not self.incr_class.update_moments_matrix:
            logger.warning(
                "The moment matrix has not been updated, only the inverse matrix has been updated. "
                "This means that moments_matrix x inverse_moments_matrix is not equal to the identity matrix."
            )
        return self

    def save_model(self) -> dict[str, Any]:
        """Save the MomentsMatrix model as a dictionary.

        Returns
        -------
        dict
            The MomentsMatrix model represented as a dictionary, with keys:
            - "n": int, the degree of the polynomial basis
            - "polynomial_basis": str, the polynomial basis used
            - "inv_opt": str, the inversion option used
            - "incr_opt": str, the incrementation option used
            - "monomials_matrix": list[list[float]] | None, the monomials matrix
            - "moments_matrix": list[list[float]] | None, the moments matrix
            - "inverse_moments_matrix": list[list[float]] | None, the inverse moments matrix
            - "N": int, the number of points integrated in the moments matrix
        """
        return {
            "n": self.n,
            "polynomial_basis": self.polynomial_basis.name,
            "inv_opt": self.inv_opt.name,
            "incr_opt": self.incr_opt.name,
            "monomials_matrix": self.monomials_matrix.tolist() if self.monomials_matrix is not None else None,
            "moments_matrix": self.moments_matrix.tolist() if self.moments_matrix is not None else None,
            "inverse_moments_matrix": self.inverse_moments_matrix.tolist() if self.inverse_moments_matrix is not None else None,
            "N": self.N,
        }

    def load_model(self, model_dict: dict) -> Self:
        """Load the model parameters from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the model parameters, with keys:
            - "n": int, the degree of the polynomial basis
            - "polynomial_basis": str, the polynomial basis used
            - "inv_opt": str, the inversion option used
            - "incr_opt": str, the incrementation option used
            - "monomials_matrix": list[list[float]] | None, the monomials matrix
            - "moments_matrix": list[list[float]] | None, the moments matrix
            - "inverse_moments_matrix": list[list[float]] | None, the inverse moments matrix
            - "N": int, the number of points integrated in the moments matrix

        Returns
        -------
        Self
            The updated MomentsMatrix instance.
        """
        # Load the parameters
        self.n = model_dict["n"]
        self.polynomial_basis = IMPLEMENTED_POLYNOMIAL_BASIS[model_dict["polynomial_basis"]]
        self.inv_opt = IMPLEMENTED_INVERSION_OPTIONS[model_dict["inv_opt"]]
        self.incr_opt = IMPLEMENTED_INCREMENTERS_OPTIONS[model_dict["incr_opt"]]
        # Initialize the functions
        self.polynomial_class: type[BasePolynomialBasis] = self.polynomial_basis.value
        self.inv_class: type[BaseInverter] = self.inv_opt.value
        self.incr_class: type[BaseIncrementer] = self.incr_opt.value
        # Load the moments matrix and its inverse
        self.monomials_matrix = np.array(model_dict["monomials_matrix"], dtype=np.int8) if model_dict["monomials_matrix"] is not None else None
        self.moments_matrix = np.array(model_dict["moments_matrix"], dtype=np.float64) if model_dict["moments_matrix"] is not None else None
        self.inverse_moments_matrix = (
            np.array(model_dict["inverse_moments_matrix"], dtype=np.float64) if model_dict["inverse_moments_matrix"] is not None else None
        )
        self.N = model_dict["N"]
        return self

    def is_fitted(self) -> bool:
        """Check if the MomentsMatrix is fitted.

        Returns
        -------
        bool
            True if the MomentsMatrix is fitted, False otherwise.
        """
        return self.monomials_matrix is not None and self.moments_matrix is not None and self.inverse_moments_matrix is not None and self.N > 0

    def copy(self) -> Self:
        """Create a copy of the MomentsMatrix instance.

        Returns
        -------
        Self
            A copy of the MomentsMatrix instance.
        """
        return copy.deepcopy(self)
