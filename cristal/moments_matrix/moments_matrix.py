"""
Class for computing and storing moments matrix and its inverse.
"""

import copy
import logging
from typing import Any, Self

import numpy as np

from ..incrementers import IMPLEMENTED_INCREMENTERS, BaseIncrementer
from ..inverters import IMPLEMENTED_INVERTERS, BaseInverter
from ..polynomials import IMPLEMENTED_POLYNOMIALS, BasePolynomialFamily, MultivariatePolynomialBasis
from ..type_checking import check_types, positive_integer

logger = logging.getLogger(__name__)


class MomentsMatrix:
    """Class for computing and storing moments matrix and its inverse.

    Attributes
    ----------
    n : int
        The degree of the polynomial basis.
    polynomial_family_opt : str
        The polynomial basis to use.
    inverter_opt : str
        The inversion option to use.
    incrementer_opt : str
        The incrementation option to use.
    polynomial_func : Callable(np.ndarray, np.ndarray) -> np.ndarray
        The function to apply the polynomial combinations.
    inv_class : BaseInverter
        The class to use for inverting the moments matrix.
    incr_class : BaseIncrementer
        The class to use for incrementing the inverse moments matrix.
    multidegree_combinations : np.ndarray | None
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
        polynomial_family_opt: IMPLEMENTED_POLYNOMIALS = IMPLEMENTED_POLYNOMIALS.MONOMIALS,
        inverter_opt: IMPLEMENTED_INVERTERS = IMPLEMENTED_INVERTERS.INV,
        incrementer_opt: IMPLEMENTED_INCREMENTERS = IMPLEMENTED_INCREMENTERS.INVERSE,
    ):
        """Initialize the MomentsMatrix.

        Parameters
        ----------
        n : int
            The degree of the polynomial basis.
        polynomial_family_opt : IMPLEMENTED_POLYNOMIALS, optional
            The polynomial family to use, by default IMPLEMENTED_POLYNOMIALS.MONOMIALS
        inverter_opt : IMPLEMENTED_INVERTERS, optional
            The inversion option to use, by default IMPLEMENTED_INVERTERS.INV
        incrementer_opt : IMPLEMENTED_INCREMENTERS, optional
            The incrementation option to use, by default IMPLEMENTED_INCREMENTERS.INVERSE
        """
        # Initialize the parameters
        self.n = n
        self.polynomial_family_opt = polynomial_family_opt
        self.inverter_opt = inverter_opt
        self.incrementer_opt = incrementer_opt
        # Initialize the functions
        self.polynomial_class: type[BasePolynomialFamily] = self.polynomial_family_opt.value
        self.inverter_class: type[BaseInverter] = self.inverter_opt.value
        self.incrementer_class: type[BaseIncrementer] = self.incrementer_opt.value
        # Initialize the moments matrix and its inverse (these variables will be set during the fit method)
        self.multidegree_combinations = None
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
        # Take modulus if x is complex
        # if np.iscomplexobj(x):
        #     x = np.abs(x)
        self.N = x.shape[0]
        # Generate the monomials based on the degree n and the number of features in x
        multidegree_combinations = MultivariatePolynomialBasis.generate_multidegree_combinations(self.n, x.shape[1])
        self.multidegree_combinations = multidegree_combinations
        # Compute the design matrix
        X = MultivariatePolynomialBasis.make_design_matrix(x, multidegree_combinations, self.polynomial_class)
        # Compute the moments matrix and its inverse
        moments_matrix: np.ndarray = np.dot(X.T, X) / self.N
        self.moments_matrix = moments_matrix
        self.inverse_moments_matrix = self.inverter_class.invert(moments_matrix)
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
        # Take modulus if x is complex
        # if np.iscomplexobj(x):
        #     x = np.abs(x)
        # Compute the design matrix for the input data
        v_matrix = MultivariatePolynomialBasis.make_design_matrix(x, self.multidegree_combinations, self.polynomial_class)  # type: ignore
        # Compute the scores using the moments matrix and its inverse
        # The score is computed as v(xx) @ M^-1 @ v(xx)^T for each sample xx in x
        temp = np.dot(v_matrix, self.inverse_moments_matrix)  # type: ignore
        scores = np.sum(temp * v_matrix, axis=1)
        # Take modulus if scores are complex
        # Keep a complex matrix and vectors v and apply modulus to the scores only
        if np.iscomplexobj(scores):
            scores = np.abs(scores)
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
        self.incrementer_class.increment(self, x, self.N, self.inverter_class, sym=sym)
        self.N += x.shape[0]
        if not self.incrementer_class.update_moments_matrix:
            logger.warning(
                "The moments matrix has not been updated, only the inverse matrix has been updated. "
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
            - "polynomial_family_opt": str, the polynomial basis used
            - "inverter_opt": str, the inversion option used
            - "incrementer_opt": str, the incrementation option used
            - "multidegree_combinations": list[list[float]] | None, the monomials matrix
            - "moments_matrix": list[list[float]] | None, the moments matrix
            - "inverse_moments_matrix": list[list[float]] | None, the inverse moments matrix
            - "N": int, the number of points integrated in the moments matrix
        """
        return {
            "n": self.n,
            "polynomial_family_opt": self.polynomial_family_opt.name,
            "inverter_opt": self.inverter_opt.name,
            "incrementer_opt": self.incrementer_opt.name,
            "multidegree_combinations": self.multidegree_combinations.tolist() if self.multidegree_combinations is not None else None,
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
            - "polynomial_family_opt": str, the polynomial basis used
            - "inverter_opt": str, the inversion option used
            - "incrementer_opt": str, the incrementation option used
            - "multidegree_combinations": list[list[float]] | None, the monomials matrix
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
        self.polynomial_family_opt = IMPLEMENTED_POLYNOMIALS[model_dict["polynomial_family_opt"]]
        self.inverter_opt = IMPLEMENTED_INVERTERS[model_dict["inverter_opt"]]
        self.incrementer_opt = IMPLEMENTED_INCREMENTERS[model_dict["incrementer_opt"]]
        # Initialize the functions
        self.polynomial_class: type[BasePolynomialFamily] = self.polynomial_family_opt.value
        self.inverter_class: type[BaseInverter] = self.inverter_opt.value
        self.incrementer_class: type[BaseIncrementer] = self.incrementer_opt.value
        # Load the moments matrix and its inverse
        self.multidegree_combinations = (
            np.array(model_dict["multidegree_combinations"], dtype=np.int8) if model_dict["multidegree_combinations"] is not None else None
        )
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
        return (
            self.multidegree_combinations is not None and self.moments_matrix is not None and self.inverse_moments_matrix is not None and self.N > 0
        )

    def copy(self) -> Self:
        """Create a copy of the MomentsMatrix instance.

        Returns
        -------
        Self
            A copy of the MomentsMatrix instance.
        """
        return copy.deepcopy(self)
