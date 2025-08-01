"""
MomentsMatrix class for computing and storing moments matrix and its inverse.
"""

import copy
from typing import Any, Self

import numpy as np

from cristal.helper_classes.incrementers import IMPLEMENTED_INCREMENTATERS_OPTIONS
from cristal.helper_classes.inversion import IMPLEMENTED_INVERSION_OPTIONS
from cristal.helper_classes.polynomial_basis import IMPLEMENTED_POLYNOMIAL_BASIS, PolynomialsBasisGenerator
from cristal.utils.type_checking import check_in_list, check_types, positive_integer

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

    Methods
    -------
    __init__(n, polynomial_basis="monomials", inv_opt="inv", incr_opt="inverse")
        Initialize the MomentsMatrix with the degree of the polynomials,
        the polynomial basis type, the inversion option, and the incrementation option.
    fit(x)
        Fit the MomentsMatrix to the input data.
    score_samples(x)
        Compute the score for each sample in the input data.
    update(x, sym=True)
        Update the inverse of the moments matrix with new data.
    save_model()
        Save the model parameters.
    load_model(model_dict)
        Load the model parameters from a dictionary.
    is_fitted()
        Check if the MomentsMatrix is fitted.
    copy()
        Create a copy of the MomentsMatrix instance.
    """

    @check_types(
        {
            "n": positive_integer,
            "polynomial_basis": lambda x: check_in_list(x, IMPLEMENTED_POLYNOMIAL_BASIS),
            "inv_opt": lambda x: check_in_list(x, IMPLEMENTED_INVERSION_OPTIONS),
            "incr_opt": lambda x: check_in_list(x, IMPLEMENTED_INCREMENTATERS_OPTIONS),
        }
    )
    def __init__(self, n: int, polynomial_basis: str = "monomials", inv_opt: str = "inv", incr_opt: str = "inverse"):
        """Initialize the MomentsMatrix.

        Parameters
        ----------
        n : int
            The degree of the polynomial basis.
        polynomial_basis : str, optional
            The polynomial basis to use, by default "monomials"
        inv_opt : str, optional
            The inversion option to use, by default "inv"
        incr_opt : str, optional
            The incrementation option to use, by default "inverse"
        """
        # Initialize the parameters
        self.n = n
        self.polynomial_basis = polynomial_basis
        self.inv_opt = inv_opt
        self.incr_opt = incr_opt
        # Initialize the functions
        self.polynomial_class = IMPLEMENTED_POLYNOMIAL_BASIS[polynomial_basis]
        self.inv_class = IMPLEMENTED_INVERSION_OPTIONS[inv_opt]
        self.incr_class = IMPLEMENTED_INCREMENTATERS_OPTIONS[incr_opt]
        # Initialize the moments matrix and its inverse (these variables will be set during the fit method)
        self.monomials_matrix = None
        self.moments_matrix = None
        self.inverse_moments_matrix = None
        self.N = 0  # Number of points integrated in the moments matrix

    def fit(self, x: np.ndarray) -> Self:
        """Construct the moments matrix and its inverse from the input data.

        Parameters
        ----------
        x : np.ndarray
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
        x : np.ndarray
            The input data. Shape should be (n_samples, n_features).

        Returns
        -------
        np.ndarray
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

    def update(self, x: np.ndarray, sym: bool = True) -> Self:
        """Update the inverse of the moments matrix with new data.

        Parameters
        ----------
        x : np.ndarray
            The new input data.
        sym : bool, optional
            Whether the inverse of the moments matrix should be considered as symmetric, by default True

        Returns
        -------
        Self
            The updated MomentsMatrix instance.
        """
        self.incr_class.increment(self, x, self.N, self.inv_class.invert, sym=sym)
        self.N += x.shape[0]
        if self.incr_opt != "inverse":
            print(
                "Warning: Incrementation option is not 'inverse', the moment matrix has not been updated, only the inverse matrix has been updated."
            )
        return self

    def save_model(self) -> dict[str, Any]:
        """Save the model parameters.

        Returns
        -------
        dict
            A dictionary containing the model parameters.
        """
        return {
            "n": self.n,
            "polynomial_basis": self.polynomial_basis,
            "inv_opt": self.inv_opt,
            "incr_opt": self.incr_opt,
            "monomials_matrix": self.monomials_matrix.tolist() if self.monomials_matrix is not None else None,
            "moments_matrix": self.moments_matrix.tolist() if self.moments_matrix is not None else None,
            "inverse_moments_matrix": self.inverse_moments_matrix.tolist() if self.inverse_moments_matrix is not None else None,
            "N": self.N,
        }

    def load_model(self, model_dict: dict[str, Any]) -> Self:
        """Load the model parameters from a dictionary.

        Parameters
        ----------
        model_dict : dict[str, Any]
            A dictionary containing the model parameters.
            Should contain keys: "n", "polynomial_basis", "inv_opt", "incr_opt", "monomials_matrix", "moments_matrix", "inverse_moments_matrix", "N".

        Returns
        -------
        Self
            The updated MomentsMatrix instance.
        """
        # Load the parameters
        self.n = model_dict["n"]
        self.polynomial_basis = model_dict["polynomial_basis"]
        self.inv_opt = model_dict["inv_opt"]
        self.incr_opt = model_dict["incr_opt"]
        # Initialize the functions
        self.polynomial_class = IMPLEMENTED_POLYNOMIAL_BASIS[self.polynomial_basis]
        self.inv_class = IMPLEMENTED_INVERSION_OPTIONS[self.inv_opt]
        self.incr_class = IMPLEMENTED_INCREMENTATERS_OPTIONS[self.incr_opt]
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
