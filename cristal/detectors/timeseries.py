"""
This module implements the :class:`UTSCF` (Univariate Time Series Christoffel Function) anomaly detection algorithm.
"""

import logging

import numpy as np

from ..decomposers import IMPLEMENTED_DECOMPOSERS, BaseDecomposer
from ..incrementers import IMPLEMENTED_INCREMENTERS
from ..inverters import IMPLEMENTED_INVERTERS
from ..moments_matrix.moments_matrix import MomentsMatrix
from ..polynomials import IMPLEMENTED_POLYNOMIALS
from ..regularizers import IMPLEMENTED_REGULARIZERS
from ..type_checking import check_types, positive_integer
from .base import BaseDetector, NotFittedError

logger = logging.getLogger(__name__)

# TODO Take into account the reconstruction error in the score formula


class UTSCF(BaseDetector):
    """
    Univariate Time Series Christoffel Function

    Attributes
    ----------
    n: int
        The degree of polynomials, usually set between 2 and 8.
    regularizer_opt: IMPLEMENTED_REGULARIZERS
        The regularization method to use to divide the scores.
    C: float | int
        The constant factor used in the :const:`"vu_C"` or :const:`"constant"` regularizer.
    decomposer_opt: IMPLEMENTED_DECOMPOSERS
        The decomposition method to use to decompose the timeseries into :attr:`d` components.
    decomposer_kwargs: dict | None
        The keyword arguments to pass to the decomposer.
    decomposer_class: type[BaseDecomposer]
        The class of the decomposer to use.
    d: int
        The number of components to retain during decomposition.
    moments_matrix: MomentsMatrix
        The moments matrix object.
    regularizer_value: float | int | None
        The regularization factor computed based on the degree and dimension. None if not fitted.
    indices_train: np.ndarray | None
        The decomposition indices retained during training. None if not fitted.
    """

    @check_types(
        {
            "n": positive_integer,
            "d": positive_integer,
        }
    )
    def __init__(
        self,
        n: int,
        regularizer_opt: IMPLEMENTED_REGULARIZERS = IMPLEMENTED_REGULARIZERS.VU_C,
        C: float | int = 1,
        polynomial_family_opt: IMPLEMENTED_POLYNOMIALS = IMPLEMENTED_POLYNOMIALS.MONOMIALS,
        inverter_opt: IMPLEMENTED_INVERTERS = IMPLEMENTED_INVERTERS.FPD,
        incrementer_opt: IMPLEMENTED_INCREMENTERS = IMPLEMENTED_INCREMENTERS.INVERSE,
        decomposer_opt: IMPLEMENTED_DECOMPOSERS = IMPLEMENTED_DECOMPOSERS.WINDOW_FOURIER,
        decomposer_kwargs: dict | None = None,
        d: int = 5,
    ):
        """Initialize the UTSCF model.

        Parameters
        ----------
        n : int
            The degree of polynomials, usually set between 2 and 8.
        regularizer_opt : IMPLEMENTED_REGULARIZERS, optional
            The regularization method to use to divide the scores, by default IMPLEMENTED_REGULARIZERS.VU_C
            See :attr:`~cristal.regularizers.IMPLEMENTED_REGULARIZERS` for more details.
        C : float | int, optional
            The constant factor used in the :const:`"vu_C"` or :const:`"constant"` regularizer, by default 1
        polynomial_family_opt : IMPLEMENTED_POLYNOMIALS, optional
            The polynomial basis to use for the moments matrix, by default IMPLEMENTED_POLYNOMIALS.MONOMIALS
            See :attr:`~cristal.polynomials.IMPLEMENTED_POLYNOMIALS` for more details.
        inverter_opt : IMPLEMENTED_INVERTERS, optional
            The inversion option to use for the moments matrix, by default IMPLEMENTED_INVERTERS.FPD
            See :attr:`~cristal.inverters.IMPLEMENTED_INVERTERS` for more details.
        incrementer_opt : IMPLEMENTED_INCREMENTERS, optional
            The incrementation option to use for updating the moments matrix, by default IMPLEMENTED_INCREMENTERS.INVERSE
            See :attr:`~cristal.incrementers.IMPLEMENTED_INCREMENTERS` for more details.
        decomposer_opt : IMPLEMENTED_DECOMPOSERS, optional
            The decomposition method to use to decompose the timeseries into :attr:`d` components, by default IMPLEMENTED_DECOMPOSERS.WINDOW_FOURIER
            See :attr:`~cristal.decomposers.IMPLEMENTED_DECOMPOSERS` for more details.
        decomposer_kwargs : dict | None, optional
            The keyword arguments to pass to the decomposer, by default None
        d : int, optional
            The number of components to retain during decomposition, by default 5
        """
        # Initialize the parameters
        self.n = n
        self.regularizer_opt = regularizer_opt
        self.C = C
        self.decomposer_opt = decomposer_opt
        self.decomposer_kwargs = decomposer_kwargs if decomposer_kwargs is not None else {}
        self.d = d
        # Initialize the MomentsMatrix
        self.moments_matrix = MomentsMatrix(
            n, incrementer_opt=incrementer_opt, polynomial_family_opt=polynomial_family_opt, inverter_opt=inverter_opt
        )
        # Initialize the functions
        self.decomposer_class: type[BaseDecomposer] = self.decomposer_opt.value
        # Initialize the variables set during the fit method
        self.regularizer_value = None
        self.indices_train = None

    def fit(self, x: np.ndarray):
        self.assert_shape_unfitted(x)
        indices, coefs = self.decomposer_class.decompose(x, n_coefs=self.d, **self.decomposer_kwargs)
        self.indices_train = indices
        self.moments_matrix.fit(coefs.reshape(-1, self.d))
        self.regularizer_value = self.regularizer_opt.value.compute_value(self.n, self.d, self.C)
        return self

    def update(self, x: np.ndarray, sym: bool = True):
        """Update the current model with instances in x.

        Parameters
        ----------
        x: np.ndarray (N', d)
            The input data to update the model.
        sym: bool, optional
            Whether the inverse of the moments matrix should be considered as symmetric, by default True

        Returns
        -------
        UTSCF
            The updated UTSCF instance.
        """
        self.assert_shape_fitted(x)  # also check is_fitted and so if self.indices_train is not None
        coefs = self.decomposer_class.get_coefficients_at_indices(x, self.indices_train, **self.decomposer_kwargs)  # type: ignore
        self.moments_matrix.update(coefs.reshape(-1, self.d), sym=sym)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)  # also check is_fitted and so if self.indices_train is not None
        coefs = self.decomposer_class.get_coefficients_at_indices(x, self.indices_train, **self.decomposer_kwargs)  # type: ignore
        return self.moments_matrix.score_samples(coefs.reshape(-1, self.d)) / self.regularizer_value

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return (1 / self.score_samples(x)) - 1

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            evals[i] = self.decision_function(xx.reshape(-1, self.d))
            self.update(xx.reshape(-1, self.d))
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i, xx in enumerate(x):
            xx.reshape(1, -1)
            preds[i] = self.predict(xx.reshape(-1, self.d))
            self.update(xx.reshape(-1, self.d))
        return preds

    def save_model(self):
        """Saves the UTSCF model as a dictionary.

        Returns
        -------
        dict
            The UTSCF model represented as a dictionary, with keys:
                - "n": int, the degree of the polynomial basis.
                - "regularizer_opt": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "moments_matrix": dict, the saved moments matrix.
                - "decomposer_opt": str, the decomposition method used.
                - "decomposer_kwargs": dict, the keyword arguments for the decomposer.
                - "d": int, the number of components to retain during decomposition.
                - "regularizer_value": float | int | None, the regularization factor.
                - "indices_train": np.ndarray | None, the decomposition indices retained during training. None if not fitted.
        """
        return {
            "n": self.n,
            "regularizer_opt": self.regularizer_opt.name,
            "C": self.C,
            "moments_matrix": self.moments_matrix.save_model(),
            "decomposer_opt": self.decomposer_opt.name,
            "decomposer_kwargs": self.decomposer_kwargs,
            "d": self.d,
            "regularizer_value": self.regularizer_value,
            "indices_train": self.indices_train.tolist() if self.indices_train is not None else None,
        }

    def load_model(self, model_dict: dict):
        """Loads the UTSCF model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the UTSCF parameters returned by :func:`save_model()`:
                - "n": int, the degree of the polynomial basis.
                - "regularizer_opt": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "moments_matrix": dict, the saved moments matrix.
                - "decomposer_opt": str, the decomposition method used.
                - "decomposer_kwargs": dict, the keyword arguments for the decomposer.
                - "d": int, the number of components to retain during decomposition.
                - "regularizer_value": float | int | None, the regularization factor.
                - "indices_train": np.ndarray | None, the decomposition indices retained during training. None if not fitted.

        Returns
        -------
        UTSCF
            The loaded UTSCF instance with updated parameters.
        """
        self.n = model_dict["n"]
        self.regularizer_opt = IMPLEMENTED_REGULARIZERS[model_dict["regularizer_opt"]]
        self.C = model_dict["C"]
        self.moments_matrix.load_model(model_dict["moments_matrix"])
        self.decomposer_opt = IMPLEMENTED_DECOMPOSERS[model_dict["decomposer_opt"]]
        self.decomposer_class = self.decomposer_opt.value
        self.decomposer_kwargs = model_dict["decomposer_kwargs"]
        self.d = model_dict["d"]
        self.regularizer_value = model_dict["regularizer_value"]
        self.indices_train = np.array(model_dict["indices_train"]) if model_dict["indices_train"] is not None else None
        return self

    def is_fitted(self):
        return self.moments_matrix.is_fitted() and self.regularizer_value is not None and self.indices_train is not None

    def method_name(self) -> str:
        return "UTSCF"

    @staticmethod
    def assert_shape_unfitted(x: np.ndarray):
        """Asserts that the input array x has the correct shape to fit the model.

        Parameters
        ----------
        x : np.ndarray
            The input data to check. Should be a 2D array with shape (N, d)
            where N is the number of samples and d is the number of features.

        Raises
        ------
        ValueError
            If the input array does not have the expected shape.
        """
        if x.ndim != 1 or x.shape[0] == 0:
            raise ValueError(f"The expected array shape: (N,) do not match the given array shape: {x.shape}")

    def assert_shape_fitted(self, x: np.ndarray):
        """Asserts that the input array x has the correct shape to use the model after it has been fitted.

        Parameters
        ----------
        x : np.ndarray
            The input data to check. Should be a 2D array with shape (N, d)
            where N is the number of samples and d is the number of features (the same as during fitting).

        Raises
        ------
        NotFittedError
            If the model has not been fitted yet.
        ValueError
            If the input array does not have the expected shape.
        """
        if not self.is_fitted():
            raise NotFittedError(
                f"This {self.method_name()} instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator."
            )
        d = self.__dict__.get("d")
        if x.ndim != 1 or x.shape[0] == 0:
            raise ValueError(f"The expected array shape: (N,) do not match the given array shape: {x.shape}")
