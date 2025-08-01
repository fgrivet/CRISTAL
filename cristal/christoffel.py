"""
Christoffel Function and Growth Detector classes
"""

import numpy as np

from cristal.helper_classes import (
    IMPLEMENTED_INCREMENTATERS_OPTIONS,
    IMPLEMENTED_INVERSION_OPTIONS,
    IMPLEMENTED_POLYNOMIAL_BASIS,
    IMPLEMENTED_REGULARIZATION_OPTIONS,
    BaseDetector,
    MomentsMatrix,
)
from cristal.utils.type_checking import check_in_list, check_types, positive_integer


class DyCF(BaseDetector):
    """
    Dynamical Christoffel Function

    Attributes
    ----------
    n: int
        The degree of polynomials, usually set between 2 and 8.
    regularization: str
        The regularization method to use for the score.
    C: float | int
        The constant factor used in the "vu_C" or "constant" regularization.
    moments_matrix: MomentsMatrix
        The moments matrix object.
    d: int
        The dimension of the input data. 0 if not fitted.
    regularizer: float | int | None
        The regularization factor computed based on the degree and dimension. None if not fitted.

    Methods
    -------
    See BaseDetector methods
    """

    @check_types(
        {
            "n": positive_integer,
            "incr_opt": lambda x: check_in_list(x, IMPLEMENTED_INCREMENTATERS_OPTIONS),
            "polynomial_basis": lambda x: check_in_list(x, IMPLEMENTED_POLYNOMIAL_BASIS),
            "regularization": lambda x: check_in_list(x, IMPLEMENTED_REGULARIZATION_OPTIONS),
            "inv": lambda x: check_in_list(x, IMPLEMENTED_INVERSION_OPTIONS),
        }
    )
    def __init__(
        self,
        n: int,
        regularization: str = "vu_C",
        C: float | int = 1,
        incr_opt: str = "inverse",
        polynomial_basis: str = "monomials",
        inv_opt: str = "fpd_inv",
    ):
        """Initialize the DyCF model.

        Parameters
        ----------
        n: int
            The degree of polynomials, usually set between 2 and 8.
        regularization: str, optional, by default "vu_C"
            The regularization method to use for the score. Options include:
            - "vu" (score divided by d^{3p/2}),
            - "vu_C" (score divided by d^{3p/2}/C),
            - "comb" (score divided by comb(p+d, d)),
            - "constant" (score divided by C).
        C: float | int, optional, by default 1
            The constant factor used in the "vu_C" or "constant" regularization.
        incr_opt: str, optional, by default "inverse"
            The incrementation option to use for updating the moments matrix. Options include:
            - "inverse" (use the inverse of the moments matrix),
            - "sherman" (use the Sherman-Morrison formula iteratively),
            - "woodbury" (use the Woodbury matrix identity).
        polynomial_basis: str, optional, by default "monomials"
            The polynomial basis to use for the moments matrix. Options include:
            - "monomials" (standard monomials),
            - "chebyshev_t1" (Chebyshev polynomials of the first kind),
            - "chebyshev_t2" (Chebyshev polynomials of the second kind),
            - "chebyshev_u" (Chebyshev polynomials of the u kind),
            - "legendre" (Legendre polynomials).
        inv_opt: str, optional, by default "fpd_inv"
            The inversion option to use for the moments matrix. Options include:
            - "inv" (use the standard inverse),
            - "pinv" (use the pseudo-inverse),
            - "pd_inv" (use the positive definite inverse),
            - "fpd_inv" (use the cholesky decomposition inverse).
        """
        # Initialize the parameters
        self.n = n
        self.regularization = regularization
        self.C = C
        # Initialize the MomentsMatrix
        self.moments_matrix = MomentsMatrix(n, incr_opt=incr_opt, polynomial_basis=polynomial_basis, inv_opt=inv_opt)
        # Initialize the variables set during the fit method
        self.d = 0
        self.regularizer = None

    def fit(self, x: np.ndarray):
        self.assert_shape_unfitted(x)
        self.d = x.shape[1]
        self.moments_matrix.fit(x)
        self.regularizer = IMPLEMENTED_REGULARIZATION_OPTIONS[self.regularization].regularizer(self.n, self.d, self.C)
        return self

    def update(self, x, sym=True):
        self.assert_shape_fitted(x)
        self.moments_matrix.update(x, sym=sym)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        return self.moments_matrix.score_samples(x.reshape(-1, self.d)) / self.regularizer

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
        return {
            "n": self.n,
            "regularization": self.regularization,
            "C": self.C,
            "moments_matrix": self.moments_matrix.save_model(),
            "d": self.d,
            "regularizer": self.regularizer,
        }

    def load_model(self, model_dict: dict):
        self.n = model_dict["n"]
        self.regularization = model_dict["regularization"]
        self.C = model_dict["C"]
        self.moments_matrix.load_model(model_dict["moments_matrix"])
        self.d = model_dict["d"]
        self.regularizer = model_dict["regularizer"]
        return self

    def is_fitted(self):
        return self.moments_matrix.is_fitted() and self.d > 0 and self.regularizer is not None

    def method_name(self):
        return "DyCF"


class DyCG(BaseDetector):
    """
    Dynamical Christoffel Growth

    Attributes
    ----------
    degrees: ndarray, optional
        The degrees of at least two DyCF models inside (default is np.array([2, 8]))
    models: list[DyCF]
        The list of DyCF models with different degrees n.
    d: int
        The dimension of the input data. 0 if not fitted.

    Methods
    -------
    See BaseDetector methods
    """

    @check_types({"degrees": lambda x: len(x) > 1})
    def __init__(self, degrees: np.ndarray = np.array([2, 8]), **dycf_kwargs):
        """Initialize the DyCG model.

        Parameters
        ----------
        degrees: np.ndarray, optional
            The degrees of at least two DyCF models inside (default is np.array([2, 8]))
        dycf_kwargs: dict, optional
            Additional keyword arguments to pass to the DyCF constructor (other than n).
        """
        # Initialize the parameters
        self.degrees = degrees
        self.models = [DyCF(n=n, **dycf_kwargs) for n in self.degrees]
        # Initialize the dimension set during the fit method
        self.d = 0

    def fit(self, x):
        self.assert_shape_unfitted(x)
        self.d = x.shape[1]
        for model in self.models:
            model.fit(x)
        return self

    def update(self, x: np.ndarray):
        self.assert_shape_fitted(x)
        for model in self.models:
            model.update(x)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        score = np.zeros((len(x), 1))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            scores = np.array([m.score_samples(d_)[0] for m in self.models])
            s_diff = np.diff(scores) / np.diff(self.degrees)
            score[i] = np.mean(s_diff)
        return score

    def decision_function(self, x):
        self.assert_shape_fitted(x)
        return -1 * self.score_samples(x)

    def predict(self, x):
        self.assert_shape_fitted(x)
        return np.where(self.decision_function(x) < 0, -1, 1)

    def eval_update(self, x):
        self.assert_shape_fitted(x)
        evals = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            evals[i] = self.decision_function(d_)
            self.update(d_)
        return evals

    def predict_update(self, x):
        self.assert_shape_fitted(x)
        preds = np.zeros(len(x))
        for i, d in enumerate(x):
            d_ = d.reshape(1, -1)
            preds[i] = self.predict(d_)
            self.update(d_)
        return preds

    def save_model(self):
        return {"degrees": self.degrees.tolist(), "p": self.d, "models": [dycf_model.save_model() for dycf_model in self.models]}

    def load_model(self, model_dict: dict):
        self.degrees = np.array(model_dict["degrees"])
        self.d = model_dict["p"]
        for i, dycf_model_dict in enumerate(model_dict["models"]):
            self.models[i].load_model(dycf_model_dict)
        return self

    def is_fitted(self):
        return all(model.is_fitted() for model in self.models) and self.d > 0

    def method_name(self):
        return "DyCG"
