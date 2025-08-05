"""
Christoffel Function and Growth Detector classes
"""

import logging

import numpy as np

from cristal.helper_classes.base import BaseDetector
from cristal.helper_classes.incrementers import IMPLEMENTED_INCREMENTERS_OPTIONS
from cristal.helper_classes.inversion import IMPLEMENTED_INVERSION_OPTIONS
from cristal.helper_classes.moment_matrix import MomentsMatrix
from cristal.helper_classes.polynomial_basis import IMPLEMENTED_POLYNOMIAL_BASIS
from cristal.helper_classes.regularizer import IMPLEMENTED_REGULARIZATION_OPTIONS
from cristal.utils.type_checking import check_all_int, check_types, positive_integer

logger = logging.getLogger("CRISTAL")


class DyCF(BaseDetector):
    """
    Dynamical Christoffel Function

    Attributes
    ----------
    n: int
        The degree of polynomials, usually set between 2 and 8.
    regularization: IMPLEMENTED_REGULARIZATION_OPTIONS
        The regularization method to use for the score.
    C: float | int
        The constant factor used in the "vu_C" or "constant" regularization.
    moments_matrix: MomentsMatrix
        The moments matrix object.
    d: int
        The dimension of the input data. 0 if not fitted.
    regularizer: float | int | None
        The regularization factor computed based on the degree and dimension. None if not fitted.
    """

    @check_types(
        {
            "n": positive_integer,
        }
    )
    def __init__(
        self,
        n: int,
        regularization: IMPLEMENTED_REGULARIZATION_OPTIONS = IMPLEMENTED_REGULARIZATION_OPTIONS.VU_C,
        C: float | int = 1,
        polynomial_basis: IMPLEMENTED_POLYNOMIAL_BASIS = IMPLEMENTED_POLYNOMIAL_BASIS.MONOMIALS,
        inv_opt: IMPLEMENTED_INVERSION_OPTIONS = IMPLEMENTED_INVERSION_OPTIONS.FPD_INV,
        incr_opt: IMPLEMENTED_INCREMENTERS_OPTIONS = IMPLEMENTED_INCREMENTERS_OPTIONS.INVERSE,
    ):
        """Initialize the DyCF model.

        Parameters
        ----------
        n: int
            The degree of polynomials, usually set between 2 and 8.
        regularization: IMPLEMENTED_REGULARIZATION_OPTIONS, optional, by default IMPLEMENTED_REGULARIZATION_OPTIONS.VU_C
            The regularization method to use to divide the scores.
            See :attr:`~cristal.helper_classes.regularizer.IMPLEMENTED_REGULARIZATION_OPTIONS` for more details.
        C: float | int, optional, by default 1
            The constant factor used in the :const:`"vu_C"` or :const:`"constant"` regularization.
        polynomial_basis: IMPLEMENTED_POLYNOMIAL_BASIS, optional, by default IMPLEMENTED_POLYNOMIAL_BASIS.MONOMIALS
            The polynomial basis to use for the moments matrix.
            See :attr:`~cristal.helper_classes.polynomial_basis.IMPLEMENTED_POLYNOMIAL_BASIS` for more details.
        inv_opt: IMPLEMENTED_INVERSION_OPTIONS, optional, by default IMPLEMENTED_INVERSION_OPTIONS.FPD_INV
            The inversion option to use for the moments matrix.
            See :attr:`~cristal.helper_classes.inversion.IMPLEMENTED_INVERSION_OPTIONS` for more details.
        incr_opt: IMPLEMENTED_INCREMENTERS_OPTIONS, optional, by default IMPLEMENTED_INCREMENTERS_OPTIONS.INVERSE
            The incrementation option to use for updating the moments matrix.
            See :attr:`~cristal.helper_classes.incrementers.IMPLEMENTED_INCREMENTERS_OPTIONS` for more details.
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
        self.regularizer = self.regularization.value.regularizer(self.n, self.d, self.C)
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
        DyCF
            The updated DyCF instance.
        """
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
        """Saves the DyCF model as a dictionary.

        Returns
        -------
        dict
            The DyCF model represented as a dictionary, with keys:
                - "n": int, the degree of the polynomial basis.
                - "regularization": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "moments_matrix": dict, the saved moments matrix.
                - "d": int, the dimension of the input data.
                - "regularizer": float | int | None, the regularization factor.
        """
        return {
            "n": self.n,
            "regularization": self.regularization.name,
            "C": self.C,
            "moments_matrix": self.moments_matrix.save_model(),
            "d": self.d,
            "regularizer": self.regularizer,
        }

    def load_model(self, model_dict: dict):
        """Loads the DyCF model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the DyCF parameters returned by :func:`save_model()`:
                - "n": int, the degree of the polynomial basis.
                - "regularization": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "moments_matrix": dict, the saved moments matrix.
                - "d": int, the dimension of the input data.
                - "regularizer": float | int | None, the regularization factor.

        Returns
        -------
        DyCF
            The loaded DyCF instance with updated parameters.
        """
        self.n = model_dict["n"]
        self.regularization = IMPLEMENTED_REGULARIZATION_OPTIONS[model_dict["regularization"]]
        self.C = model_dict["C"]
        self.moments_matrix.load_model(model_dict["moments_matrix"])
        self.d = model_dict["d"]
        self.regularizer = model_dict["regularizer"]
        return self

    def is_fitted(self):
        return self.moments_matrix.is_fitted() and self.d > 0 and self.regularizer is not None

    def method_name(self) -> str:
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
        """Saves the DyCG model as a dictionary.

        Returns
        -------
        dict
            The DyCG model represented as a dictionary, with keys:
                - "degrees": list[int], the degrees of the polynomial basis.
                - "d": int, the dimension of the input data.
                - "models": list[dict], the saved DyCF models.
        """
        return {"degrees": self.degrees.tolist(), "d": self.d, "models": [dycf_model.save_model() for dycf_model in self.models]}

    def load_model(self, model_dict: dict):
        """Loads the DyCG model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the DyCG model parameters returned by :func:`save_model()`:
                - "degrees": list[int], the degrees of the polynomial basis.
                - "d": int, the dimension of the input data.
                - "models": list[dict], the saved DyCF models.

        Returns
        -------
        DyCG
            The loaded DyCG model with updated parameters.
        """
        self.degrees = np.array(model_dict["degrees"])
        self.d = model_dict["d"]
        for i, dycf_model_dict in enumerate(model_dict["models"]):
            self.models[i].load_model(dycf_model_dict)
        return self

    def is_fitted(self):
        return all(model.is_fitted() for model in self.models) and self.d > 0

    def method_name(self) -> str:
        return "DyCG"


class BaggingDyCF(DyCF):
    """
    Bagging DyCF model.

    Attributes
    ----------
    n_estimators: int
        The number of DyCF models in the ensemble.
    models: list[DyCF]
        The list of DyCF models in the ensemble.
    d: int
        The dimension of the input data. 0 if not fitted.
    """

    @check_types({"n_values": check_all_int, "n_estimators": positive_integer})
    def __init__(self, n_values: int | list, n_estimators: int = 10, **dycf_kwargs):
        """Initialize the BaggingDyCF model.

        Parameters
        ----------
        n_values: int | list[int]
            The degree of the polynomial basis or a list of degrees for the DyCF models.
            If a single integer is provided, it will be used for all models.
            If a list is provided, its length should match n_estimators, and each model will have a different degree.
        n_estimators: int, optional
            The number of DyCF models in the ensemble (default is 10).
        dycf_kwargs: dict, optional
            Additional keyword arguments to pass to the DyCF constructor.
        """
        # Check if n_values is a single integer or a list of size n_estimators
        if isinstance(n_values, int):
            n_values = [n_values] * n_estimators
        elif len(n_values) != n_estimators:
            raise ValueError(
                f"n_values must be an int or a list of length {n_estimators}. Provided: {len(n_values)} values. Expected: {n_estimators} values."
            )
        # Initialize the parameters
        self.n_values: list[int] = n_values
        self.n_estimators = n_estimators
        self.models = [DyCF(n=n_values[i], **dycf_kwargs) for i in range(n_estimators)]
        self.d = 0
        # Initialize the mean degree of the models (for plotting purposes)
        self.n: float = np.mean(n_values)  # type: ignore
        logger.debug("BaggingDyCF initialized with %d models of mean degrees: %d", n_estimators, self.n)
        # Initialize the regularization, C and regularizer (for plotting purposes)
        self.regularization = self.models[0].regularization
        self.C = self.models[0].C
        # Initialize the regularizer to None, it will be set during the fit method
        self.regularizer = None

    @check_types({"n_samples": lambda x: x is None or positive_integer(x)})
    def fit(self, x: np.ndarray, n_samples: int | None = None):
        """Generate n_estimators DyCF models and fit them on the data.

        Parameters
        ----------
        x : np.ndarray (N, d)
            The input data to fit the models on.
        n_samples : int | None, optional
            The number of samples to use for fitting each model. If None or greater than N, all samples are used, by default None.

        Returns
        -------
        BaggingDyCF
            The fitted BaggingDyCF instance.
        """
        self.assert_shape_unfitted(x)
        self.d = x.shape[1]
        all_train_data = x.copy()
        for model in self.models:
            if n_samples is None or n_samples >= len(all_train_data):
                train_data = all_train_data
            else:
                np.random.shuffle(all_train_data)
                train_data = all_train_data[:n_samples]
            model.fit(train_data)
        self.regularizer = self.regularization.value.regularizer(self.n, self.d, self.C)
        return self

    def update(self, x: np.ndarray, sym: bool = True):
        self.assert_shape_fitted(x)
        for model in self.models:
            model.update(x, sym=sym)
        return self

    def score_samples(self, x):
        self.assert_shape_fitted(x)
        x_reshaped = x.reshape(-1, self.d)
        model_scores = np.array([model.score_samples(x_reshaped) for model in self.models])
        scores = np.mean(model_scores, axis=0)
        return scores

    def save_model(self):
        """Saves the BaggingDyCF model as a dictionary.

        Returns
        -------
        dict
            The BaggingDyCF model represented as a dictionary, with keys:
                - "n_values": list[int], the degrees of the polynomial basis for each DyCF model.
                - "n_estimators": int, the number of DyCF models in the ensemble.
                - "d": int, the dimension of the input data.
                - "regularization": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "regularizer": float | int | None, the regularization factor.
                - "models": list[dict], the saved DyCF models.
        """
        return {
            "n_values": self.n_values,
            "n_estimators": self.n_estimators,
            "d": self.d,
            "regularization": self.regularization.name,
            "C": self.C,
            "regularizer": self.regularizer,
            "models": [model.save_model() for model in self.models],
        }

    def load_model(self, model_dict: dict):
        """Loads the BaggingDyCF model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the BaggingDyCF model parameters returned by :func:`save_model()`:
                - "n_values": list[int], the degrees of the polynomial basis for each DyCF model.
                - "n_estimators": int, the number of DyCF models in the ensemble.
                - "d": int, the dimension of the input data.
                - "regularization": str, the regularization method used.
                - "C": float | int, the constant factor used in the regularization.
                - "regularizer": float | int | None, the regularization factor.
                - "models": list[dict], the saved DyCF models.

        Returns
        -------
        BaggingDyCF
            The loaded BaggingDyCF model with updated parameters.
        """
        self.n_values = model_dict["n_values"]
        self.n_estimators = model_dict["n_estimators"]
        self.d = model_dict["d"]
        self.regularization = IMPLEMENTED_REGULARIZATION_OPTIONS[model_dict["regularization"]]
        self.C = model_dict["C"]
        self.regularizer = model_dict["regularizer"]
        for i, model_dict_i in enumerate(model_dict["models"]):
            self.models[i].load_model(model_dict_i)
        return self

    def is_fitted(self):
        return all(model.is_fitted() for model in self.models) and self.d > 0

    def method_name(self) -> str:
        return "BaggingDyCF"
