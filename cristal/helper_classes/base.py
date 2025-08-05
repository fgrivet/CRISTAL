"""
Base classes for :class:`outlier detection methods <BaseDetector>`, :class:`plotters <BasePlotter>`,
:class:`inverters <BaseInverter>`, :class:`incrementers <BaseIncrementer>`, and :class:`polynomial basis functions <BasePolynomialBasis>`.

Also contains the :class:`NotFittedError` exception.
"""

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Self

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from cristal.helper_classes.moment_matrix import MomentsMatrix


__all__ = [
    "BaseDetector",
    "BasePlotter",
    "BaseIncrementer",
    "BaseInverter",
    "BasePolynomialBasis",
    "BaseRegularizer",
    "NotFittedError",
]


class BaseDetector(ABC):
    """
    Base class for outlier detection methods
    """

    @abstractmethod
    def fit(self, x: np.ndarray) -> Self:
        """Generates a model that fit dataset x.

        Parameters
        ----------
        x : np.ndarray (N, d)
            The input data to fit the model.

        Returns
        -------
        Self
            The fitted model.
        """

    @abstractmethod
    def update(self, x: np.ndarray) -> Self:
        """Updates the current model with instances in x.

        Parameters
        ----------
        x : np.ndarray (N', d)
            The input data to update the model.

        Returns
        -------
        Self
            The updated model.
        """

    @abstractmethod
    def score_samples(self, x: np.ndarray) -> np.ndarray:
        """Computes the outlier scores for the samples in x.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data to compute the scores.

        Returns
        -------
        np.ndarray (L,)
            The outlier scores for each sample.
        """

    @abstractmethod
    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Computes the decision function for the samples in x.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data to compute the decision function.

        Returns
        -------
        np.ndarray (L,)
            The decision function values for each sample.
        """

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicts the outlier labels for the samples in x.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data to predict the labels.

        Returns
        -------
        np.ndarray (L,)
            The predicted labels for each sample.
        """

    def fit_predict(self, x: np.ndarray) -> np.ndarray:
        """Fits the model with x and predicts the outlier labels for the samples in x.

        Parameters
        ----------
        x : np.ndarray (N, d)
            The input data to fit and predict.

        Returns
        -------
        np.ndarray (N,)
            The predicted labels for each sample.
        """
        return self.fit(x).predict(x)

    @abstractmethod
    def eval_update(self, x: np.ndarray) -> np.ndarray:
        """Evaluates the model on the samples in x and updates it with inliers.
        This method iterates over each sample in x.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data to evaluate and update the model.

        Returns
        -------
        np.ndarray (L,)
            The decision function values for each sample.
        """

    @abstractmethod
    def predict_update(self, x: np.ndarray) -> np.ndarray:
        """Predict the model on the samples in x and updates it with inliers.

        Parameters
        ----------
        x : np.ndarray (L, d)
            The input data to evaluate and update the model.

        Returns
        -------
        np.ndarray (L,)
            The predicted labels for each sample.
        """

    @abstractmethod
    def save_model(self) -> dict:
        """Saves the model as a dictionary.

        Returns
        -------
        dict
            The model represented as a dictionary.
        """

    @abstractmethod
    def load_model(self, model_dict: dict) -> Self:
        """Loads the model from a dictionary.

        Parameters
        ----------
        model_dict : dict
            The dictionary containing the model parameters returned by save_model().

        Returns
        -------
        Self
            The loaded model.
        """

    def copy(self) -> Self:
        """Returns a copy of the model.

        Returns
        -------
        Self
            A copy of the current model.
        """
        return copy.deepcopy(self)

    @abstractmethod
    def is_fitted(self) -> bool:
        """Checks if the model is fitted.

        Returns
        -------
        bool
            True if the model is fitted, False otherwise.
        """

    @abstractmethod
    def method_name(self) -> str:
        """Returns the name of the method.

        Returns
        -------
        str
            The name of the method.
        """

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
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError(f"The expected array shape: (N, d) do not match the given array shape: {x.shape}")

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
        if x.ndim != 2 or x.shape[1] != self.__dict__["d"]:
            raise ValueError(f"The expected array shape: (N, {self.__dict__['d']}) do not match the given array shape: {x.shape}")


class BasePlotter(ABC):
    """
    Base class for outlier detection results plotters. This requires that data is 2-dimensional.
    """

    @abstractmethod
    def levelset(
        self,
        x: np.ndarray,
        *,
        n_x1: int,
        n_x2: int,
        levels: list | None,
        percentiles: list | None,
        show: bool,
        save: bool,
        save_title: str,
        close: bool,
    ) -> None | tuple[Figure, Axes]:
        """Generates the levelset plot and optionally show, save and close.

        Parameters
        ----------
        x : np.ndarray
            The input data to plot, shape (n_samples, n_features=2).
        n_x1 : int
            Number of points along the first dimension for the grid.
        n_x2 : int
            Number of points along the second dimension for the grid.
        levels : list[int | float] | None
            The levels to plot.
        percentiles : list[int | float] | None
            The percentiles to compute and plot as levels.
        show : bool
            Whether to show the plot.
        save : bool
            Whether to save the plot to a file.
        save_title : str
            Title for the saved plot file.
        close : bool
            Whether to close the plot or return it after saving or showing.

        Returns
        -------
        None | tuple[Figure, Axes]
            The figure and axes objects if not closing the plot, otherwise None.
        """

    @abstractmethod
    def boundary(self, x, *, n_x1, n_x2, show, save, save_title, close) -> None | tuple[Figure, Axes]:
        """Generates the boundary plot and optionally show, save and close.

        Parameters
        ----------
        x : np.ndarray
            The input data to plot, shape (n_samples, n_features=2).
        n_x1 : int
            Number of points along the first dimension for the grid.
        n_x2 : int
            Number of points along the second dimension for the grid.
        show : bool
            Whether to show the plot.
        save : bool
            Whether to save the plot to a file.
        save_title : str
            Title for the saved plot file.
        close : bool
            Whether to close the plot or return it after saving or showing.

        Returns
        -------
        None | tuple[Figure, Axes]
            The figure and axes objects if not closing the plot, otherwise None.
        """


class BaseInverter(ABC):
    """
    Base class for matrix inversion methods.
    """

    @staticmethod
    @abstractmethod
    def invert(matrix: np.ndarray) -> np.ndarray:
        """Invert the given matrix.

        Parameters
        ----------
        matrix : np.ndarray (n, n)
            The input matrix to invert.

        Returns
        -------
        np.ndarray (n, n)
            The inverse of the input matrix.
        """


class BaseIncrementer(ABC):
    """
    Base class for moments matrix incrementers.
    """

    update_moments_matrix: bool = False  #: Whether the method updates the moments matrix during incrementing

    @staticmethod
    @abstractmethod
    def increment(mm: "MomentsMatrix", x: np.ndarray, n: int, inv_class: type[BaseInverter], sym: bool = True):
        """Increment the inverse moments matrix (and possibly the moments matrix).

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


class BasePolynomialBasis(ABC):
    """
    Base class for polynomial basis functions.
    """

    @staticmethod
    @abstractmethod
    def func(x: np.ndarray, monomials_matrix: np.ndarray) -> np.ndarray:
        """Apply the polynomial basis function to the input data.

        Parameters
        ----------
        x : np.ndarray (1, d) or (d,)
            The input data.
        monomials_matrix : np.ndarray (s(n), d)
            The monomials matrix. Should be of shape (s(n), d) where s(n) is the number of monomials and d is the dimension of the input data.

        Returns
        -------
        np.ndarray (s(n), d)
            The transformed input data.
        """


class BaseRegularizer(ABC):
    """
    Base class for regularization methods.
    """

    @staticmethod
    @abstractmethod
    def regularizer(n: int | float, d: int, C: float | int) -> float:
        """Compute the regularization value.

        Parameters
        ----------
        n : int
            The polynomial basis maximum degree.
        d : int
            The dimension of the input data.
        C : float | int
            A constant used in the regularization computation.

        Returns
        -------
        float
            The computed regularization value.
        """


class NotFittedError(Exception):
    """This exception is raised when a BaseDetector is used before it has been fitted."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"NotFittedError: {self.message}"
