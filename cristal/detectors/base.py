"""
Base class for outlier detection methods.
"""

import copy
from abc import ABC, abstractmethod
from typing import Self

import numpy as np


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
        d = self.__dict__.get("d")
        if x.ndim != 2 or x.shape[1] != d:
            raise ValueError(f"The expected array shape: (N, {d}) do not match the given array shape: {x.shape}")


class NotFittedError(Exception):
    """This exception is raised when a :class:`BaseDetector` is used before it has been fitted."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message

    def __str__(self):
        return f"NotFittedError: {self.message}"
