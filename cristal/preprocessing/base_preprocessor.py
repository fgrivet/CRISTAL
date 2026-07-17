"""Base class for all preprocessors."""

from abc import ABC, abstractmethod
from typing import Generic

from sklearn.base import BaseEstimator, TransformerMixin

from ..commons.base_commons import BaseCommons
from ..types import ArrayLike, DTypeLike


# pylint: disable=unused-variable
class BasePreprocessor(ABC, BaseCommons, Generic[ArrayLike, DTypeLike], BaseEstimator, TransformerMixin):
    """Abstract base class for all preprocessors."""

    @abstractmethod
    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "BasePreprocessor":
        """Fit the preprocessor to the data.

        Parameters
        ----------
        X : ArrayLike
            The input data to fit.
        y : ArrayLike | None, optional
            Ignored.

        Returns
        -------
        BasePreprocessor
            The fitted Preprocessor.
        """

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None, **fit_params) -> ArrayLike:
        """Fit and transform the data.

        Parameters
        ----------
        X : ArrayLike
            The input data.
        y : ArrayLike | None, optional
            Ignored.
        fit_params
            Ignored.

        Returns
        -------
        ArrayLike
            The transformed data.
        """
        return self.fit(X, y).transform(X)

    @abstractmethod
    def transform(self, X: ArrayLike) -> ArrayLike:
        """Apply the transformation.

        Parameters
        ----------
        X : ArrayLike
            The input data to transform.

        Returns
        -------
        ArrayLike
            The transformed data.
        """

    @abstractmethod
    def inverse_transform(self, X: ArrayLike) -> ArrayLike:
        """Undo the transformation.

        Parameters
        ----------
        X : ArrayLike
            The transformed data to inverse.

        Returns
        -------
        ArrayLike
            The data in its original space.
        """
