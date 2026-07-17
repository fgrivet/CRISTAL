"""MinMaxScaler for both 2D and 3D data."""

from typing import Generic

from ...backend.base_backend import Backend
from ...commons.base_commons import BaseCommons
from ...types import ArrayLike, DTypeLike
from ..base_preprocessor import BasePreprocessor


# pylint: disable=unused-variable
class MinMaxScaler(BasePreprocessor, BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Scale features to a given range.

    This scaler transforms each feature (column or channel) to the range specified by `feature_range`.
    For 2D input (N_points, N_features), scaling is applied per column (axis=0).
    For 3D input (N_window, window_size, N_channel), the first two dimensions are flattened
    and scaling is applied per channel (axis=0).

    .. version-added:: 0.0.3

    Parameters
    ----------
    feature_range : tuple (min, max), default=(-1, 1)
        The desired range of transformed data.

    Attributes
    ----------
    feature_range : tuple (min, max)
        The desired range of transformed data.
    min_ : ArrayLike of shape (N_features,) | None
        Minimum values for each feature/channel, set during :func:`fit`.
    max_ : ArrayLike of shape (N_features,) | None
        Maximum values for each feature/channel, set during :func:`fit`.
    n_features : int | None
        The number of features if the input data is 2D, or the window_size if the input data is 3D, set during :func:`fit`.
    n_channel : int | None
        The number of channels (if the input data is 3D), set during :func:`fit`.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    Examples
    --------
    >>> import numpy as np
    >>> X_2d = np.array([[1, 2], [3, 4], [5, 6]])
    >>> scaler = MinMaxScaler(feature_range=(0, 1))
    >>> X_scaled = scaler.fit_transform(X_2d)
    >>> X_original = scaler.inverse_transform(X_scaled)
    >>> np.allclose(X_2d, X_original)
    True

    >>> X_3d = np.random.rand(10, 5, 3)  # (N_window, window_size, N_channel)
    >>> scaler_3d = MinMaxScaler(feature_range=(-1, 1))
    >>> X_3d_scaled = scaler_3d.fit_transform(X_3d)
    >>> X_3d_original = scaler_3d.inverse_transform(X_3d_scaled)
    >>> np.allclose(X_3d, X_3d_original)
    True
    """

    requires = ["backend"]

    def __init__(self, feature_range: tuple[int, int] = (-1, 1)):
        """Class constructor.

        Define the attributes.

        Parameters
        ----------
        feature_range : tuple, optional
            The desired range of transformed data, by default (-1, 1)
        """
        if len(feature_range) != 2:
            raise ValueError(f"feature range must be a 2D tuple. Got {len(feature_range)}D.")
        if feature_range[0] > feature_range[1]:
            raise ValueError(f"feature range must be (min_, max_) with min_ <= max_. Got {feature_range}.")

        self.feature_range = feature_range
        """The desired range of transformed data."""

        # Variables defined during fitting
        self.min_: ArrayLike | None = None
        """Minimum values for each feature/channel, set during :func:`fit`."""
        self.max_: ArrayLike | None = None
        """Maximum values for each feature/channel, set during :func:`fit`."""
        self.n_features: int | None = None
        """The number of features if the input data is 2D, or the window_size if the input data is 3D, set during :func:`fit`."""
        self.n_channel: int | None = None
        """The number of channels (if the input data is 3D), set during :func:`fit`."""

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        """The backend to use for the computation."""

    def _reshape_input(self, X) -> tuple[ArrayLike, int, int | None]:
        """Reshape input to 2D for uniform processing.

        Parameters
        ----------
        X : _type_
            The input data. Either (N_points, N_features), or (N_window, window_size, N_channel).

        Returns
        -------
        tuple[ArrayLike, int, int | None]
            X reshaped, n_features, n_channel

        Raises
        ------
        ValueError
            If X is not 2D or 3D.
        """

        if X.ndim == 2:
            return X, X.shape[1], None
        elif X.ndim == 3:
            n_windows, n_channel = X.shape[1:]
            return X.reshape(-1, n_channel), n_windows, n_channel
        else:
            raise ValueError("Input must be 2D or 3D.")

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """
        Compute the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : ArrayLike of shape (N_points, N_features) or (N_window, window_size, N_channel)
            Input data.
        y : ArrayLike | None, optional
            Ignored.

        Returns
        -------
        BasePreprocessor
            The fitted scaler.

        Raises
        ------
        ValueError
            If the backed is not bound.
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the MinMaxScaler class before using it.")

        X_2d, self.n_features, self.n_channel = self._reshape_input(X)
        self.min_ = self.backend.min(X_2d, axis=0)
        self.max_ = self.backend.max(X_2d, axis=0)
        return self

    def transform(self, X: ArrayLike):
        """
        Scale features of X according to feature_range.

        Parameters
        ----------
        X : ArrayLike of shape (N_points, N_features) or (N_window, window_size, N_channel)
            Input data.

        Returns
        -------
        ArrayLike of same shape as X
            Scaled data.

        Raises
        ------
        ValueError
            If the model is not fitted.

            If the input data has invalid shape.
        """
        if self.max_ is None or self.min_ is None or self.n_features is None:
            raise ValueError("MinMaxScaler must be fitted before using it.")

        X_2d, n_features, n_channel = self._reshape_input(X)
        if n_features != self.n_features or n_channel != self.n_channel:
            raise ValueError(f"Invalid shape. Should be (N, {self.n_features}, {self.n_channel}). Got {X.shape}.")

        scale_ = self.max_ - self.min_
        scale_[scale_ == 0] = 1.0  # Avoid division by zero

        X_scaled = (X_2d - self.min_) / scale_ * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        if self.n_channel is None:
            return X_scaled
        return X_scaled.reshape((-1, self.n_features, self.n_channel))

    def inverse_transform(self, X: ArrayLike):
        """
        Undo the scaling of X according to feature_range.

        Parameters
        ----------
        X : ArrayLike of shape (N_points, N_features) or (N_window, window_size, N_channel)
            Input data.

        Returns
        -------
        ArrayLike of same shape as X
            Original data.

        Raises
        ------
        ValueError
            If the model is not fitted.

            If the input data has invalid shape.
        """
        if self.max_ is None or self.min_ is None or self.n_features is None:
            raise ValueError("MinMaxScaler must be fitted before using it.")

        X_2d, n_features, n_channel = self._reshape_input(X)
        if n_features != self.n_features or n_channel != self.n_channel:
            raise ValueError(f"Invalid shape. Should be (N, {self.n_features}, {self.n_channel}). Got {X.shape}.")

        scale_ = self.max_ - self.min_
        scale_[scale_ == 0] = 1.0  # Avoid division by zero

        X_original = (X_2d - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * scale_ + self.min_

        if self.n_channel is None:
            return X_original
        return X_original.reshape((-1, self.n_features, self.n_channel))
