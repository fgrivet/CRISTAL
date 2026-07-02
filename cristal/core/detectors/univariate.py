"""Contains our Univariate version of the Christoffel function based outlier detection algorithm."""

from typing import Literal, cast

from ...config.detector_config import StaticDetectorConfig
from ...types import ArrayLike, DTypeLike
from .base_detector import BaseCGDetector, BaseDetector


class UCF(BaseDetector[ArrayLike, DTypeLike, StaticDetectorConfig]):
    """Class to compute our Univariate version of the Christoffel function based outlier detection algorithm.

    Attributes
    ----------
    X_train: ArrayLike | None
        The training data of shape (N_samples_train, d), set during :func:`fit`.

    See Also
    --------
    cristal.core.detectors.base_detector.BaseDetector : For more attributes.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.
    """

    def __init__(self, n: int | Literal["auto"], config: StaticDetectorConfig[ArrayLike, DTypeLike] = StaticDetectorConfig(), z: int = 0):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        n : int | Literal["auto"]
            The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`.
        config : StaticDetectorConfig[ArrayLike, DTypeLike], optional
            The configuration of the model, by default StaticDetectorConfig().
        z : int, optional
            The value on which to evaluate the univariate CF :math:`\\Lambda^{\\nu_{x}}_{n}(z)`, by default 0.
        """
        if z < 0:
            raise ValueError(f"z must be >= 0. Got {z}.")
        super().__init__(n, config)
        self.intrinsic_dim = 1
        self.z = z

        # Variables defined during fitting specific to UCF
        self.X_train: ArrayLike | None = None  #: The training data of shape (N_samples_train, d).

    def _compute_scores(self, component_support: ArrayLike, component_X: ArrayLike) -> ArrayLike:
        """Compute the scores for each sample with the formula :math:`\\phi(0)^T G^{-1}(x) \\phi(0)`.

        Parameters
        ----------
        component_support : ArrayLike
            If :const:`solver != qr`: :math:`G` of shape (N_samples_test, n+1, n+1). \n
            If :const:`solver = qr`: :math:`V` of shape (N_samples_test, N_samples_train, n+1) such that :math:`G = (V^T V) N\\_samples\\_train`.
        component_X : ArrayLike
            :math:`\\phi(0)` of shape (N_samples_test, n+1, 1).

        Returns
        -------
        ArrayLike
            The scores for each sample of shape (N_samples_test,)

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if self.N is None:
            raise ValueError("Model must be fitted before computing scores.")
        return self.config.solver(component_support, component_X, self.N)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute for each sample the components needed to calculate the scores.

        Parameters
        ----------
        X : ArrayLike
            The matrix containing the samples of shape (N_samples_test, d).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            :math:`G` of shape (N_samples_test, n+1, n+1) or :math:`V` of shape (N_samples_test, N_samples_train, n+1) depending on :attr:`solver`, :math:`\\phi(0)` of shape (N_samples_test, n+1, 1).

        Raises
        ------
        ValueError
            If n is not a positive integer.

            If the model is not fitted.
        """
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.d is None or self.d <= 0:
            raise ValueError(f"d must be a positive integer. Got {self.d}.")
        if self.X_train is None:
            raise ValueError("Model must be fitted before computing components.")

        N_test = len(X)

        # QR: compute only V
        if self.config.solver.solver == "qr":
            # Compute the distances between all points in X and in X_train
            D = self.config.distance(X, self.X_train)
            # Compute the polynomial basis v(x) for each distance in D
            V = self.config.polynomial_basis.vandermonde_1d(D, self.n, self.d)
            # Construct the vector v of shape (N_samples_test, n+1, 1) such that z = v^T @ (V^T @ V / N)^-1 @ v
            v = self.config.polynomial_basis.make_v(self.n, X.dtype, z=self.z, d=self.d)
            v = self.config.backend.broadcast(v, (N_test, self.n + 1)).reshape((N_test, self.n + 1, 1))
            return V, v
        # Not QR: compute G directly using batch on X_train to reduce memory consumption
        else:
            G = self.config.backend.zeros((N_test, self.n + 1, self.n + 1))
            for X_batch in self.config.storage(self.X_train):
                # Compute the distances between all points in X and in X_batch
                D = self.config.distance(X, X_batch)
                # Compute the polynomial basis v(x) for each distance in D
                V = self.config.polynomial_basis.vandermonde_1d(D, self.n, self.d)
                # Compute the Gram matrix G
                G += self.config.backend.swap(V, 1, 2) @ V
            G /= len(self.X_train)
            # Construct the vector v of shape (N_samples_test, n+1, 1) such that z = v^T @ G^-1 @ v
            v = self.config.polynomial_basis.make_v(self.n, X.dtype, z=self.z, d=self.d)
            v = self.config.backend.broadcast(v, (N_test, self.n + 1)).reshape((N_test, self.n + 1, 1))

            return G, v

    def _crop_components(self, component_support: ArrayLike, component_X: ArrayLike, n_crop: int) -> tuple[ArrayLike, ArrayLike]:
        """Crop the computed components to calculate the scores for a degree :math:`n_crop \\leq n`.

        Parameters
        ----------
        component_support : ArrayLike
            If :const:`solver != qr`: :math:`G` of shape (N_samples_test, n+1, n+1). \n
            If :const:`solver = qr`: :math:`V` of shape (N_samples_test, N_samples_train, n+1) such that :math:`G = (V^T V) N\\_samples\\_train`.
        component_X : ArrayLike
            :math:`\\phi(0)` of shape (N_samples_test, n+1, 1).
        n_crop : int
            The maximum polynomial basis degree. Must be lower or equal to :attr:`n`.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            :math:`G` of shape (N_samples_test, n_crop+1, n_crop+1) or :math:`V` of shape (N_samples_test, N_samples_train, n_crop+1) depending on :attr:`solver`, :math:`\\phi(0)` of shape (N_samples_test, n_crop+1, 1).

        Raises
        ------
        ValueError
            If n_crop is not a positive integer, lower or equal to :attr:`n`.
        """
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if n_crop > self.n:
            raise ValueError(f"n ({n_crop}) must be lower or equal than maximum degree ({self.n}).")

        if self.config.solver.solver == "qr":
            # component_support shape is (N_test, N, n+1) with the polynomial basis of each distance in D (thus N_test, N)
            # component_X shape is (N_test, n+1, 1) the vector of size n+1 to apply to M for each testing point
            return component_support[:, :, : n_crop + 1], component_X[:, : n_crop + 1]
        return component_support[:, : n_crop + 1, : n_crop + 1], component_X[:, : n_crop + 1]

    def update(self, X: ArrayLike, online: Literal["constant", "increment"]):
        if self.X_train is None:
            raise ValueError("Model must be fitted before being updated.")
        if online == "constant":
            # Concatenate the previous training set with the new one
            self.X_train = self.config.backend.concat([self.X_train, X], axis=0)
        else:  # Increment
            # Replace the oldest data in the training set by the new one (FIFO)
            self.X_train = self.config.backend.concat([self.X_train[len(X) :], X], axis=0)
        return self

    def fit(self, X: ArrayLike) -> BaseDetector:
        """Fit the model to the data :attr:`X`.
        Compute the :attr:`threshold` used for prediction.

        Parameters
        ----------
        X : ArrayLike
            The training data of shape (N_samples_train, d).

        Returns
        -------
        BaseDetector
            The fitted model.

        Raises
        ------
        ValueError
            If :attr:`X` is not a 2D ArrayLike.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")

        # Define The number of training data and the diension of training data
        N, d = X.shape

        # Preprocess the data
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.fit_transform(X)  # type: ignore
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)

        # Save the information on training data
        self.N = cast(int, N)
        self.d = cast(int, d)
        self.X_train = X

        # Define the degree if set to auto using n = N**(1/(2+d)) with d=1 because univariate
        self.n = self._compute_n(self.n, self.N)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return self.N is not None and self.d is not None and self.X_train is not None and self.threshold is not None and isinstance(self.n, int)


class UCG(BaseCGDetector):
    """Class to compute our Univariate version of the Christoffel function based outlier detection scores, and predictions based on the growth of scores as the degree :attr:`n` increases.

    See Also
    --------
    UCF, cristal.core.detectors.base_detector.BaseDetector : For more attributes.
    """

    def __init__(self, n_list: list[int], config: StaticDetectorConfig = StaticDetectorConfig(), z: int = 0, *args, **kwargs):
        super().__init__(UCF, n=n_list, config=config, z=z, *args, **kwargs)


class MTSUCF(UCF):
    """Class to compute our Univariate version of the Christoffel function based outlier detection algorithm for multivariate time series.
    It extends :class:`UCF` to 3D input data.

    Attributes
    ----------
    X_train: ArrayLike | None
        The training data of shape (N_windows, window_size, d), set during :func:`fit`.

    See Also
    --------
    UCF, cristal.core.detectors.base_detector.BaseDetector : For more attributes.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.
    """

    def _preprocess_test_data(self, X: ArrayLike) -> ArrayLike:
        """Transform the data to the ArrayLike type, then preprocess it.

        Parameters
        ----------
        X : ArrayLike
            The data points of shape (N_windows, window_size, d).

        Returns
        -------
        ArrayLike
            The preprocess data points. The shape depends on the  preprocessing step.

        Raises
        ------
        ValueError
            If X is not a 3D ArrayLike.

            If the model is not fitted.

            If the degree :attr:`n` is not positive.

            If the size of the windows of the testing data `window_size` is not the same as the one of the training data :attr:`window_size`.

            If the number of paramters of the testing data `d` is not the same as the one of the training data :attr:`d`.
        """
        if X.ndim != 3:
            raise ValueError(f"X must be a 3D ArrayLike of shape (N_samples_test, window_size={self.window_size}, d={self.d}). Got {X.shape}.")
        if not self.is_fitted():
            raise ValueError("Model must be fitted before scoring samples.")
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        window_size = X.shape[1]
        if window_size != self.window_size:
            raise ValueError(f"Window size mismatch between training ({self.window_size}) and test ({window_size}).")
        d = X.shape[2]
        if d != self.d:
            raise ValueError(f"Dimension mismatch between training ({self.d}) and test ({d}).")

        # Preprocess the data
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.transform(X)  # type: ignore

        return X

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute for each sample the components needed to calculate the scores.

        Parameters
        ----------
        X : ArrayLike
            The matrix containing the samples of shape (N_samples_test, window_size, d).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            :math:`G` of shape (N_samples_test, n+1, n+1) or :math:`V` of shape (N_samples_test, N_samples_train, n+1) depending on :attr:`solver`, :math:`\\phi(0)` of shape (N_samples_test, n+1, 1).

        Raises
        ------
        ValueError
            If n is not a positive integer.

            If the model is not fitted.
        """
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.d is None or self.d <= 0:
            raise ValueError(f"d must be a positive integer. Got {self.d}.")
        if self.X_train is None:
            raise ValueError("Model must be fitted before computing components.")

        N_test = len(X)

        # QR: compute only V
        if self.config.solver.solver == "qr":
            # Compute the distances between all points in X and in X_train
            D = self.config.backend.zeros((N_test, len(self.X_train)))
            for idx_signal in range(X.shape[2]):
                D += self.config.distance(X[:, :, idx_signal], self.X_train)
            # Compute the polynomial basis v(x) for each distance in D
            V = self.config.polynomial_basis.vandermonde_1d(D, self.n, self.d)
            # Construct the vector v of shape (N_samples_test, n+1, 1) such that z = v^T @ (V^T @ V / N)^-1 @ v
            v = self.config.polynomial_basis.make_v(self.n, X.dtype, z=self.z, d=self.d)
            v = self.config.backend.broadcast(v, (N_test, self.n + 1)).reshape((N_test, self.n + 1, 1))
            return V, v
        # Not QR: compute G directly using batch on X_train to reduce memory consumption
        else:
            G = self.config.backend.zeros((N_test, self.n + 1, self.n + 1))
            for X_batch in self.config.storage(self.X_train):
                # Compute the distances between all points in X and in X_batch
                D = self.config.backend.zeros((N_test, len(X_batch)))
                for idx_signal in range(X.shape[2]):
                    D += self.config.distance(X[:, :, idx_signal], X_batch[:, :, idx_signal])
                # Compute the polynomial basis v(x) for each distance in D
                V = self.config.polynomial_basis.vandermonde_1d(D, self.n, self.d)
                # Compute the Gram matrix G
                G += self.config.backend.swap(V, 1, 2) @ V
            G /= len(self.X_train)
            # Construct the vector v of shape (N_samples_test, n+1, 1) such that z = v^T @ G^-1 @ v
            v = self.config.polynomial_basis.make_v(self.n, X.dtype, z=self.z, d=self.d)
            v = self.config.backend.broadcast(v, (N_test, self.n + 1)).reshape((N_test, self.n + 1, 1))

            return G, v

    def fit(self, X: ArrayLike) -> BaseDetector:
        """Fit the model to the data :attr:`X`.
        Compute the :attr:`threshold` used for prediction.

        Parameters
        ----------
        X : ArrayLike
            The training data of shape (N_samples_train, window_size, d).

        Returns
        -------
        BaseDetector
            The fitted model.

        Raises
        ------
        ValueError
            If :attr:`X` is not a 3D ArrayLike.
        """
        if X.ndim != 3:
            raise ValueError(f"X must be a 3D ArrayLike. Got {X.shape}.")

        # Define The number of training data and the diension of training data
        N, window_size, d = X.shape

        # Preprocess the data
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.fit_transform(X)  # type: ignore
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)

        # Save the information on training data
        self.N = cast(int, N)
        self.window_size = cast(int, window_size)
        self.d = cast(int, d)
        self.X_train = X

        # Define the degree if set to auto using n = N**(1/(2+d)) with d=1 because univariate
        self.n = self._compute_n(self.n, self.N)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d)

        assert self.is_fitted(), "Error during fitting."
        return self


class MTSUCG(BaseCGDetector):
    """Class to compute our Univariate version of the Christoffel function based outlier detection scores for multivariate time series, and predictions based on the growth of scores as the degree :attr:`n` increases.

    See Also
    --------
    MTSUCF, cristal.core.detectors.base_detector.BaseDetector : For more attributes.
    """

    def __init__(self, n_list: list[int], config: StaticDetectorConfig = StaticDetectorConfig(), z: int = 0, *args, **kwargs):
        super().__init__(MTSUCF, n=n_list, config=config, z=z, *args, **kwargs)
