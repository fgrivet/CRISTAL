from typing import Literal, cast

from ...config.detector_config import StaticDetectorConfig
from ...types import ArrayLike, DTypeLike
from .base_detector import BaseCGDetector, BaseDetector


class UCF(BaseDetector[ArrayLike, DTypeLike, StaticDetectorConfig]):
    def __init__(self, n: int | Literal["auto"], config: StaticDetectorConfig[ArrayLike, DTypeLike]):
        super().__init__(n, config)
        self.intrinsic_dim = 1

        # Variables defined during fitting specific to UCF
        self.X_train: ArrayLike | None = None  #: The training data

    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike:
        # Here component_support is the vandermonde matrix for each point and component_x the make_v depending on the polynomial basis
        if self.N is None:
            raise ValueError("Model must be fitted before computing scores.")
        return self.config.solver(component_support, component_x, self.N)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.d is None or self.d <= 0:
            raise ValueError(f"d must be a positive integer. Got {self.d}.")

        N_test = len(X)

        # Compute the distances between all points in X and in X_train
        D = self.config.distance(X, self.X_train)
        # Compute the polynomial basis v(x) for each distance in D
        M = self.config.polynomial_basis.vandermonde_1d(D, self.n, self.d)
        # Construct the vector V such that Z = V^T @ (M^T @ M)^-1 @ V
        V = self.config.polynomial_basis.make_v(self.n, int)
        # Expands v to have dimensions (N_test, n+1, 1) by repeating its elements
        V = self.config.backend.broadcast(V, (N_test, self.n + 1)).reshape((N_test, self.n + 1, 1))
        return M, V

    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if n > self.n:
            raise ValueError(f"n ({n}) must be lower or equal than self.n ({self.n}).")

        # M shape is (N_test, N, n+1) with the polynomial basis of each distance in D (thus N_test, N)
        # V shape is (N_test, n+1, 1) the vector of size n+1 to apply to M for each testing point
        return component_support[:, :, : n + 1], component_x[:, : n + 1]

    def fit(self, X: ArrayLike) -> BaseDetector:
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")

        # Define The number of training data and the diension of training data
        N, d = X.shape

        # Preprocess the data
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.fit_transform(X)
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)

        # Save the information on training data
        self.N = cast(int, N)
        self.d = cast(int, d)
        self.X_train = X

        # Define the degree if set to auto using n = N**(1/(2+d)) with d=1 because univariate
        self.n = self._compute_n(self.n, self.N)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d, self.config.C)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return self.N is not None and self.d is not None and self.X_train is not None and self.threshold is not None and isinstance(self.n, int)


def UCG(n_list: list[int], config: StaticDetectorConfig[ArrayLike, DTypeLike]):
    return BaseCGDetector(UCF, n=n_list, config=config)
