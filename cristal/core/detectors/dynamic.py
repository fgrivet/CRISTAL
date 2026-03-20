from math import comb
from typing import Literal, cast

from ...config.detector_config import DynamicDetectorConfig
from ...types import ArrayLike, DTypeLike
from .base_detector import BaseCGDetector, BaseDetector


class DyCF(BaseDetector[ArrayLike, DTypeLike, DynamicDetectorConfig]):
    def __init__(self, n: int | Literal["auto"], config: DynamicDetectorConfig[ArrayLike, DTypeLike]):
        super().__init__(n, config)

        # Variables defined during fitting specific to DyCF
        self.M: ArrayLike | None = None  #: The moments matrix
        self.M_inv: ArrayLike | None = None  #: The inverse of the moments matrix

    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike:
        # Here component_support corresponds to M^{-1} and component_x to v(x)
        # Compute the scores v(x) @ M^{-1} @ v(x)^T for each x
        return self.config.backend.einsum("ni,ij,nj->n", component_x, component_support, component_x)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.M_inv is None:
            raise ValueError("M_inv must be computed.")

        # Compute the polynomial basis v(x) for each point in X
        V = self.config.polynomial_basis.vandermonde_nd(X, self.n)
        # M_inv is already computed here
        return self.M_inv, V

    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]:
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.d is None or self.d <= 0:
            raise ValueError(f"d must be a positive integer. Got {self.d}.")
        if n > self.n:
            raise ValueError(f"n ({n}) must be lower or equal than self.n ({self.n}).")
        if self.M is None:
            raise ValueError("M must be fitted.")

        new_size = comb(self.d + n, n)

        if component_support.shape[0] == new_size:
            return component_support, component_x

        # Crop and then inverse again
        M_inv = self.config.inverter(self.M[:new_size, :new_size])
        return M_inv, component_x[:, :new_size]

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
        self.intrinsic_dim = self.d

        # Define the degree if set to auto using n = N**(1/(2+d))
        self.n = self._compute_n(self.n, self.N)

        # Compute the moments matrix
        matrix_size = comb(self.d + self.n, self.n)
        M = self.config.backend.zeros((matrix_size, matrix_size))
        for X_batch in self.config.storage(X):
            V = self.config.polynomial_basis.vandermonde_nd(X_batch, self.n)
            M += V.T @ V
        M = (M + M.T) / (2 * N)  # Ensure symmetry of M and normalize by N
        self.M = cast(ArrayLike, M)

        # Compute the inverse of the moments matrix
        self.M_inv = self.config.inverter(self.M)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d, self.config.C)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return (
            self.N is not None
            and self.d is not None
            and self.intrinsic_dim is not None
            and self.M is not None
            and self.M_inv is not None
            and self.threshold is not None
            and isinstance(self.n, int)
        )

    # TODO
    def update(self, X: ArrayLike) -> "DyCF": ...
    def save_model(self): ...
    def load_model(self): ...


def DyCG(n_list: list[int], config: DynamicDetectorConfig[ArrayLike, DTypeLike]):
    return BaseCGDetector(DyCF, n=n_list, config=config)
