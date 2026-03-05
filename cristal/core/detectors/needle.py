from typing import Literal, cast

from ...config.detector_config import StaticDetectorConfig
from ..types import ArrayLike, DTypeLike
from .base_detector import BaseCGDetector, BaseDetector


class NeedleCF(BaseDetector[ArrayLike, DTypeLike, StaticDetectorConfig]):
    def __init__(self, n: int | Literal["auto"], config: StaticDetectorConfig[ArrayLike, DTypeLike]):
        super().__init__(n, config)
        self.intrinsic_dim = 1

        # Variables defined during fitting specific to NeedleCF
        self.X_train: ArrayLike | None = None  #: The training data

    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike:
        # component_support = T_n_A[:, : , n+1] = T_n(A)  --> Shape (N_test, 1, 1)
        # component_x = T_n_A_B[:, :, n+1] = T_n(A - B) --> Shape (N_test, N, 1)
        # Compute T_n(A)^2 and 1/N sum(T_n(A - B)^2)
        num = self.config.backend.pow(component_support, 2)  # Shape (N_test, 1, 1)
        denom = self.config.backend.mean(self.config.backend.pow(component_x, 2), axis=1, keepdims=True)  # Shape (N_test, 1, 1)

        # Final result : 1 / int(q^2) = T_n(A)^2  / mean(T_n(A - B)^2)
        res = num / denom  # Shape (N_test, 1, 1)
        res = res.reshape((-1,))  # Shape (N_test,)

        return res

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        assert isinstance(self.n, int) and self.n > 0, "n must be a positive integer."
        assert self.d is not None and self.d > 0, "d must be a positive integer."

        # Compute the distances between all points in X and in X_train
        D = self.config.distance(X, self.X_train)  # Shape (N_test, N)
        norm = self.config.backend.sqrt(D)  # Shape (N_test, N)

        # Compute rho and delta values
        rho = self.config.backend.max(norm, axis=1, keepdims=True)  # Shape (N_test, 1)
        delta = self.config.backend.min(norm, axis=1, keepdims=True)  # Shape (N_test, 1)

        # Compute A and B such that q = T_n(A - B) / T_n(A)
        A = 1 + self.config.backend.pow(delta / rho, 2)  # Shape (N_test, 1)
        B = D / self.config.backend.pow(rho, 2)  # Shape (N_test, N)

        # Compute the differences between A and B
        diff = A - B  # Shape (N_test, N)

        # Compute T_n(A - B) and T_n(A)
        T_n_A_B = self.config.polynomial_basis.vandermonde_1d(diff, self.n, self.d, normalize=False)  # Shape (N_test, N, k)
        T_n_A = self.config.polynomial_basis.vandermonde_1d(A, self.n, self.d, normalize=False)  # Shape (N_test, 1, k)

        return T_n_A, T_n_A_B

    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]:
        assert isinstance(self.n, int) and self.n > 0, "n must be a positive integer."
        assert n <= self.n, "n must be lower or equal than self.n"

        return component_support[:, :, [n]], component_x[:, :, [n]]

    def fit(self, X: ArrayLike) -> BaseDetector:
        assert X.ndim == 2, "X must be a 2D ArrayLike."

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


def NeedleCG(n_list: list[int], config: StaticDetectorConfig[ArrayLike, DTypeLike]):
    return BaseCGDetector(NeedleCF, n=n_list, config=config)
