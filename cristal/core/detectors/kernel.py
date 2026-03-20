from typing import Literal, cast

from ...commons.distance import Distance
from ...config.detector_config import DynamicDetectorConfig
from ...types import ArrayLike, DTypeLike, Number
from .base_detector import BaseCGDetector, BaseDetector


class KernelCF(BaseDetector[ArrayLike, DTypeLike, DynamicDetectorConfig]):
    def __init__(
        self,
        n: int | Literal["auto"],
        config: DynamicDetectorConfig[ArrayLike, DTypeLike],
        kernel: Literal["linear", "rbf"] = "rbf",
        rho: Literal["auto"] | Number = "auto",
        sigma: Literal["auto"] | Number = "auto",
    ):
        super().__init__(n, config)

        # Variables specific to KernelCF
        self.kernel = kernel
        self.kernel_func = self._rbf_kernel if self.kernel == "rbf" else self._linear_kernel
        self.sigma: Literal["auto"] | Number = sigma  #: The parameter sigma for the rbf kernel
        self.rho: Literal["auto"] | Number = rho  #: The parameter rho for the regularization of the kernel

        # Variables defined during fitting specific to KernelCF
        self.X_train: ArrayLike | None = None  #: The training data
        self.G = None  #: The gram matrix
        self.G_inv = None  #: The inverse of the gram matrix

    def _rbf_kernel(self, X, Y=None):
        if self.sigma == "auto":
            raise ValueError(f"Sigma must be specified when using RBF kernel. Got {self.sigma}.")
        gamma = 1 / (2 * self.sigma**2)
        distance = Distance(metric="euclidean")
        distance.backend = self.config.backend
        D = distance(X, Y, p=2)
        return self.config.backend.exp(-gamma * D)

    def _linear_kernel(self, X, Y=None):
        if Y is None:
            Y = X
        return (1 + X @ Y.T) ** self.n

    def _compute_sigma(self, sigma: Literal["auto"] | Number, d: int) -> float:
        if sigma == "auto":
            return self.config.backend.sqrt(d) / 2
        return float(sigma)

    def _compute_rho(self, rho: Literal["auto"] | Number, n: int) -> float:
        if rho == "auto":
            if self.G is None:
                raise ValueError("G must be set before computing rho.")
            return self.config.backend.norm(self.G, p="fro") / (self.config.C * self.config.backend.sqrt(n))
        return float(rho)

    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike:
        # component_x = gamma, component_support = g
        # Compute the scores gamma - g^T @ (rho I + G)^{-1} @ g
        return (component_x - self.config.backend.einsum("ni,nn,ni->i", component_support, self.G_inv, component_support)) / self.rho

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        g = self.kernel_func(self.X_train, X)
        gamma = self.config.backend.diag(self.kernel_func(X))
        return g, gamma

    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]:
        # Kernelized version, do not change the size of the matrices
        return component_support, component_x

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
        self.X_train = X

        # Define the degree if set to auto using n = N**(1/(2+d)) with d=1 because univariate
        self.n = self._compute_n(self.n, self.N)

        self.sigma = self._compute_sigma(self.sigma, self.d)
        self.G = self.kernel_func(X, X)
        self.rho = self._compute_rho(self.rho, self.n)
        self.G_inv = self.config.inverter(self.G, eps=self.rho)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d, self.config.C)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return self.N is not None and self.d is not None and self.X_train is not None and self.threshold is not None and isinstance(self.n, int)


def KernelCG(
    n_list: list[int],
    config: DynamicDetectorConfig[ArrayLike, DTypeLike],
    rho: Literal["auto"] | Number = "auto",
    sigma: Literal["auto"] | Number = "auto",
):
    return BaseCGDetector(KernelCF, n=n_list, config=config, rho=rho, sigma=sigma)
