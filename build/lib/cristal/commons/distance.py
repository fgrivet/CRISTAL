# pylint: disable=W0718
from typing import Generic, Literal, get_args

from ..backend.base_backend import Backend
from ..core.types import ArrayLike, DTypeLike
from .base_commons import BaseCommons

IMPLEMENTED_DISTANCE = Literal["euclidean", "mahalanobis"]


class Distance(BaseCommons, Generic[ArrayLike, DTypeLike]):
    requires = ["backend"]

    def __init__(self, metric: IMPLEMENTED_DISTANCE = "euclidean"):
        assert metric in get_args(IMPLEMENTED_DISTANCE), f"metric must be in {IMPLEMENTED_DISTANCE}. Got {metric}."
        self.metric = metric

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]

    def _cdist_euclidean(self, X: ArrayLike, Y: ArrayLike | None = None, p: int = 2) -> ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Distance class before using it."

        # Compute X_norms and Y_norms
        X_norms = self.backend.norm2D(X)
        if Y is None:
            new_Y = X
            Y_norms = X_norms
        else:
            new_Y = Y
            Y_norms = self.backend.norm2D(new_Y)

        # Use D = ||X - Y||_2^2 = ||X||_2^2 + ||Y||_2^2^T - 2 * X * Y^T
        D = X_norms[:, None] + Y_norms[None, :] - 2 * X @ new_Y.T

        # Clamp to avoid numerical errors (no negative values)
        D = self.backend.clip(D, min_=0.0)
        # If distances between X and X, diagonal must be 0
        if Y is None:
            D = self.backend.fill_diagonal(D, 0.0)

        # Returns ||X - Y||_2^p
        if p != 2:
            D = D ** (p / 2)
        return D

    def _compute_covariance_matrix(self, X: ArrayLike) -> ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Distance class before using it."

        # Compute the covariance matrix
        X_mean = self.backend.mean(X, axis=0)
        X_centered = X - X_mean
        N: int = X_centered.shape[0]
        Sigma = (X_centered.T @ X_centered) / (N - 1)
        return Sigma

    def cdist(self, X: ArrayLike, Y: ArrayLike | None = None, p=2) -> ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Distance class before using it."

        if self.metric == "mahalanobis":
            new_Y = X if Y is None else Y

            # Compute the covariance matrix
            Sigma = self._compute_covariance_matrix(new_Y)

            # Compute the Cholesky decomposition of Sigma
            upper = False
            L = self.backend.cholesky(Sigma, upper=upper, allow_adding_reg=True)

            # Solve Z_X = X L^{-1}
            Z_X = self.backend.solve_triangular(L, X.T, upper=upper).T
            # Solve Z_Y = Y L^{-1}
            Z_Y = Z_X if Y is None else self.backend.solve_triangular(L, new_Y.T, upper=upper).T

            # D = D_E(L^-1 X, L^-1 Y)
            D = self._cdist_euclidean(Z_X, Z_Y, p=p)

        else:
            D = self._cdist_euclidean(X, Y, p=p)

        return D

    def __call__(self, X: ArrayLike, Y: ArrayLike | None = None, p=2) -> ArrayLike:
        return self.cdist(X=X, Y=Y, p=p)
