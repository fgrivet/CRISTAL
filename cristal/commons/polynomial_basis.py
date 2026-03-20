from typing import Generic, Literal, cast, get_args

from ..backend.base_backend import Backend
from ..types import ArrayLike, DTypeLike
from .base_commons import BaseCommons

IMPLEMENTED_POLYNOMIAL_BASIS = Literal["monomials", "chebyshev"]


class PolynomialBasis(BaseCommons, Generic[ArrayLike, DTypeLike]):
    requires = ["backend"]

    def __init__(self, basis: IMPLEMENTED_POLYNOMIAL_BASIS = "chebyshev", normalize: bool = True):
        if basis not in get_args(IMPLEMENTED_POLYNOMIAL_BASIS):
            raise ValueError(f"basis must be in {IMPLEMENTED_POLYNOMIAL_BASIS}. Got {basis}.")
        self.basis = basis
        self.normalize = normalize

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]

    def generate_multi_indices_combinations(self, max_degree, dimensions):
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if max_degree <= 0:
            raise ValueError(f"max_degree must be positive. Got {max_degree}.")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive. Got {dimensions}.")

        def generate_exact_degree(total_degree: int, dims_left: int):
            if dims_left == 1:
                yield (total_degree,)
                return

            for value in range(total_degree, -1, -1):
                for tail in generate_exact_degree(total_degree - value, dims_left - 1):
                    yield (value,) + tail

        indices_comb = [indices for total_degree in range(max_degree + 1) for indices in (generate_exact_degree(total_degree, dimensions))]
        return self.backend.to_array_like(indices_comb, dtype=int)

    # ---------- 1D ----------
    def vandermonde_1d(self, X: ArrayLike, n: int, d: float, normalize: bool | None = None) -> ArrayLike:
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")
        if d <= 0:
            raise ValueError(f"d must be positive. Got {d}.")

        if normalize is None:
            normalize = self.normalize

        k = n + 1
        dtype = cast(DTypeLike, X.dtype)

        # Monomials
        if self.basis == "monomials":
            powers = self.backend.arange(0, k, dtype=dtype)
            return self.backend.pow(X[:, :, None], powers[None, None, :])

        # Chebyshev
        M, N = X.shape

        if normalize:
            X = cast(ArrayLike, X / (2 * d) - 1)

        V = self.backend.zeros((M, N, k), dtype=dtype)
        V[..., 0] = 1

        if n >= 1:
            V[..., 1] = X

        for j in range(2, k):
            # T_j = 2 * X * T_{j-1} - T_{j-2}
            V[..., j] = cast(ArrayLike, 2 * X * V[..., j - 1] - V[..., j - 2])

        return V

    # ---------- ND ----------
    def vandermonde_nd(self, X: ArrayLike, n: int) -> ArrayLike:
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")

        M, d = X.shape
        alpha = self.generate_multi_indices_combinations(n, d)  # (s_d(n), d)

        # Monomials
        if self.basis == "monomials":
            # broadcasting
            X_exp = X[:, None, :]  # (M, 1, d)
            alpha_exp = alpha[None, :, :]  # (1, s_d(n), d)

            V = self.backend.prod(self.backend.pow(X_exp, alpha_exp), axis=2)

            return V

        # Chebyshev
        # Compute all 1D Chebyshev up to degree n for each dimension
        temp = self.backend.stack([self.vandermonde_1d(X[:, j : j + 1], n, d=d, normalize=False) for j in range(d)], axis=-1)
        temp = temp.reshape(M, n + 1, d)  # (M, n+1, d)

        # Gather appropriate degrees per dimension
        V = self.backend.ones((M, alpha.shape[0]), dtype=cast(DTypeLike, X.dtype))
        for dim in range(d):
            V *= temp[:, alpha[:, dim], dim]

        return V

    # ---------- v ----------
    def make_v(self, n: int, dtype: DTypeLike):
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")

        # Monomials
        if self.basis == "monomials":
            v = self.backend.zeros((n + 1,), dtype=dtype)
            v[0] = 1
            return v

        # Chebyshev
        v = self.backend.ones((n + 1,), dtype=dtype)
        v[1::2] = -1
        return v
