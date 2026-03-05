from typing import Generic, Literal, cast, get_args

from ..backend.base_backend import Backend
from ..core.types import ArrayLike, DTypeLike
from .base_commons import BaseCommons
from .inverter import Inverter
from .polynomial_basis import PolynomialBasis

IMPLEMENTED_INCREMENTER = Literal["inverse", "sherman", "woodbury"]


class Incrementer(BaseCommons, Generic[ArrayLike, DTypeLike]):
    requires = ["backend", "inverter", "polynomial_basis"]

    def __init__(self, method: IMPLEMENTED_INCREMENTER = "woodbury"):
        assert method in get_args(IMPLEMENTED_INCREMENTER), f"method must be in {IMPLEMENTED_INCREMENTER}. Got {method}."
        self.method = method

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        self.inverter: Inverter[ArrayLike, DTypeLike]
        self.polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike]

    def increment(self, M: ArrayLike, N: int, X: ArrayLike, n: int) -> tuple[ArrayLike, int, ArrayLike] | ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Incrementer class before using it."
        assert self.inverter is not None, "An inverter must be bound to the Incrementer class before using it."
        assert self.polynomial_basis is not None, "A polynomial basis must be bound to the Incrementer class before using it."

        N_prime: int = X.shape[0]
        new_N = N_prime + N

        # nd because increment is impossible for the univariate version
        V = self.polynomial_basis.vandermonde_nd(X, n)

        # Inverse
        if self.method == "inverse":
            # Revert the mean of the moments matrix
            M *= N
            # Add new data
            M += V.T @ V
            # Recompute the mean of the moments matrix
            M /= new_N
            # Inverse the new moments matrix
            M_inv = self.inverter(M)
            return M, new_N, M_inv

        # Revert the mean of the inverse moments matrix
        M /= N

        # Sherman
        if self.method == "sherman":
            # Add point by point using Sherman-Morrison formula
            for row in V:
                v = cast(ArrayLike, row)
                # Compute the left-hand side of the Sherman-Morrison formula: M^-1 u
                left = M @ v
                # Compute the denominator: v^T M^-1 u
                denom = V.T @ left
                # Divide the left-hand side by (1 + denom)
                # Reduce the division cost from O(N^2) to O(N) by using the fact that left is a vector and numerator is a matrix
                left_div = left / (1 + denom)
                M -= left_div @ left.T
            # Compute the mean of the updated inverse moments matrix
            return M * new_N

        # Woodbury
        # Define the identity matrix C
        C = self.backend.eye(N_prime)
        # Compute the product V @ M^-1
        V_M_inv = V @ M
        # Compute the sum C^-1 + V @ M^-1 @ U
        sum_ = C + V_M_inv @ V.T
        # Compute the inverse of the sum
        if N == 1:
            sum_inv = 1 / sum_
        else:
            sum_inv = self.inverter(sum_)
        # Compute M^-1 @ U @ (C^-1 + V @ M^-1 @ U)^-1
        # M is symmetric and U = V^T so M^-1 @ U = (V @ M^-1)^T
        M_inv_U_sum_inv = V_M_inv.T @ sum_inv
        # Compute the product M^-1 @ U @ (C^-1 + V @ M^-1 @ U)^-1 @ V^T @ M^-1
        prod = M_inv_U_sum_inv @ V_M_inv
        M -= prod
        # Compute the mean of the updated inverse moments matrix
        return M * new_N

    def __call__(self, M: ArrayLike, N: int, X: ArrayLike, n: int) -> tuple[ArrayLike, int, ArrayLike] | ArrayLike:
        return self.increment(M=M, N=N, X=X, n=n)
