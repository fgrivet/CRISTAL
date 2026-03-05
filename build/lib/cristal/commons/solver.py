from typing import Generic, Literal, cast, get_args

from ..backend.base_backend import Backend
from ..core.types import ArrayLike, DTypeLike
from .base_commons import BaseCommons

IMPLEMENTED_SOLVER = Literal["cholesky", "qr", "inverse", "solve"]


class Solver(BaseCommons, Generic[ArrayLike, DTypeLike]):
    requires = ["backend"]

    def __init__(self, solver: IMPLEMENTED_SOLVER = "solve", eps: float | None = None):
        assert solver in get_args(IMPLEMENTED_SOLVER), f"solver must be in {IMPLEMENTED_SOLVER}. Got {solver}."
        self.solver = solver
        self.eps = eps

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]

    def solve(self, V: ArrayLike, v: ArrayLike, N: int) -> ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Solver class before using it."

        x = None

        # QR decomposition does not need to compute G
        if self.solver == "qr":
            # z = v^T R^-1 R^-T v with y = R^-T v and x = R^-1 y
            _, R = self.backend.qr(V)
            # G = V^T V / N, so either we divide R by sqrt(N) and solve the systems
            # Or we divide R in a system by N once

            y = self.backend.solve(self.backend.swap(R, 1, 2) / N, v)  # R^T y = v
            x = self.backend.solve(R, y)  # R x = y

        # Other solvers need to compute G
        else:
            G = self.backend.swap(V, 1, 2) @ V / N
            G = (G + self.backend.swap(G, 1, 2)) / 2  # Ensure G is symmetric

            k: int = V.shape[-1]
            dtype = cast(DTypeLike, V.dtype)
            eye = self.backend.eye(k, dtype=dtype)

            # Regularization
            if self.eps is not None:
                G += self.eps * eye

            if self.solver == "inverse":
                # z = v^T G^-1 v with x = G^-1 v but compute explicitly G^-1
                # Inverse G using solve
                G_inv = self.backend.solve(G, eye)
                x = G_inv @ v

            if self.solver == "solve":
                # z = v^T G^-1 v with x = G^-1 v
                x = self.backend.solve(G, v)  # G x = v

            if self.solver == "cholesky":
                # z = v^T L^-T L^-1 v with y = L^-1 v and x = L^-T y
                L = self.backend.cholesky(G)
                y = self.backend.solve(L, v)  # L y = v
                x = self.backend.solve(self.backend.swap(L, 1, 2), y)  # L^T x = y

        if x is None:
            raise ValueError(f"Wrong solver name / solver implementation in Solver. Got {self.solver}. Should be in {IMPLEMENTED_SOLVER}")

        return self.backend.einsum("i,mi->m", v[0, :, 0], x[..., 0])  # z = v^T x

    def __call__(self, V: ArrayLike, v: ArrayLike, N: int) -> ArrayLike:
        return self.solve(V=V, v=v, N=N)
