from typing import Generic, Literal, get_args

from ..backend.base_backend import Backend
from ..core.types import ArrayLike, DTypeLike
from .base_commons import BaseCommons

IMPLEMENTED_INVERTER = Literal["inv", "pseudo", "solve", "fpd"]


class Inverter(BaseCommons, Generic[ArrayLike, DTypeLike]):
    requires = ["backend"]

    inds_cache = {}

    def __init__(self, method: IMPLEMENTED_INVERTER = "fpd", eps: float | None = None):
        assert method in get_args(IMPLEMENTED_INVERTER), f"method must be in {IMPLEMENTED_INVERTER}. Got {method}."
        self.method = method
        self.eps = eps

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]

    def invert(self, X: ArrayLike, eps: float | None = None) -> ArrayLike:
        assert self.backend is not None, "A backend must be bound to the Inverter class before using it."

        eps = eps or self.eps  # Possibly override the default eps value
        if eps is not None:
            X += self.backend.eye(X.shape[0]) * eps

        # inv
        if self.method == "inv":
            return self.backend.inv(X)

        # pseudo
        if self.method == "pseudo":
            return self.backend.pinv(X)

        # solve
        if self.method == "solve":
            I = self.backend.eye(X.shape[0])
            return self.backend.solve(X, I)

        # fpd
        return self.backend.inverse_cholesky(X, upper=False, allow_adding_reg=True)

    def __call__(self, X: ArrayLike, eps: float | None = None) -> ArrayLike:
        return self.invert(X, eps=eps)
