"""Contains the :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>`: a backend using torch with `cpu` and `GPU` implementations."""

import logging
from typing import Any, Literal, Optional, TypeGuard, overload

import numpy as np
import torch

from ..types import Number, ShapeType
from .base_backend import Backend

logger = logging.getLogger(__name__)


class TorchBackend(Backend[torch.Tensor, torch.dtype]):
    """Backend using torch with `cpu` and `GPU` implementations."""

    def __init__(self, dtype: torch.dtype = torch.float64, device: Optional[str | torch.device] = None):
        super().__init__(dtype)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device={device}")
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.generator = torch.Generator(device=self.device)

    # ===== Type =====

    def is_array_like(self, x: Any) -> TypeGuard[torch.Tensor]:
        return isinstance(x, torch.Tensor) and (x.device == self.device)

    def to_array_like(self, x: Any, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.cpu().detach().numpy()

    # ===== Creation =====

    def empty(self, shape: ShapeType, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.empty(shape, device=self.device, dtype=dtype)

    def zeros(self, shape: ShapeType, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.zeros(shape, device=self.device, dtype=dtype)

    def ones(self, shape: ShapeType, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.ones(shape, device=self.device, dtype=dtype)

    def eye(self, n: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.eye(n, device=self.device, dtype=dtype)

    def full(self, shape: ShapeType, fill_value: int, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, fill_value, device=self.device, dtype=dtype)  # type: ignore

    def arange(self, start_or_stop: Number, /, stop: Number | None = None, step: Number = 1, *, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        if stop is None:
            return torch.arange(start_or_stop, dtype=dtype, device=self.device)
        return torch.arange(start_or_stop, stop, step, dtype=dtype, device=self.device)

    def copy(self, A: torch.Tensor) -> torch.Tensor:
        return A.detach().clone()

    # ===== Random =====

    def set_seed(self, seed: int):
        self.generator.manual_seed(seed)

    def random(self, shape: ShapeType) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def randn(self, mean, std, shape: ShapeType) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device) * std + mean

    def randint(self, low: int | torch.Tensor, high: int | torch.Tensor, shape: ShapeType) -> torch.Tensor:
        dtype = torch.int64

        if not torch.is_tensor(low):
            low = torch.tensor(low, device=self.device)
        if not torch.is_tensor(high):
            high = torch.tensor(high, device=self.device)

        try:
            low_broadcast = low.expand(shape).to(dtype)
            high_broadcast = high.expand(shape).to(dtype)
        except RuntimeError:
            raise ValueError(f"Incompatible shapes between low / high and shape {shape}.")

        if torch.any(low_broadcast > high_broadcast):
            raise ValueError(f"Invalid range: low ({low}) > high ({high}).")

        range_ = high_broadcast - low_broadcast
        max_int = torch.iinfo(dtype).max

        limit = max_int - (max_int % range_)

        result = torch.empty(shape, dtype=dtype, device=self.device)
        done = torch.zeros(shape, dtype=torch.bool, device=self.device)

        while not torch.all(done):
            r = torch.randint(0, max_int, shape, dtype=dtype, device=self.device)  # type: ignore

            valid = r < limit
            new_vals = (r % range_) + low_broadcast

            mask = valid & (~done)
            result[mask] = new_vals[mask]
            done |= mask

        return result

    # ===== Shape ops =====

    def swap(self, A: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        return torch.swapdims(A, axis1, axis2)

    def concat(self, arrays: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
        try:
            return torch.cat(arrays, dim=axis)
        except RuntimeError as exc:
            raise ValueError(f"Cannot concat to shape arrays along axis {axis}.") from exc

    def stack(self, arrays: list[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    def broadcast(self, A: torch.Tensor, shape: ShapeType) -> torch.Tensor:
        try:
            return A.expand(shape)
        except RuntimeError as exc:
            raise ValueError(f"Cannot broadcast to shape {shape}.") from exc

    # ===== Reduction ops =====

    @overload
    def sum(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def sum(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def sum(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.sum(dim=axis, keepdim=keepdims)

    @overload
    def prod(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def prod(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def prod(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        if axis is None:
            res = torch.prod(A.flatten())
            if keepdims:
                res = res[tuple([None for _ in range(A.ndim)])]
            return res
        return torch.prod(A, dim=axis, keepdim=keepdims)

    def cumsum(self, A: torch.Tensor, axis=None) -> torch.Tensor:
        if axis is None:
            return A.flatten().cumsum(0)
        return A.cumsum(dim=axis)

    @overload
    def min(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def min(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def min(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.amin(dim=axis, keepdim=keepdims)  # type: ignore

    @overload
    def max(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def max(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def max(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.amax(dim=axis, keepdim=keepdims)  # type: ignore

    @overload
    def argmin(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> int: ...

    @overload
    def argmin(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def argmin(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | int:
        return A.argmin(dim=axis, keepdim=keepdims)

    @overload
    def argmax(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> int: ...

    @overload
    def argmax(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def argmax(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | int:
        return A.argmax(dim=axis, keepdim=keepdims)

    @overload
    def mean(self, A: torch.Tensor, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def mean(self, A: torch.Tensor, axis: int, keepdims: bool = False) -> torch.Tensor: ...

    def mean(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.mean(dim=axis, keepdim=keepdims)

    @overload
    def std(self, A: torch.Tensor, axis: None = None, ddof: int = 0, keepdims: bool = False) -> float: ...

    @overload
    def std(self, A: torch.Tensor, axis: int, ddof: int = 0, keepdims: bool = False) -> torch.Tensor: ...

    def std(self, A: torch.Tensor, axis=None, ddof: int = 0, keepdims: bool = False) -> torch.Tensor | float:
        return A.std(dim=axis, correction=ddof, keepdim=keepdims)

    @overload
    def norm(self, A: torch.Tensor, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: Literal[False] = False) -> float: ...

    @overload
    def norm(self, A: torch.Tensor, p: Literal["inf", "-inf", "fro", "nuc"] | int, keepdims: Literal[True]) -> torch.Tensor: ...

    def norm(self, A: torch.Tensor, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: bool = False) -> torch.Tensor | float:
        return torch.linalg.norm(A, ord=p, keepdim=keepdims)

    def norm2D(self, A: torch.Tensor) -> torch.Tensor:
        if A.ndim != 2:
            raise ValueError("Input must be 2D.")
        return (A**2).sum(dim=1)

    def einsum(self, subscripts: str, *operands: torch.Tensor) -> torch.Tensor:
        return torch.einsum(subscripts, *operands)

    # ===== Tensor ops =====

    @overload
    def where(self, condition: torch.Tensor, true_val: Any, false_val: Any) -> torch.Tensor: ...

    @overload
    def where(self, condition: torch.Tensor, true_val: None = None, false_val: None = None) -> tuple[torch.Tensor, ...]: ...

    def where(self, condition: torch.Tensor, true_val=None, false_val=None) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if true_val is None or false_val is None:
            return torch.where(condition)
        return torch.where(condition, true_val, false_val)

    def clip(self, A: torch.Tensor, min_=None, max_=None) -> torch.Tensor:
        return torch.clamp(A, min=min_, max=max_)

    def fill_diagonal(self, A: torch.Tensor, val) -> torch.Tensor:
        try:
            return self.copy(A).fill_diagonal_(val)
        except RuntimeError as exc:
            raise ValueError(f"Cannot fill diagonal with {val}.") from exc

    def diag(self, A: torch.Tensor) -> torch.Tensor:
        return torch.diag(A)

    # ===== Math ops =====

    def sign(self, A: torch.Tensor) -> torch.Tensor:
        return torch.sign(A)

    def isnan(self, A: torch.Tensor) -> torch.Tensor:
        return torch.isnan(A)

    def pow(self, A: torch.Tensor, power: Number | torch.Tensor) -> torch.Tensor:
        return torch.pow(A, power)

    def sqrt(self, A: torch.Tensor) -> torch.Tensor:
        if torch.any(A < 0):
            raise ValueError("Trying to compute the sqrt of negative values.")
        return torch.sqrt(A)

    def abs(self, A: torch.Tensor) -> torch.Tensor:
        return torch.abs(A)

    def exp(self, A: torch.Tensor) -> torch.Tensor:
        return torch.exp(A)

    def log(self, A: torch.Tensor) -> torch.Tensor:
        if torch.any(A < 0):
            raise ValueError("Trying to compute the log of negative values.")
        return torch.log(A)

    def cos(self, A: torch.Tensor) -> torch.Tensor:
        return torch.cos(A)

    def sin(self, A: torch.Tensor) -> torch.Tensor:
        return torch.sin(A)

    def tan(self, A: torch.Tensor) -> torch.Tensor:
        return torch.tan(A)

    def cosh(self, A: torch.Tensor) -> torch.Tensor:
        return torch.cosh(A)

    def sinh(self, A: torch.Tensor) -> torch.Tensor:
        return torch.sinh(A)

    def tanh(self, A: torch.Tensor) -> torch.Tensor:
        return torch.tanh(A)

    def arccos(self, A: torch.Tensor) -> torch.Tensor:
        return torch.arccos(A)

    def arccosh(self, A: torch.Tensor) -> torch.Tensor:
        return torch.arccosh(A)

    # ===== Linear algebra =====

    def inv(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(A)

    def pinv(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.pinv(A)

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(A, b)

    def solve_triangular(self, A: torch.Tensor, B: torch.Tensor, upper: bool = False) -> torch.Tensor:
        B_2D = B
        if B_2D.ndim == 1:
            B_2D = B_2D.view(-1, 1)
        res = torch.linalg.solve_triangular(A, B_2D, upper=upper)
        if B.ndim == 1:
            res = res.view(-1)
        return res

    def cholesky(self, A: torch.Tensor, upper=False, allow_adding_reg: bool = True) -> torch.Tensor:
        if torch.linalg.det(A) <= 0:
            L, info = A, 1
        else:
            L, info = torch.linalg.cholesky_ex(A, upper=upper)
        if info != 0:
            if allow_adding_reg:
                A_reg = self.copy(A)
                # Add regularization and try again
                eye = self.eye(A.shape[-1], dtype=A.dtype)
                for eps in range(12, 3, -1):
                    A_reg = A + 10 ** (-eps) * eye
                    # While not spd, find a new eps
                    if torch.linalg.det(A_reg) <= 0:
                        continue
                    L, new_info = torch.linalg.cholesky_ex(A_reg, upper=upper)
                    if new_info == 0:
                        logger.info("Regularization successful with eps=1e-%d", eps)
                        # If successful, break out of the loop
                        break
                else:
                    # If all attempts fail, raise an error
                    raise ValueError("Could not compute Cholesky decomposition. The matrix may not be positive definite.")
            else:
                # Error and no regularization is allowed, so we raise an error
                raise ValueError("Could not compute Cholesky decomposition. The matrix may not be positive definite.")
        return L

    def inverse_cholesky(self, A: torch.Tensor, upper: bool = False, allow_adding_reg=True) -> torch.Tensor:
        if torch.linalg.det(A) <= 0:
            inv, info = A, 1
        else:
            inv, info = torch.linalg.inv_ex(A)
        if info != 0:
            if allow_adding_reg:
                A_reg = self.copy(A)
                # Add regularization and try again
                eye = self.eye(A.shape[-1], dtype=A.dtype)
                for eps in range(12, 3, -1):
                    A_reg = A + 10 ** (-eps) * eye
                    # While not spd, find a new eps
                    if torch.linalg.det(A_reg) <= 0:
                        continue
                    inv, new_info = torch.linalg.inv_ex(A_reg)
                    if new_info == 0:
                        logger.info("Regularization successful with eps=1e-%d", eps)
                        # If successful, break out of the loop
                        break
                else:
                    # If all attempts fail, raise an error
                    raise ValueError("Could not compute Cholesky inverse. The matrix may not be positive definite.")
            else:
                # Error and no regularization is allowed, so we raise an error
                raise ValueError("Could not compute Cholesky inverse. The matrix may not be positive definite.")
        return inv

    def qr(self, A: torch.Tensor, mode="reduced") -> tuple[torch.Tensor, torch.Tensor]:
        return torch.linalg.qr(A, mode=mode)

    def vander(self, A: torch.Tensor, degree: int, increasing: bool = True) -> torch.Tensor:
        return torch.vander(A, degree + 1, increasing=increasing).to(A.dtype)

    def lstsq(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lstsq(A, B).solution
