from typing import Any, Literal, Optional, TypeGuard, overload

import numpy as np
import torch

from ..core.types import ShapeType
from .base_backend import Backend


class TorchBackend(Backend[torch.Tensor, torch.dtype]):

    def __init__(self, dtype: torch.dtype = torch.float64, device: Optional[str | torch.device] = None):
        super().__init__(dtype)
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
        return torch.full(shape, fill_value, device=self.device, dtype=dtype)

    def arange(self, start: int, stop: int, step: int = 1, dtype: torch.dtype | None = None) -> torch.Tensor:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return torch.arange(start=start, end=stop, step=step, dtype=dtype, device=self.device)

    # ===== Random =====

    def set_seed(self, seed: int):
        self.generator.manual_seed(seed)

    def random(self, shape: ShapeType) -> torch.Tensor:
        return torch.rand(shape, generator=self.generator, device=self.device)

    def randn(self, mean, std, shape: ShapeType) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device=self.device) * std + mean

    def randint(self, low: int | torch.Tensor, high: int | torch.Tensor, shape: ShapeType) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self.generator, device=self.device)

    # ===== Shape ops =====

    def swap(self, A: torch.Tensor, axis1: int, axis2: int) -> torch.Tensor:
        return torch.swapdims(A, axis1, axis2)

    def concat(self, arrays: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    def stack(self, arrays: list[torch.Tensor], axis: int) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    def broadcast(self, A: torch.Tensor, shape: ShapeType) -> torch.Tensor:
        return A.expand(shape)

    # ===== Reduction ops =====

    def sum(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.sum(dim=axis, keepdim=keepdims)

    def prod(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return torch.prod(A, dim=axis, keepdim=keepdims)

    def cumsum(self, A: torch.Tensor, axis=None) -> torch.Tensor:
        return A.cumsum(dim=axis)

    def min(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.amin(dim=axis, keepdim=keepdims)

    def max(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.amax(dim=axis, keepdim=keepdims)

    def argmin(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.argmin(dim=axis, keepdim=keepdims)

    def argmax(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.argmax(dim=axis, keepdim=keepdims)

    def mean(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.mean(dim=axis, keepdim=keepdims)

    def std(self, A: torch.Tensor, axis=None, keepdims: bool = False) -> torch.Tensor | float:
        return A.std(dim=axis, keepdim=keepdims)

    def norm(self, A: torch.Tensor, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: bool = False) -> torch.Tensor | float:
        return torch.norm(A, p=p, keepdim=keepdims)

    def norm2D(self, A: torch.Tensor) -> torch.Tensor:
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
        return A.fill_diagonal_(val)

    def diag(self, A: torch.Tensor) -> torch.Tensor:
        return torch.diag(A)

    # ===== Math ops =====

    def nan(self) -> float:
        return torch.nan

    def isnan(self, A: torch.Tensor) -> torch.Tensor:
        return torch.isnan(A)

    def pow(self, A: torch.Tensor, power: int | float | torch.Tensor) -> torch.Tensor:
        return torch.pow(A, power)

    def sqrt(self, A: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(A)

    def abs(self, A: torch.Tensor) -> torch.Tensor:
        return torch.abs(A)

    def exp(self, A: torch.Tensor) -> torch.Tensor:
        return torch.exp(A)

    def log(self, A: torch.Tensor) -> torch.Tensor:
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

    # ===== Linear algebra =====

    def inv(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(A)

    def pinv(self, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.pinv(A)

    def solve(self, A: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.linalg.solve(A, b)

    def solve_triangular(self, A: torch.Tensor, B: torch.Tensor, upper: bool = False) -> torch.Tensor:
        return torch.linalg.solve_triangular(A, B, upper=upper)

    def cholesky(self, A: torch.Tensor, upper=False, allow_adding_reg: bool = True) -> torch.Tensor:
        L, info = torch.linalg.cholesky_ex(A, upper=upper)
        if info != 0:
            if allow_adding_reg:
                # Add regularization and try again
                eye = self.eye(A.shape[-1], dtype=A.dtype)
                for eps in range(12, 3, -1):
                    A += 10 ** (-eps) * eye
                    L, new_info = torch.linalg.cholesky_ex(A, upper=upper)
                    if new_info == 0:
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
        inv, info = torch.linalg.inv_ex(A)
        if info != 0:
            if allow_adding_reg:
                # Add regularization and try again
                eye = self.eye(A.shape[-1], dtype=A.dtype)
                for eps in range(12, 3, -1):
                    A += 10 ** (-eps) * eye
                    inv, new_info = torch.linalg.inv_ex(A)
                    if new_info == 0:
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

    def vander(self, a: torch.Tensor, degree: int, increasing: bool = True) -> torch.Tensor:
        return torch.vander(a, degree + 1, increasing=increasing)

    def lstsq(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        return torch.linalg.lstsq(A, B).solution
