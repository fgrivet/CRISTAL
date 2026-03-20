from typing import Any, Literal, TypeGuard, overload

import numpy as np
import scipy
import scipy.linalg
from numpy.typing import DTypeLike as NumpyDTypeLike

from ..types import Number, ShapeType
from .base_backend import Backend


class NumpyBackend(Backend[np.ndarray, NumpyDTypeLike]):

    def __init__(self, dtype: NumpyDTypeLike = np.float64):
        super().__init__(dtype)
        self.generator = np.random.default_rng()

    # ===== Type =====

    def is_array_like(self, x: Any) -> TypeGuard[np.ndarray]:
        return isinstance(x, np.ndarray)

    def to_array_like(self, x: Any, dtype: NumpyDTypeLike | None = None) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.asarray(x, dtype=dtype)

    def to_numpy(self, x: np.ndarray) -> np.ndarray:
        return x

    # ===== Creation =====

    def zeros(self, shape: ShapeType, dtype: NumpyDTypeLike | None = None) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: ShapeType, dtype: NumpyDTypeLike | None = None) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.ones(shape, dtype=dtype)

    def eye(self, n: int, dtype: NumpyDTypeLike | None = None) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.eye(n, dtype=dtype)

    def full(self, shape: ShapeType, fill_value: Any, dtype: NumpyDTypeLike | None = None) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.full(shape, fill_value, dtype=dtype)

    def arange(
        self,
        start_or_stop: Number,
        /,
        stop: Number | None = None,
        step: Number = 1,
        *,
        dtype: NumpyDTypeLike | None = None,
    ) -> np.ndarray:
        dtype = dtype or self.default_dtype  # dtype if is not None else self.default_dtype
        return np.arange(start_or_stop, stop, step, dtype=dtype)

    def copy(self, A: np.ndarray) -> np.ndarray:
        return A.copy()

    # ===== Random =====

    def set_seed(self, seed: int):
        self.generator = np.random.default_rng(seed)

    def random(self, shape: ShapeType) -> np.ndarray:
        return self.generator.random(shape)

    def randn(self, mean, std, shape: ShapeType) -> np.ndarray:
        return self.generator.normal(mean, std, shape)

    def randint(self, low: int | np.ndarray, high: int | np.ndarray, shape: ShapeType) -> np.ndarray:
        return self.generator.integers(low, high, shape)

    # ===== Shape ops =====

    def swap(self, A: np.ndarray, axis1: int, axis2: int) -> np.ndarray:
        return np.swapaxes(A, axis1, axis2)

    def concat(self, arrays: list[np.ndarray], axis: int = 0) -> np.ndarray:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: list[np.ndarray], axis: int) -> np.ndarray:
        return np.stack(arrays, axis=axis)

    def broadcast(self, A, shape):
        return np.broadcast_to(A, shape)

    # ===== Reduction ops =====

    @overload
    def sum(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def sum(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def sum(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | float:
        return A.sum(axis=axis, keepdims=keepdims)

    @overload
    def prod(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def prod(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def prod(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | float:
        return np.prod(A, axis=axis, keepdims=keepdims)

    def cumsum(self, A: np.ndarray, axis=None) -> np.ndarray:
        return A.cumsum(axis=axis)

    @overload
    def min(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def min(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def min(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | float:
        return A.min(axis=axis, keepdims=keepdims)

    @overload
    def max(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def max(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def max(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | float:
        return A.max(axis=axis, keepdims=keepdims)

    @overload
    def argmin(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> int: ...

    @overload
    def argmin(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def argmin(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | int:
        return A.argmin(axis=axis, keepdims=keepdims)  # type: ignore

    @overload
    def argmax(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> int: ...

    @overload
    def argmax(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def argmax(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | int:
        return A.argmax(axis=axis, keepdims=keepdims)  # type: ignore

    @overload
    def mean(self, A: np.ndarray, axis: None = None, keepdims: bool = False) -> float: ...

    @overload
    def mean(self, A: np.ndarray, axis: int, keepdims: bool = False) -> np.ndarray: ...

    def mean(self, A: np.ndarray, axis=None, keepdims: bool = False) -> np.ndarray | float:
        return A.mean(axis=axis, keepdims=keepdims)

    @overload
    def std(self, A: np.ndarray, axis: None = None, ddof: int = 0, keepdims: bool = False) -> float: ...

    @overload
    def std(self, A: np.ndarray, axis: int, ddof: int = 0, keepdims: bool = False) -> np.ndarray: ...

    def std(self, A: np.ndarray, axis=None, ddof: int = 0, keepdims: bool = False) -> np.ndarray | float:
        return A.std(axis=axis, ddof=ddof, keepdims=keepdims)

    @overload
    def norm(self, A: np.ndarray, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: Literal[False] = False) -> float: ...

    @overload
    def norm(self, A: np.ndarray, p: Literal["inf", "-inf", "fro", "nuc"] | int, keepdims: Literal[True]) -> np.ndarray: ...

    def norm(self, A: np.ndarray, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: bool = False) -> np.ndarray | float:
        return np.linalg.norm(A, ord=p, keepdims=keepdims)  # type: ignore

    def norm2D(self, A: np.ndarray) -> np.ndarray:
        return np.einsum("ij,ij->i", A, A)

    def einsum(self, subscripts: str, *operands: np.ndarray) -> np.ndarray:
        return np.einsum(subscripts, *operands)

    # ===== Tensor ops =====

    @overload
    def where(self, condition: np.ndarray, true_val: Any, false_val: Any) -> np.ndarray: ...

    @overload
    def where(self, condition: np.ndarray, true_val: None = None, false_val: None = None) -> tuple[np.ndarray, ...]: ...

    def where(self, condition: np.ndarray, true_val=None, false_val=None) -> np.ndarray | tuple[np.ndarray, ...]:
        if true_val is None or false_val is None:
            return np.where(condition)
        return np.where(condition, true_val, false_val)

    def clip(self, A: np.ndarray, min_=None, max_=None) -> np.ndarray:
        return np.clip(A, min_, max_)

    def fill_diagonal(self, A: np.ndarray, val) -> np.ndarray:
        B = self.copy(A)
        np.fill_diagonal(B, val)
        return B

    def diag(self, A: np.ndarray) -> np.ndarray:
        return np.diag(A)

    # ===== Math ops =====

    def isnan(self, A: np.ndarray) -> np.ndarray:
        return np.isnan(A)

    def pow(self, A: np.ndarray, power: int | float | np.ndarray) -> np.ndarray:
        return np.pow(A, power)

    def sqrt(self, A: np.ndarray) -> np.ndarray:
        if np.any(A < 0):
            raise ValueError("Trying to compute the sqrt of negative values.")
        return np.sqrt(A)

    def abs(self, A: np.ndarray) -> np.ndarray:
        return np.abs(A)

    def exp(self, A: np.ndarray) -> np.ndarray:
        return np.exp(A)

    def log(self, A: np.ndarray) -> np.ndarray:
        if np.any(A < 0):
            raise ValueError("Trying to compute the log of negative values.")
        return np.log(A)

    def cos(self, A: np.ndarray) -> np.ndarray:
        return np.cos(A)

    def sin(self, A: np.ndarray) -> np.ndarray:
        return np.sin(A)

    def tan(self, A: np.ndarray) -> np.ndarray:
        return np.tan(A)

    def cosh(self, A: np.ndarray) -> np.ndarray:
        return np.cosh(A)

    def sinh(self, A: np.ndarray) -> np.ndarray:
        return np.sinh(A)

    def tanh(self, A: np.ndarray) -> np.ndarray:
        return np.tanh(A)

    # ===== Linear algebra =====

    def inv(self, A: np.ndarray) -> np.ndarray:
        return scipy.linalg.inv(A)

    def pinv(self, A: np.ndarray) -> np.ndarray:
        result = scipy.linalg.pinv(A, return_rank=False)
        if isinstance(result, tuple):
            return result[0]
        return result

    def solve(self, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        # Batch version doesn't work in Numpy so we create a batch version
        if A.ndim == 3:
            return self.stack([self.solve(A[i], b[i]) for i in range(len(A))], axis=0)
        return np.linalg.solve(A, b)

    def solve_triangular(self, A: np.ndarray, B: np.ndarray, upper: bool = False) -> np.ndarray:
        return scipy.linalg.solve_triangular(A, B, lower=not upper)

    def cholesky(self, A: np.ndarray, upper: bool = False, allow_adding_reg: bool = True) -> np.ndarray:
        # Batch version doesn't work in Numpy so we create a batch version
        if A.ndim == 3:
            return self.stack([self.cholesky(A[i], upper=upper, allow_adding_reg=allow_adding_reg) for i in range(len(A))], axis=0)

        if np.linalg.det(A) <= 0:
            L, info = A, 1
        else:
            L, info = scipy.linalg.lapack.dpotrf(A, lower=not upper)  # type: ignore
        if info != 0:
            if allow_adding_reg:
                A_reg = self.copy(A)
                # Add regularization and try again
                eye = self.eye(A.shape[-1], dtype=A.dtype)
                for eps in range(12, 3, -1):
                    A_reg = A + 10 ** (-eps) * eye
                    # While not spd, find a new eps
                    if np.linalg.det(A_reg) <= 0:
                        continue
                    L, new_info = scipy.linalg.lapack.dpotrf(A_reg, lower=not upper)  # type: ignore
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

    def inverse_cholesky(self, A: np.ndarray, upper: bool = False, allow_adding_reg: bool = True) -> np.ndarray:
        # Batch version doesn't work in Numpy so we create a batch version
        if A.ndim == 3:
            return self.stack([self.inverse_cholesky(A[i], upper=upper, allow_adding_reg=allow_adding_reg) for i in range(len(A))], axis=0)

        L = self.cholesky(A, upper=upper, allow_adding_reg=allow_adding_reg)
        inv, info = scipy.linalg.lapack.dpotri(L, lower=not upper)  # type: ignore
        if info != 0:
            raise ValueError("Could not compute Cholesky inverse. The matrix may not be positive definite.")
        # Make inv symetric
        n = len(inv)
        rows, cols = np.triu_indices(n, k=1) if not upper else np.tril_indices(n, k=-1)
        inv[rows, cols] = inv[cols, rows]
        return inv

    def qr(self, A: np.ndarray, mode="reduced") -> tuple[np.ndarray, np.ndarray]:
        return np.linalg.qr(A, mode=mode)

    def vander(self, a: np.ndarray, degree: int, increasing: bool = True) -> np.ndarray:
        return np.vander(a, degree + 1, increasing=increasing)

    def lstsq(self, A: np.ndarray, B: np.ndarray):
        res = scipy.linalg.lstsq(A, B, lapack_driver="gelsy")
        if res is None:
            raise ValueError("Error in least squares solver. Could not solve the system.")
        return res[0]
