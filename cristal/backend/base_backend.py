from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeGuard, overload

import numpy as np

from ..core.types import ArrayLike, DTypeLike, Number, ShapeType


class Backend(ABC, Generic[ArrayLike, DTypeLike]):
    def __init__(self, default_dtype: DTypeLike):
        self.default_dtype = default_dtype

    # ===== Type =====

    @abstractmethod
    def is_array_like(self, x: Any) -> TypeGuard[ArrayLike]:
        """Check if `x` has a valid type for this backend.

        Parameters
        ----------
        x : Any
            The data to check.

        Returns
        -------
        TypeGuard[ArrayLike]
            True if `x` is a valid type for this backend, False otherwise.
        """

    @abstractmethod
    def to_array_like(self, x: Any, dtype: DTypeLike | None = None) -> ArrayLike:
        """Transform `x` into a valid type for this backend.

        Parameters
        ----------
        x : Any
            The data to convert.

        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The data converted.
        """

    @abstractmethod
    def to_numpy(self, x: ArrayLike) -> np.ndarray:
        """Convert `x` into a numpy array.

        Parameters
        ----------
        x : ArrayLike
            The data to convert.

        Returns
        -------
        np.ndarray
            The data converted into a numpy array.
        """

    # ===== Creation =====

    @abstractmethod
    def zeros(self, shape: ShapeType, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an object of zeros with the given `shape` and `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the object to create.

        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The object filled with zeros of the given `shape` and `dtype`.
        """

    @abstractmethod
    def ones(self, shape: ShapeType, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an object of ones with the given `shape` and `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the object to create.

        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The object filled with ones of the given `shape` and `dtype`.
        """

    @abstractmethod
    def eye(self, n: int, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an identity matrix of size `n x n` with the given `dtype`.

        Parameters
        ----------
        n : int
            The size of the identity matrix to create.

        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The identity matrix of size `n x n` with the given `dtype`.
        """

    @abstractmethod
    def full(self, shape: ShapeType, fill_value: Any, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an object of `fill_value` with the given `shape` and `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the object to create.

        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The object filled with `fill_value` of the given `shape` and `dtype`.
        """

    # TODO overload for only stop
    @abstractmethod
    def arange(self, start: int, stop: int, step: int = 1, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create a range of values from `start` to `stop` with the given `step` and `dtype`.

        Parameters
        ----------
        start : int
            The first value of the range (inclusive).
        stop : int
            The last value of the range (exclusive).
        step : int, optional
            The step size of the range (positive or negative), by default 1
        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The range of values of the given `dtype`.
        """

    # ===== Random =====

    @abstractmethod
    def set_seed(self, seed: int):
        """Fix the random `seed` for reproducibility.

        Parameters
        ----------
        seed : int
            The seed to use.
        """

    @abstractmethod
    def random(self, shape: ShapeType) -> ArrayLike:
        """Draw random samples from a "continuous uniform" distribution over [0.0, 1.0) with the given `shape` and backend `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the generated samples.

        Returns
        -------
        ArrayLike
            The generated samples of given `shape` and backend `dtype`.
        """

    @abstractmethod
    def randn(self, mean: Number | ArrayLike, std: Number | ArrayLike, shape: ShapeType) -> ArrayLike:
        """Draw random samples from a normal (Gaussian) distribution with given `mean`, `std`, `shape` and backend `dtype`.

        Parameters
        ----------
        mean : Number | ArrayLike
            The mean of the normal distribution.
        std : Number | ArrayLike
            The standard deviation of the normal distribution.
        shape : ShapeType
            The shape of the generated samples.

        Returns
        -------
        ArrayLike
            The generated samples of given `shape` and backend `dtype`.
        """

    @abstractmethod
    def randint(self, low: int | ArrayLike, high: int | ArrayLike, shape: ShapeType) -> ArrayLike:
        """Draw random integers from the "discrete uniform" distribution over [`low`, `high`) with given `shape` and backend `dtype`.

        Parameters
        ----------
        low : int | ArrayLike
            The lower boundary of the output interval (inclusive).
        high : int | ArrayLike
            The higher boundary of the output interval (exclusive).
        shape : ShapeType
            The shape of the generated samples.

        Returns
        -------
        ArrayLike
            The generated integers in [`low`, `high`) of given `shape` and backend `dtype`.
        """

    # ===== Shape ops =====
    @abstractmethod
    def swap(self, A: ArrayLike, axis1: int, axis2: int) -> ArrayLike: ...

    @abstractmethod
    def concat(self, arrays: list[ArrayLike], axis: int = 0) -> ArrayLike: ...

    @abstractmethod
    def stack(self, arrays: list[ArrayLike], axis: int) -> ArrayLike: ...

    @abstractmethod
    def broadcast(self, A: ArrayLike, shape: ShapeType) -> ArrayLike: ...

    # ===== Reduction ops =====

    # TODO overload to return ArrayLike if axis is not None and float otherwise

    @abstractmethod
    def sum(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def prod(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def cumsum(self, A: ArrayLike, axis=None) -> ArrayLike: ...

    @abstractmethod
    def min(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def max(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def argmin(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def argmax(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def mean(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def std(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def norm(self, A: ArrayLike, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: bool = False) -> ArrayLike | float: ...

    @abstractmethod
    def norm2D(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def einsum(self, subscripts: str, *operands: ArrayLike) -> ArrayLike: ...

    # ===== Tensor ops =====

    @overload
    def where(self, condition: ArrayLike, true_val: Any, false_val: Any) -> ArrayLike: ...

    @overload
    def where(self, condition: ArrayLike, true_val: None = None, false_val: None = None) -> tuple[ArrayLike, ...]: ...

    @abstractmethod
    def where(self, condition: ArrayLike, true_val=None, false_val=None) -> ArrayLike | tuple[ArrayLike, ...]: ...

    @abstractmethod
    def clip(self, A: ArrayLike, min_: Number | None = None, max_: Number | None = None) -> ArrayLike: ...

    @abstractmethod
    def fill_diagonal(self, A: ArrayLike, val: Number) -> ArrayLike: ...

    @abstractmethod
    def diag(self, A: ArrayLike) -> ArrayLike: ...

    # ===== Math ops =====

    @abstractmethod
    def nan(self) -> float: ...

    @abstractmethod
    def isnan(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def pow(self, A: ArrayLike, power: Number | ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def sqrt(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def abs(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def exp(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def log(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def cos(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def sin(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def tan(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def cosh(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def sinh(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def tanh(self, A: ArrayLike) -> ArrayLike: ...

    # ===== Linear algebra =====

    @abstractmethod
    def inv(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def pinv(self, A: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def solve(self, A: ArrayLike, b: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def solve_triangular(self, A: ArrayLike, B: ArrayLike, upper: bool = False) -> ArrayLike: ...

    @abstractmethod
    def cholesky(self, A: ArrayLike, upper: bool = False, allow_adding_reg: bool = True) -> ArrayLike: ...

    @abstractmethod
    def inverse_cholesky(self, A: ArrayLike, upper: bool = False, allow_adding_reg: bool = True) -> ArrayLike: ...

    @abstractmethod
    def qr(self, A: ArrayLike, mode: Literal["reduced", "complete", "r"] = "reduced") -> tuple[ArrayLike, ArrayLike]: ...

    @abstractmethod
    def vander(self, a: ArrayLike, degree: int, increasing: bool = True) -> ArrayLike: ...

    @abstractmethod
    def lstsq(self, A: ArrayLike, B: ArrayLike) -> ArrayLike: ...
