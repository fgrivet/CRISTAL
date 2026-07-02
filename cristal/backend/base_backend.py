"""Contains Base class for all backends."""

from abc import ABC, abstractmethod
from typing import Any, Generic, Literal, TypeGuard, overload

import numpy as np

from ..types import ArrayLike, DTypeLike, Number, ShapeType


class Backend(ABC, Generic[ArrayLike, DTypeLike]):
    """Base class for all backends. Contains all methods that a Backend should implementer in order to be used in CRISTAL."""

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

        Examples
        --------
        >>> is_array_like(ArrayLike([1, 2]))
        True
        >>> is_array_like([1, 2])
        False
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

        Raises
        ------
        ValueError
            If `x` cannot be converted to a valid type for this backend.

        Examples
        --------
        >>> to_array_like([1, 2, 3])
        ArrayLike([1, 2, 3])
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

        Examples
        --------
        >>> to_numpy(ArrayLike([1, 2, 3]))
        array([1, 2, 3])
        """

    # ===== Creation =====

    @abstractmethod
    def empty(self, shape: ShapeType, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an empty object with the given `shape` and `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the object to create.
        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The empty object of the given `shape` and `dtype`.

        Examples
        --------
        >>> empty(3).shape # or equivalently empty((3,))
        (3,)
        >>> zeros((2, 5)).shape
        (2, 5)
        """

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

        Examples
        --------
        >>> zeros(3) # or equivalently zeros((3,))
        ArrayLike([0., 0., 0.])
        >>> zeros((2, 5))
        ArrayLike([[0., 0., 0., 0., 0.],
                   [0., 0., 0., 0., 0.]])
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

        Examples
        --------
        >>> ones(3) # or equivalently ones((3,))
        ArrayLike([1., 1., 1.])
        >>> ones((2, 5))
        ArrayLike([[1., 1., 1., 1., 1.],
                   [1., 1., 1., 1., 1.]])
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

        Examples
        --------
        >>> eye(3)
        array([[1., 0., 0.],
               [0., 1., 0.],
               [0., 0., 1.]])
        """

    @abstractmethod
    def full(self, shape: ShapeType, fill_value: Any, dtype: DTypeLike | None = None) -> ArrayLike:
        """Create an object of `fill_value` with the given `shape` and `dtype`.

        Parameters
        ----------
        shape : ShapeType
            The shape of the object to create.
        fill_value : Any
            The value to fill the array with.
        dtype : DTypeLike | None, optional
            The target dtype of the output. If `None`, the default dtype of the backend, by default `None`.

        Returns
        -------
        ArrayLike
            The object filled with `fill_value` of the given `shape` and `dtype`.

        Examples
        --------
        >>> full(3, 4) # or equivalently full((3,), 4)
        ArrayLike([4., 4., 4.])
        >>> full((2, 5), 8)
        ArrayLike([[8., 8., 8., 8., 8.],
                   [8., 8., 8., 8., 8.]])
        """

    @abstractmethod
    def arange(self, start_or_stop: Number, /, stop: Number | None = None, step: Number = 1, *, dtype: DTypeLike | None = None) -> ArrayLike:
        """Return an evenly spaced array of values with given start, stop, and step.
        If only one argument is given, it is used as `stop` (start defaults to 0, step to 1).
        If two arguments are given, they are used as `start` and `stop` (step defaults to 1).
        If three arguments are given, they are used as `start`, `stop`, and `step`.

        Parameters
        ----------
        start_or_stop : Number
            Start value (inclusive) of the range. If only one argument is provided, this is treated as the stop value (exclusive) and start defaults to 0.
        stop : Number | None, optional
            Stop value (exclusive). Ignored if only one argument is provided, by default None.
        step : Number, optional
            Step size between values, by default 1.
        dtype : DTypeLike | None, optional
            Desired data type for the output ArrayLike, by default None.

        Returns
        -------
        ArrayLike
            An ArrayLike of evenly spaced values in the range [start, stop) with given step.

        Raises
        ------
        ValueError
            If `step` is zero.

        Examples
        --------
        >>> arange(5)           # stop=5
        ArrayLike([0, 1, 2, 3, 4])
        >>> arange(1, 5)        # start=1, stop=5
        ArrayLike([1, 2, 3, 4])
        >>> arange(1, 5, 2)     # start=1, stop=5, step=2
        ArrayLike([1, 3])
        """

    @abstractmethod
    def copy(self, A: ArrayLike) -> ArrayLike:
        """Create a copy of `A`.

        Parameters
        ----------
        A : ArrayLike
            The array to copy.

        Returns
        -------
        ArrayLike
            The copy of `A`.

        Examples
        --------
        >>> A = ArrayLike([1, 2, 3])
        >>> B = copy(A)
        >>> B[0] = 10
        >>> A[0]  # Original A remains unchanged
        1
        """

    # ===== Random =====
    @abstractmethod
    def set_seed(self, seed: int):
        """Fix the random `seed` for reproducibility.

        Parameters
        ----------
        seed : int
            The seed to use.

        Examples
        --------
        >>> set_seed(42)
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

        Examples
        --------
        >>> random((2, 2))
        ArrayLike([[0.1, 0.5],
                   [0.9, 0.2]])
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

        Raises
        ------
        ValueError
            If `std` is negative.

        Examples
        --------
        >>> randn(0, 1, (2, 2))
        ArrayLike([[0.5, -0.1],
                   [0.2, 1.5]])
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

        Raises
        ------
        ValueError
            If `low` >= `high`.

        Examples
        --------
        >>> randint(0, 5, (2, 2))
        ArrayLike([[1, 4],
                   [0, 3]])
        """

    # ===== Shape ops =====
    @abstractmethod
    def swap(self, A: ArrayLike, axis1: int, axis2: int) -> ArrayLike:
        """Swap the order of two axes of an ArrayLike.

        Parameters
        ----------
        A : ArrayLike
            The array to swap.
        axis1 : int
            The first axis to swap.
        axis2 : int
            The second axis to swap.

        Returns
        -------
        ArrayLike
            The ArrayLike `A` with axes `axis1` and `axis2` swapped.

        Raises
        ------
        ValueError
            If `axis1` or `axis2` are out of bounds for `A`.

        Examples
        --------
        >>> A = ArrayLike([[[3, 3, 4],
                            [3, 1, 2]],
                           [[3, 3, 4],
                            [4, 4, 4]],
                           [[1, 2, 4],
                            [1, 1, 4]],
                           [[2, 1, 1],
                            [0, 4, 0]]])
        >>> swap(A, 2, 1) # or equivalently swap(A, 1, 2)
        ArrayLike([[[3, 3],
                    [3, 1],
                    [4, 2]],
                   [[3, 4],
                    [3, 4],
                    [4, 4]],
                   [[1, 1],
                    [2, 1],
                    [4, 4]],
                   [[2, 0],
                    [1, 4],
                    [1, 0]]])
        """

    @abstractmethod
    def concat(self, arrays: list[ArrayLike], axis: int = 0) -> ArrayLike:
        """Join a sequence of arrays along an existing axis.

        Parameters
        ----------
        arrays : list[ArrayLike]
            A list of arrays to concatenate. All arrays must have the same shape except along the `axis`.
        axis : int, optional
            The axis along which to join, by default 0.

        Returns
        -------
        ArrayLike
            The concatenated array.

        Raises
        ------
        ValueError
            If arrays have mismatching shapes along non-concatenation axes.

        Examples
        --------
        >>> concat([ArrayLike([1, 2]), ArrayLike([3, 4])], axis=0)
        ArrayLike([1, 2, 3, 4])
        >>> concat([ArrayLike([[1]]), ArrayLike([[2]])], axis=0)
        ArrayLike([[1], [2]])
        >>> concat([ArrayLike([[1]]), ArrayLike([[2]])], axis=1)
        ArrayLike([[1, 2]])
        """

    @abstractmethod
    def stack(self, arrays: list[ArrayLike], axis: int) -> ArrayLike:
        """Join a sequence of arrays along a new axis.

        Parameters
        ----------
        arrays : list[ArrayLike]
            A list of arrays to stack. All arrays must have the same shape.
        axis : int
            The axis in the result along which the input arrays are stacked.

        Returns
        -------
        ArrayLike
            The stacked array.

        Examples
        --------
        >>> stack([ArrayLike([1, 2, 3]), ArrayLike([4, 5, 6])], axis=0)
        ArrayLike([[1, 2, 3],
                   [4, 5, 6]])
        >>> stack([ArrayLike([1, 2, 3]), ArrayLike([4, 5, 6])], axis=1)
        ArrayLike([[1, 4],
                   [2, 5],
                   [3, 6]])
        """

    @abstractmethod
    def broadcast(self, A: ArrayLike, shape: ShapeType) -> ArrayLike:
        """Expand arrays to match shape.

        Parameters
        ----------
        A : ArrayLike
            The array to broadcast.
        shape : ShapeType
            The target shape.

        Returns
        -------
        ArrayLike
            The array `A` broadcast to the given `shape`.

        Raises
        ------
        ValueError
            If `A` cannot be broadcast to `shape`.

        Examples
        --------
        >>> broadcast(ArrayLike([1, 2]), (4, 2))
        ArrayLike([[1, 2],
                   [1, 2],
                   [1, 2],
                   [1, 2]])
        """

    # ===== Reduction ops =====
    @overload
    def sum(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def sum(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def sum(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the sum of array elements over a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the sum is computed. If `None`, sum all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The sum of the elements.

        Examples
        --------
        >>> sum(ArrayLike([[1, 2], [3, 4]]))
        10.0
        >>> sum(ArrayLike([[1, 2], [3, 4]]), axis=0)
        ArrayLike([4, 6])
        >>> sum(ArrayLike([[1, 2], [3, 4]]), axis=0, keepdims=True)
        ArrayLike([[4, 6]])
        >>> sum(ArrayLike([[1, 2], [3, 4]]), axis=1)
        ArrayLike([3, 7])
        >>> sum(ArrayLike([[1, 2], [3, 4]]), axis=1, keepdims=True)
        ArrayLike([[3],
                   [7]])
        """

    @overload
    def prod(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def prod(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def prod(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the product of array elements over a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the product is computed. If `None`, prod all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The product of the elements.

        Examples
        --------
        >>> prod(ArrayLike([[1, 2], [3, 4]]))
        24.0
        >>> prod(ArrayLike([[1, 2], [3, 4]]), axis=0)
        ArrayLike([3, 8])
        >>> prod(ArrayLike([[1, 2], [3, 4]]), axis=0, keepdims=True)
        ArrayLike([[3, 8]])
        >>> prod(ArrayLike([[1, 2], [3, 4]]), axis=1)
        ArrayLike([2, 12])
        >>> prod(ArrayLike([[1, 2], [3, 4]]), axis=1, keepdims=True)
        ArrayLike([[2],
                   [12]])
        """

    @abstractmethod
    def cumsum(self, A: ArrayLike, axis=None) -> ArrayLike:
        """Return the cumulative sum of array elements over a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the cumulative sum is computed. If `None`, sum all elements into 1D array, by default None.

        Returns
        -------
        ArrayLike
            The cumulative sum of the elements.

        Examples
        --------
        >>> cumsum(ArrayLike([1, 2, 3]))
        ArrayLike([1, 3, 6])
        >>> cumsum(ArrayLike([[1, 2], [3, 4]]), axis=0)
        ArrayLike([[1, 3, 6, 10]])
        >>> cumsum(ArrayLike([[1, 2], [3, 4]]), axis=0)
        ArrayLike([[1, 2],
                   [4, 6]])
        """

    @overload
    def min(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def min(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def min(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the minimum value along a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the minimum is computed. If `None`, find minimum over all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The minimum value.

        Examples
        --------
        >>> min(ArrayLike([[1, 2], [3, 4]]))
        1.0
        >>> min(ArrayLike([[1, 2], [3, 4]]), axis=1)
        ArrayLike([1, 3])
        """

    @overload
    def max(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def max(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def max(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the maximum value along a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the maximum is computed. If `None`, find maximum over all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The maximum value.

        Examples
        --------
        >>> max(ArrayLike([[1, 2], [3, 4]]))
        4.0
        >>> max(ArrayLike([[1, 2], [3, 4]]), axis=1)
        ArrayLike([2, 4])
        """

    @overload
    def argmin(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> int: ...
    @overload
    def argmin(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def argmin(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | int:
        """Return indices of the minimum values along a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which to index. If `None`, flatten and find global index, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | int
            Indices of the minimum values.

        Examples
        --------
        >>> argmin(ArrayLike([[1, 2], [3, 4]]))
        0
        >>> argmin(ArrayLike([[1, 2], [4, 3]]), axis=1)
        ArrayLike([0, 1])
        """

    @overload
    def argmax(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> int: ...
    @overload
    def argmax(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def argmax(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | int:
        """Return indices of the maximum values along a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which to index. If `None`, flatten and find global index, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | int
            Indices of the maximum values.

        Examples
        --------
        >>> argmax(ArrayLike([[1, 2], [3, 4]]))
        3
        >>> argmax(ArrayLike([[1, 2], [4, 3]]), axis=1)
        ArrayLike([1, 0])
        """

    @overload
    def quantile(self, A: ArrayLike, q: float | ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def quantile(self, A: ArrayLike, q: float | ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def quantile(self, A: ArrayLike, q: float | ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the q-th quantile along a specified axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        q : float | ArrayLike
            Quantile or sequence of quantiles to compute, which must be between 0 and 1 inclusive.
        axis : int | None, optional
            Axis along which the quantile is computed. If `None`, quantile over all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The quantile value.

        Raises
        ------
        ValueError
            If q is outside the range [0, 1].

        Examples
        --------
        >>> quantile(ArrayLike([[1, 2], [3, 4]]), 0.5)
        2.5
        >>> quantile(ArrayLike([[1, 2], [3, 4]]), 0.25, axis=1)
        ArrayLike([1.25, 3.25])
        """

    @overload
    def mean(self, A: ArrayLike, axis: None = None, keepdims: bool = False) -> float: ...
    @overload
    def mean(self, A: ArrayLike, axis: int, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def mean(self, A: ArrayLike, axis=None, keepdims: bool = False) -> ArrayLike | float:
        """Return the arithmetic mean along a specified axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the mean is computed. If `None`, mean over all elements, by default None.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The mean value.

        Raises
        ------
        ZeroDivisionError
            If computing mean over an empty array.

        Examples
        --------
        >>> mean(ArrayLike([[1, 2], [3, 4]]))
        2.5
        >>> mean(ArrayLike([[1, 2], [3, 4]]), axis=1)
        ArrayLike([1.5, 3.5])
        """

    @overload
    def std(self, A: ArrayLike, axis: None = None, ddof: int = 0, keepdims: bool = False) -> float: ...
    @overload
    def std(self, A: ArrayLike, axis: int, ddof: int = 0, keepdims: bool = False) -> ArrayLike: ...
    @abstractmethod
    def std(self, A: ArrayLike, axis=None, ddof: int = 0, keepdims: bool = False) -> ArrayLike | float:
        """Return the standard deviation of the array elements along a given axis.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        axis : int | None, optional
            Axis along which the standard deviation is computed. If `None`, compute over all elements, by default None.
        ddof : int, optional
            Delta degrees of freedom, by default 0.
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The standard deviation value.

        Examples
        --------
        >>> std(ArrayLike([1, 2, 3]))
        0.816...
        >>> std(ArrayLike([[1, 2], [3, 4]]), axis=0)
        ArrayLike([1, 1])
        """

    @overload
    def norm(self, A: ArrayLike, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: Literal[False] = False) -> float: ...
    @overload
    def norm(self, A: ArrayLike, p: Literal["inf", "-inf", "fro", "nuc"] | int, keepdims: Literal[True]) -> ArrayLike: ...
    @abstractmethod
    def norm(self, A: ArrayLike, p: Literal["inf", "-inf", "fro", "nuc"] | int = "fro", keepdims: bool = False) -> ArrayLike | float:
        """Return the matrix norm or vector norm of the array.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        p : Literal["inf", "-inf", "fro", "nuc"] | int, optional
            Order of norm. `fro` for Frobenius norm, `nuc` for nuclear norm, `inf` for infinity norm, or any integer, by default "fro".
        keepdims : bool, optional
            If `True`, the axes which are reduced are left in the result as dimensions with size one, by default False.

        Returns
        -------
        ArrayLike | float
            The norm of the array.

        Examples
        --------
        >>> norm(ArrayLike([[1, 2], [3, 4]]))
        5.477...
        >>> norm(ArrayLike([[1, 2], [3, 4]]), p=1)
        6.0
        >>> norm(ArrayLike([[1, 2], [3, 4]]), p=1, keepdims=True)
        ArrayLike([[6.]])
        """

    @abstractmethod
    def norm2D(self, A: ArrayLike) -> ArrayLike:
        """Compute the squared 2-norm (Euclidean norm) for each row of a 2D array.

        For a matrix of shape (m, n), return a vector of shape (m,) containing the sum of the squares of the elements for each row.

        .. math::

            \\|\\mathbf{A}_{i, :}\\|_2^2 = \\sum_{j=1}^{n} A_{i,j}^2

        Parameters
        ----------
        A : ArrayLike
            The input 2D ArrayLike.

        Returns
        -------
        ArrayLike
            Vector of 2-norms for each row.

        Raises
        ------
        ValueError
            If `A` is not 2D.

        Examples
        --------
        >>> norm2D(ArrayLike([[1, 2], [3, 4]]))
        ArrayLike([5, 25])
        """

    @abstractmethod
    def einsum(self, subscripts: str, *operands: ArrayLike) -> ArrayLike:
        """Perform an Einstein summation convention on the input arrays.

        Parameters
        ----------
        subscripts : str
            The subscripts string for the Einstein summation.
        *operands : ArrayLike
            The input arrays.

        Returns
        -------
        ArrayLike
            The result of the Einstein summation.

        Examples
        --------
        >>> einsum("ij,jk->ik", ArrayLike([[1, 2], [3, 4]]), ArrayLike([[5, 6], [7, 8]]))
        ArrayLike([[19, 22],
                   [43, 50]])
        """

    # ===== Tensor ops =====

    @overload
    def where(self, condition: ArrayLike, true_val: Any, false_val: Any) -> ArrayLike: ...
    @overload
    def where(self, condition: ArrayLike, true_val: None = None, false_val: None = None) -> tuple[ArrayLike, ...]: ...
    @abstractmethod
    def where(self, condition: ArrayLike, true_val=None, false_val=None) -> ArrayLike | tuple[ArrayLike, ...]:
        """Return elements chosen from `true_val` or `false_val` depending on `condition`.

        Parameters
        ----------
        condition : ArrayLike
            Where True, yield `true_val`, otherwise yield `false_val`.
        true_val : Any, optional
            Values from which to choose where `condition` is True.
        false_val : Any, optional
            Values from which to choose where `condition` is False.

        Returns
        -------
        ArrayLike | tuple[ArrayLike, ...]
            If `true_val` and `false_val` are given, returns `ArrayLike`. Otherwise, returns indices where `condition` is True, i.e. tuple[ArrayLike, ...].

        Examples
        --------
        >>> x = ArrayLike([0, 1, 2])
        >>> where(x > 1, 10, 0)
        ArrayLike([0, 0, 10])
        >>> where(x > 1)
        (ArrayLike([2]),)
        """

    @abstractmethod
    def clip(self, A: ArrayLike, min_: Number | None = None, max_: Number | None = None) -> ArrayLike:
        """Clip (limit) the values in an array to a specified range.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        min_ : Number | None, optional
            Minimum value. If `None`, no lower limit is applied, by default None.
        max_ : Number | None, optional
            Maximum value. If `None`, no upper limit is applied, by default None.

        Returns
        -------
        ArrayLike
            The clipped array.

        Examples
        --------
        >>> clip(ArrayLike([-1, 0, 1, 2]), 0, 1)
        ArrayLike([0, 0, 1, 1])
        """

    @abstractmethod
    def fill_diagonal(self, A: ArrayLike, val: Number) -> ArrayLike:
        """Fill the main diagonal of the given array with a scalar value.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        val : Number
            The value to fill the diagonal with.

        Returns
        -------
        ArrayLike
            A copy of the array with the diagonal filled.

        Raises
        ------
        ValueError
            If `A` is not 2D.

        Examples
        --------
        >>> fill_diagonal(ArrayLike([[0, 0], [0, 0]]), 5)
        ArrayLike([[5, 0],
                   [0, 5]])
        """

    @abstractmethod
    def diag(self, A: ArrayLike) -> ArrayLike:
        """Extract a diagonal from an array or construct a diagonal array.

        Parameters
        ----------
        A : ArrayLike
            If 2D, extracts diagonal. If 1D, creates diagonal matrix.

        Returns
        -------
        ArrayLike
            The extracted diagonal vector or the diagonal matrix.

        Examples
        --------
        >>> A = ArrayLike([[1, 2], [3, 4]])
        >>> diag(A)
        ArrayLike([1, 4])
        >>> diag(ArrayLike([1, 2, 3]))
        ArrayLike([[1, 0, 0],
                   [0, 2, 0],
                   [0, 0, 3]])
        """

    # ===== Math ops =====

    @abstractmethod
    def sign(self, A: ArrayLike) -> ArrayLike:
        """Calculate the sign value element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The sign value of each element.

        Examples
        --------
        >>> sign(ArrayLike([-2, -1, 0, 1, 1]))
        ArrayLike([-1, -1, 0, 1, 1])
        """

    @abstractmethod
    def isnan(self, A: ArrayLike) -> ArrayLike:
        """Test element-wise for NaN and return result as a boolean ArrayLike.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            Boolean array where True indicates NaN.

        Examples
        --------
        >>> isnan(ArrayLike([1.0, float('nan'), 3.0]))
        ArrayLike([False, True, False])
        """

    @abstractmethod
    def pow(self, A: ArrayLike, power: Number | ArrayLike) -> ArrayLike:
        """Raise elements of the array to a power.

        Parameters
        ----------
        A : ArrayLike
            The input array.
        power : Number | ArrayLike
            The exponent(s).

        Returns
        -------
        ArrayLike
            The result of raising `A` to `power`.

        Examples
        --------
        >>> pow(ArrayLike([1, 2, 3]), 2)
        ArrayLike([1, 4, 9])
        >>> pow(ArrayLike([1, 2, 3]), ArrayLike([1, 3, 4]))
        ArrayLike([ 1,  8, 81])
        """

    @abstractmethod
    def sqrt(self, A: ArrayLike) -> ArrayLike:
        """Return the non-negative square-root of an array, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The square-root of each element.

        Raises
        ------
        ValueError
            If negative values are encountered in real-valued arrays.

        Examples
        --------
        >>> sqrt(ArrayLike([4, 9, 16]))
        ArrayLike([2, 3, 4])
        """

    @abstractmethod
    def abs(self, A: ArrayLike) -> ArrayLike:
        """Calculate the absolute value element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The absolute value of each element.

        Examples
        --------
        >>> abs(ArrayLike([-1, 0, 1]))
        ArrayLike([1, 0, 1])
        """

    @abstractmethod
    def exp(self, A: ArrayLike) -> ArrayLike:
        """Calculate the exponential of all elements, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The exponential value of each element.

        Examples
        --------
        >>> exp(ArrayLike([0, 1]))
        ArrayLike([1, 2.718...])
        """

    @abstractmethod
    def log(self, A: ArrayLike) -> ArrayLike:
        """Calculate the natural logarithm, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The natural logarithm of each element.

        Raises
        ------
        ValueError
            If negative values are encountered.

        Examples
        --------
        >>> log(ArrayLike([1, 2.718...]))
        ArrayLike([0, 1])
        """

    @abstractmethod
    def cos(self, A: ArrayLike) -> ArrayLike:
        """Calculate the trigonometric cosine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array (angles in radians).

        Returns
        -------
        ArrayLike
            The cosine value of each element.

        Examples
        --------
        >>> cos(ArrayLike([0, np.pi/2]))
        ArrayLike([1, 0])
        """

    @abstractmethod
    def sin(self, A: ArrayLike) -> ArrayLike:
        """Calculate the trigonometric sine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array (angles in radians).

        Returns
        -------
        ArrayLike
            The sine value of each element.

        Examples
        --------
        >>> sin(ArrayLike([0, np.pi/2]))
        ArrayLike([0, 1])
        """

    @abstractmethod
    def tan(self, A: ArrayLike) -> ArrayLike:
        """Calculate the trigonometric tangent, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array (angles in radians).

        Returns
        -------
        ArrayLike
            The tangent value of each element.

        Examples
        --------
        >>> tan(ArrayLike([0, np.pi/4]))
        ArrayLike([0, 1])
        """

    @abstractmethod
    def cosh(self, A: ArrayLike) -> ArrayLike:
        """Calculate the hyperbolic cosine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The hyperbolic cosine of each element.

        Examples
        --------
        >>> cosh(ArrayLike([0]))
        ArrayLike([1])
        """

    @abstractmethod
    def sinh(self, A: ArrayLike) -> ArrayLike:
        """Calculate the hyperbolic sine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The hyperbolic sine of each element.

        Examples
        --------
        >>> sinh(ArrayLike([0]))
        ArrayLike([0])
        """

    @abstractmethod
    def tanh(self, A: ArrayLike) -> ArrayLike:
        """Calculate the hyperbolic tangent, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array.

        Returns
        -------
        ArrayLike
            The hyperbolic tangent of each element.

        Examples
        --------
        >>> tanh(ArrayLike([0]))
        ArrayLike([0])
        """

    @abstractmethod
    def arccos(self, A: ArrayLike) -> ArrayLike:
        """Calculate the trigonometric arccosine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array (angles in radians).

        Returns
        -------
        ArrayLike
            The arccosine value of each element.

        Examples
        --------
        >>> arccos(ArrayLike([-1, 1]))
        ArrayLike([0, pi])
        """

    @abstractmethod
    def arccosh(self, A: ArrayLike) -> ArrayLike:
        """Calculate the trigonometric hyperbolic arccosine, element-wise.

        Parameters
        ----------
        A : ArrayLike
            The input array (angles in radians).

        Returns
        -------
        ArrayLike
            The cosine value of each element.

        Examples
        --------
        >>> arccosh(ArrayLike([1]))
        ArrayLike([0])
        """

    # ===== Linear algebra =====

    @abstractmethod
    def inv(self, A: ArrayLike) -> ArrayLike:
        """Compute the (multiplicative) inverse of a square matrix.

        Parameters
        ----------
        A : ArrayLike
            The input square matrix.

        Returns
        -------
        ArrayLike
            The inverse of `A`.

        Raises
        ------
        ValueError
            If `A` is not square or not invertible.

        Examples
        --------
        >>> A = ArrayLike([[1, 2], [3, 4]])
        >>> inv(A)
        ArrayLike([[-2. , 1. ],
                   [ 1.5, -0.5]])
        """

    @abstractmethod
    def pinv(self, A: ArrayLike) -> ArrayLike:
        """Compute the (Moore-Penrose) pseudoinverse of a matrix.

        Parameters
        ----------
        A : ArrayLike
            The input matrix.

        Returns
        -------
        ArrayLike
            The pseudoinverse of `A`.

        Examples
        --------
        >>> A = ArrayLike([[1, 2], [3, 4]])
        >>> pinv(A)
        ArrayLike([[-2. , 1. ],
                   [ 1.5, -0.5]])
        """

    @abstractmethod
    def solve(self, A: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Solve the linear system Ax = b for x.

        Parameters
        ----------
        A : ArrayLike
            The coefficient matrix.
        b : ArrayLike
            The dependent variable values.

        Returns
        -------
        ArrayLike
            The solution vector `x`.

        Raises
        ------
        ValueError
            If `A` is singular.

        Examples
        --------
        >>> A = ArrayLike([[3, 1], [1, 2]])
        >>> b = ArrayLike([9, 8])
        >>> solve(A, b)
        ArrayLike([2, 3])
        """

    @abstractmethod
    def solve_triangular(self, A: ArrayLike, B: ArrayLike, upper: bool = False) -> ArrayLike:
        """Solve the equation Ax = b where A is triangular.

        Parameters
        ----------
        A : ArrayLike
            The triangular coefficient matrix.
        B : ArrayLike
            The dependent variable values.
        upper : bool, optional
            If True, A is upper triangular. If False, A is lower triangular, by default False.

        Returns
        -------
        ArrayLike
            The solution vector `x`.

        Examples
        --------
        >>> A = ArrayLike([[1, 2], [0, 1]])
        >>> B = ArrayLike([3, 4])
        >>> solve_triangular(A, B, upper=True)
        ArrayLike([1, 4])
        """

    @abstractmethod
    def cholesky(self, A: ArrayLike, upper: bool = False, allow_adding_reg: bool = True) -> ArrayLike:
        """Calculate the Cholesky factorisation of a positive definite matrix A.

        Parameters
        ----------
        A : ArrayLike
            The positive definite input matrix.
        upper : bool, optional
            If True, return the upper triangular Cholesky factor, by default False.
        allow_adding_reg : bool, optional
            If True, allow regularisation if matrix is not strictly positive definite, by default True.

        Returns
        -------
        ArrayLike
            The Cholesky factor `L` such that A = L @ L.T.

        Raises
        ------
        ValueError
            If `A` is not positive definite.

        Examples
        --------
        >>> A = ArrayLike([[4, 12], [12, 17]])
        >>> cholesky(A)
        ArrayLike([[2, 0],
                   [6, 1]])
        """

    @abstractmethod
    def inverse_cholesky(self, A: ArrayLike, upper: bool = False, allow_adding_reg: bool = True) -> ArrayLike:
        """Calculate the inverse of a positive definite matrix A using Cholesky factorisation.

        Parameters
        ----------
        A : ArrayLike
            The positive definite input matrix.
        upper : bool, optional
            Wheter to use the upper triangular factors, by default False.
        allow_adding_reg : bool, optional
            If True, allow regularisation if matrix is not strictly positive definite, by default True.

        Returns
        -------
        ArrayLike
            The inverse of the matrix A.

        Raises
        ------
        ValueError
            If `A` is not positive definite.

        Examples
        --------
        >>> A = ArrayLike([[4, 12], [12, 17]])
        >>> inverse_cholesky(A)
        ArrayLike([[0.5, 0],
                   [-3, 1]])
        """

    @abstractmethod
    def qr(self, A: ArrayLike, mode: Literal["reduced", "complete", "r"] = "reduced") -> tuple[ArrayLike, ArrayLike]:
        """Perform QR decomposition of a matrix A.

        Parameters
        ----------
        A : ArrayLike
            The input matrix.
        mode : Literal["reduced", "complete", "r"], optional
            The size of the returned Q and R matrices. `reduced` gives A=m x n -> (m x n, n x n), `complete` -> (m x m, m x n), `r` -> R only, by default "reduced".

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            Q and R matrices.

        Examples
        --------
        >>> A = ArrayLike([[1, 2], [3, 4]])
        >>> Q, R = qr(A)
        """

    @abstractmethod
    def vander(self, A: ArrayLike, degree: int, increasing: bool = True) -> ArrayLike:
        """Generate a Vandermonde matrix.

        Parameters
        ----------
        a : ArrayLike
            Input values.
        degree : int
            The degree of the Vandermonde matrix.
        increasing : bool, optional
            If True, powers descend from degree-1 to 0, otherwise 0 to degree-1, by default True.

        Returns
        -------
        ArrayLike
            The Vandermonde matrix.

        Examples
        --------
        >>> vander(ArrayLike([1, 2]), 2, increasing=True)
        ArrayLike([[1, 1],
                   [1, 2]])
        """

    @abstractmethod
    def lstsq(self, A: ArrayLike, B: ArrayLike) -> ArrayLike:
        """Return the least-squares solution to a linear matrix equation.

        Parameters
        ----------
        A : ArrayLike
            The coefficient matrix.
        B : ArrayLike
            The dependent variable values.

        Returns
        -------
        ArrayLike
            The least-squares solution `x`.

        Examples
        --------
        >>> A = ArrayLike([[1, 1], [1, -1], [1, 1]])
        >>> B = ArrayLike([2, 0, 4])
        >>> lstsq(A, B)
        ArrayLike([1.666..., 0.666...])
        """
