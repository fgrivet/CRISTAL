"""Contains the :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>` class used in detectors."""

from typing import Generic, cast, get_args

from ..backend.base_backend import Backend
from ..types import IMPLEMENTED_POLYNOMIAL_BASIS, ArrayLike, DTypeLike
from .base_commons import BaseCommons


# pylint: disable=unused-variable
class PolynomialBasis(BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Class to generate vandermonde matrices for either 1D or nD data with the given :attr:`basis`.
    Also contains helper functions related to the basis.

    Parameters
    ----------
    basis : IMPLEMENTED_POLYNOMIAL_BASIS, optional
        The polynomial basis to use, by default "chebyshev".
    normalize : bool, optional
        If :const:`True`, in 1D with :const:`chebyshev` basis, normalize the euclidean distances in :math:`[-1, 1]`
        with :math:`X' = \\frac{X}{2d} - 1`, by default True.

    Attributes
    ----------
    basis : :class:`IMPLEMENTED_POLYNOMIAL_BASIS <cristal.types.IMPLEMENTED_POLYNOMIAL_BASIS>`
        The polynomial basis to use.
    normalize : bool
        If :const:`True`, in 1D with :const:`chebyshev` basis, normalize the euclidean distances in :math:`[-1, 1]`
        with :math:`X' = \\frac{X}{2d} - 1`.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    Raises
    ------
    ValueError
        If the polynomial :const:`basis` is not valid.

    See Also
    --------
    cristal.types.IMPLEMENTED_POLYNOMIAL_BASIS : For more details on how the basis work.

    Examples
    --------
    >>> pb = PolynomialBasis(basis="chebyshev", normalize=True)
    >>>
    >>> # In Univariate mode
    >>> ## With 6D points in a distance matrix D of shape (N_samples_test, N_samples_train), i.e. 1D values
    >>> pb.vandermonde_1d(D, n=4, d=6) # Gives a 3D matrix of shape (N_samples_test, N_samples_train, n+1=5)
    >>> # Helpers
    >>> pb.make_v(n=4, dtype=int) # Gives a 1D vector of shape (n+1=5,) which contains the coefficients based on the basis such that z = v^T G^-1 v
    >>>
    >>> # In Multivariate mode
    >>> ## With 6D points in a matrix X of shape (N_samples_test, d=6)
    >>> pb.vandermonde_nd(X, n=4) # Gives a 2D matrix of shape (N_samples_test, s_d(n)) with s_d(n) = n+d choose n = 210
    >>> ## Helpers
    ### Gives a 2D matrix Alpha of shape (s_d(n)=210, d=6) with Alpha_ij = the coefficient k in P_k(X_j) for the i-th monomials of the basis
    >>> pb.generate_multi_indices_combinations(n=4, d=6)
    """

    requires = ["backend"]

    def __init__(self, basis: IMPLEMENTED_POLYNOMIAL_BASIS = "chebyshev", normalize: bool = True):
        """Class constructor.
        Define the polynomial :attr:`basis`, the normalization :attr:`normalize`, and bind the :attr:`backend`.

        Parameters
        ----------
        basis : IMPLEMENTED_POLYNOMIAL_BASIS, optional
            The polynomial basis to use, by default "chebyshev".
        normalize : bool, optional
            If :const:`True`, in 1D with :const:`chebyshev` basis, normalize the euclidean distances in :math:`[-1, 1]`
            with :math:`X' = \\frac{X}{2d} - 1`, by default True.

        Raises
        ------
        ValueError
            If the polynomial :const:`basis` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_POLYNOMIAL_BASIS : For more details on how the basis work.
        """
        if basis not in get_args(IMPLEMENTED_POLYNOMIAL_BASIS):
            raise ValueError(f"basis must be in {IMPLEMENTED_POLYNOMIAL_BASIS}. Got {basis}.")
        self.basis = basis
        """The polynomial basis to use.
        
        See Also
        --------
        cristal.types.IMPLEMENTED_POLYNOMIAL_BASIS : For more details on how the basis work.
        """

        self.normalize = normalize
        """If :const:`True`, in 1D with :const:`chebyshev` basis, normalize the euclidean distances in :math:`[-1, 1]`
        with :math:`X' = \\frac{X}{2d} - 1`, by default True."""

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        """The backend to use for the computation."""

    def _scale(self, X: ArrayLike, normalize: bool | None, d) -> ArrayLike:
        # If no parameter is passed, takes self.normalize as default value
        if normalize is None:
            normalize = self.normalize

        # If chebyshev and normalize, then apply normalization
        if normalize and self.basis == "chebyshev":
            X_scaled = X / (2 * d) - 1
            return X_scaled

        # Otherwise return X
        return X

    def generate_multi_indices_combinations(self, n: int, d: int) -> ArrayLike:
        """Compute the matrix containing the indices of the basis R_n[x].

        Parameters
        ----------
        n : int
            The maximum degree of polynomials.
        d : int
            The dimension of the data.

        Returns
        -------
        ArrayLike
            Matrix of shape (s_d(n), d) with :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ d \\end{pmatrix}`.

            Each row of the matrix :math:`\\alpha_i` represents the i-th element of the basis :math:`\\mathbb{R}_n[x]`.

        Examples
        --------
        >>> pb = PolynomialBasis(basis="chebyshev", normalize=True)
        # Gives a 2D matrix Alpha of shape (s_d(n)=210, d=6) with Alpha_ij = the coefficient k in P_k(X_j) for the i-th monomials of the basis
        >>> pb.generate_multi_indices_combinations(n=4, d=6)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")
        if d <= 0:
            raise ValueError(f"d must be positive. Got {d}.")

        def generate_exact_degree(total_degree: int, dims_left: int):
            if dims_left == 1:
                yield (total_degree,)
                return

            for value in range(total_degree, -1, -1):
                for tail in generate_exact_degree(total_degree - value, dims_left - 1):
                    yield (value,) + tail

        indices_comb = [indices for total_degree in range(n + 1) for indices in (generate_exact_degree(total_degree, d))]
        return self.backend.to_array_like(indices_comb, dtype=int)  # pyright: ignore[reportArgumentType]

    # ---------- 1D ----------
    def vandermonde_1d(self, X: ArrayLike, n: int, d: float, normalize: bool | None = None) -> ArrayLike:
        """Compute P_k(x) for all :math:`x = X_{ij}` in X and for all k in [0, n].

        Parameters
        ----------
        X : ArrayLike
            A 2D ArrayLike of shape (M, N).
        n : int
            The maximum degree of polynomials.
        d : float
            The dimension of the original data to normalize the points in X
            if :attr:`normalize` is :const:`True` and :attr:`basis` is :const:`chebyshev`.
        normalize : bool | None, optional
            Wether to normalize the points in X in :math:`[-1, 1]` with :math:`X' = \\frac{X}{2d} - 1` if :attr:`basis` is :const:`chebyshev`.
            If :const:`None`, the attribute :attr:`normalize` of the class is used, by default None.

        Returns
        -------
        ArrayLike
            A 3D ArrayLike of shape (M, N, n + 1) with for each point in X, :math:`V_{ijk} = P_{k}(X_{ij})`.

        Raises
        ------
        ValueError
            If X is not a 2D ArrayLike.

            If n or d are not positive.

        Examples
        --------
        >>> pb = PolynomialBasis(basis="chebyshev", normalize=True)
        >>> # With 6D points in a distance matrix D of shape (N_samples_test, N_samples_train), i.e. 1D values
        >>> pb.vandermonde_1d(D, n=4, d=6) # Gives a 3D matrix of shape (N_samples_test, N_samples_train, n+1=5)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")
        if d <= 0:
            raise ValueError(f"d must be positive. Got {d}.")

        k = n + 1
        dtype = cast(DTypeLike, X.dtype)

        # Monomials
        if self.basis == "monomials":
            powers = self.backend.arange(0, k, dtype=dtype).reshape((1, 1, k))
            return self.backend.pow(X[:, :, None], powers)

        # Chebyshev
        X = self._scale(X, normalize=normalize, d=d)
        ks = self.backend.arange(0, k, dtype=dtype)  # (k,) = (n+1,)

        abs_X = self.backend.abs(X)
        mask = abs_X > 1.0  # True where cosh path is needed (rare)

        X_cos = self.backend.clip(X, -1.0, 1.0)
        V = self.backend.cos(self.backend.arccos(X_cos)[..., None] * ks[None, None, :])  # (M, N, k)

        if mask.any():
            idx = self.backend.where(mask)
            X_out = cast(ArrayLike, X[idx])  # (P, Q)
            abs_X_out = cast(ArrayLike, abs_X[idx])
            eta = self.backend.arccosh(abs_X_out)  # (P, Q)
            cosh_vals = self.backend.cosh(eta[:, None] * ks[None, :])  # (P, Q, k)
            sign_k = self.backend.sign(X_out)[:, None] ** ks[None, :]  # (P, Q, k)
            V[idx] = cosh_vals * sign_k  # type: ignore

        return V

    # ---------- ND ----------
    def vandermonde_nd(self, X: ArrayLike, n: int) -> ArrayLike:
        """Compute P_k(x) for all :math:`x = X_{i}` in X and for all k in [0, n].

        Parameters
        ----------
        X : ArrayLike
            A 2D ArrayLike of shape (M, d)
        n : int
            The maximum degree of polynomials

        Returns
        -------
        ArrayLike
            A 2D ArrayLike of shape (M, s_d(n)) with for each point in X, :math:`V_{ij} = P_{j}(X_{i})`.

        Raises
        ------
        ValueError
            If X is not a 2D ArrayLike.

            If n is not positive.

        Examples
        --------
        >>> pb = PolynomialBasis(basis="chebyshev", normalize=True)
        >>> # With 6D points in a matrix X of shape (N_samples_test, d=6)
        >>> pb.vandermonde_nd(X, n=4) # Gives a 2D matrix of shape (N_samples_test, s_d(n)) with s_d(n) = n+d choose n = 210
        """
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
    def make_v(self, n: int, dtype: DTypeLike, z: int = 0, d: None | int = None) -> ArrayLike:
        """Compute the 1D vector v such that :math:`z = v^T G^{-1} v`.

        .. version-changed:: 0.0.3
            Added parameter :const:`z` to estimate the density of the original measure :math:`\\mu`.

        Parameters
        ----------
        n : int
            The maximum degree of polynomials.
        dtype : DTypeLike
            The data type of the returned ArrayLike.
        z : int >= 0, optional
            The point on which to make v, by default 0.
            For small z, the :class:`UCF <cristal.core.detectors.univariate.UCF>` gives an approximation of the density of :math:`\\mu`.
        d : None | int, optional
            The dimension of the original data to normalize :attr:`z` if :attr:`basis` is :const:`chebyshev`.


        Returns
        -------
        ArrayLike
            The vector v of shape (n+1,):

            If basis is :const:`chebyshev` : :math:`v = \\phi(x=0) = \\phi(t=-1) == [1, -1, 1, -1, \\cdots]`.

            If basis is :const:`monomials` : :math:`v = [1, 0, 0, 0, 0, \\cdots]`.

        Raises
        ------
        ValueError
            If n is not positive.

        Examples
        --------
        >>> pb = PolynomialBasis(basis="chebyshev", normalize=True)
        # Gives a 1D vector of shape (n+1=5,) which contains the coefficients based on the basis such that z = v^T G^-1 v
        >>> pb.make_v(n=4, dtype=int)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the PolynomialBasis class before using it.")
        if n <= 0:
            raise ValueError(f"n must be positive. Got {n}.")
        if z < 0:
            raise ValueError(f"z must be >= 0. Got {z}.")
        elif z > 0 and d is None:
            raise ValueError("d must be passed if z is greater than 0.")

        z_array = self.backend.to_array_like([z]).reshape(1, 1)

        # Monomials
        if self.basis == "monomials":
            if z == 0:
                v = self.backend.zeros((n + 1,), dtype=dtype)
                v[0] = 1
            else:
                v = self.vandermonde_nd(z_array, n).reshape((n + 1,))
            return v

        # Chebyshev
        if z == 0:
            v = self.backend.ones((n + 1,), dtype=dtype)
            v[1::2] = -1
        else:
            v = self.vandermonde_nd(self._scale(z_array, None, d), n).reshape((n + 1,))
        return v
