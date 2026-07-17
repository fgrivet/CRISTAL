"""Contains the :class:`Solver <cristal.commons.solver.Solver>` class used in static detectors."""

from typing import Generic, cast, get_args

from ..backend.base_backend import Backend
from ..types import IMPLEMENTED_SOLVERS, ArrayLike, DTypeLike
from .base_commons import BaseCommons


# pylint: disable=unused-variable
class Solver(BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Class to solve the equation :math:`z = v^T G^{-1} v`.

    Parameters
    ----------
    solver : IMPLEMENTED_SOLVERS, optional
        The solver to use, by default "solve"
    eps : float | None, optional
        The regularization to add to the matrix G before solving the system, by default None

    Attributes
    ----------
    solver : :class:`IMPLEMENTED_SOLVERS <cristal.types.IMPLEMENTED_SOLVERS>`
        The solver to use.
    eps : float | None
        The regularization to add to the matrix G before solving the system.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    Raises
    ------
    ValueError
        If the :const:`solver` is not valid.

    See Also
    --------
    cristal.types.IMPLEMENTED_SOLVERS : For more details on how the solvers work.

    Examples
    --------
    >>> solver = Solver(solver="solve", eps=None)
    >>> solver.backend = NumpyBackend()
    >>> # For a matrix G constructed with N points
    >>> z = solver(G, v, N)
    """

    requires = ["backend"]

    def __init__(self, solver: IMPLEMENTED_SOLVERS = "solve", eps: float | None = None):
        """Class constructor.
        Define the :attr:`solver`, the regularization :attr:`eps`, and bind the :attr:`backend`.

        Parameters
        ----------
        solver : IMPLEMENTED_SOLVERS, optional
            The solver to use, by default "solve"
        eps : float | None, optional
            The regularization to add to the matrix G before solving the system, by default None

        Raises
        ------
        ValueError
            If the :const:`solver` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_SOLVERS : For more details on how the solvers work.
        """
        # eps only for cholesky, inverse and solve, not for QR
        if solver not in get_args(IMPLEMENTED_SOLVERS):
            raise ValueError(f"solver must be in {IMPLEMENTED_SOLVERS}. Got {solver}.")
        self.solver = solver
        """The solver to use.
        
        See Also
        --------
        cristal.types.IMPLEMENTED_SOLVERS : For more details on how the solvers work.
        """

        self.eps = eps
        """The regularization to add to the matrix G before solving the system."""

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        """The backend to use for the computation."""

    def solve(self, V: ArrayLike, v: ArrayLike, N: int) -> ArrayLike:
        """Solve the equation :math:`z = v^T G^{-1} v`.

        Parameters
        ----------
        V : ArrayLike
            - If :const:`solver = qr`, the vandermonde matrix V of shape (N_samples_test, N_samples_train, n+1).
            - Otherwise, the Gram matrix :math:`G = V^T V` of shape (N_samples_test, n+1, n+1).
        v : ArrayLike
            The vector v of shape (n+1,) based on the polynomial basis used.
        N : int
            The number of points used to construct V : N_samples_train.

        Returns
        -------
        ArrayLike
            The scores for each sample. A 1D ArrayLike of shape (N_samples_test,).

        See Also
        --------
        cristal.commons.polynomial_basis.PolynomialBasis.make_v : For more details on how to construct :attr:`v`.

        Examples
        --------
        >>> solver = Solver(solver="solve", eps=None)
        >>> solver.backend = NumpyBackend()
        >>> # For a matrix G constructed with N points
        >>> z = solver(G, v, N)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the Solver class before using it.")

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
            G = (V + self.backend.swap(V, 1, 2)) / 2  # Ensure G is symmetric

            k: int = V.shape[-1]
            dtype = cast(DTypeLike, V.dtype)
            eye = self.backend.eye(k, dtype=dtype)

            # Regularization
            if self.eps is not None:
                G += self.eps * eye

            if self.solver == "inverse":
                # z = v^T G^-1 v with x = G^-1 v but compute explicitly G^-1
                # Inverse G using solve
                G_inv = self.backend.solve(G, self.backend.stack([eye for _ in range(len(G))], axis=0))
                x = G_inv @ v

            elif self.solver == "solve":
                # z = v^T G^-1 v with x = G^-1 v
                x = self.backend.solve(G, v)  # G x = v

            # Cholesky
            else:
                # z = v^T L^-T L^-1 v with y = L^-1 v and x = L^-T y
                L = self.backend.cholesky(G)
                y = self.backend.solve(L, v)  # L y = v
                x = self.backend.solve(self.backend.swap(L, 1, 2), y)  # L^T x = y

        return self.backend.einsum("i,mi->m", v[0, :, 0], x[..., 0])  # z = v^T x

    def __call__(self, V: ArrayLike, v: ArrayLike, N: int) -> ArrayLike:
        """Solve the equation :math:`z = v^T G^{-1} v`.

        .. hint::

            This function is a wrapper for :func:`solve`.

        Parameters
        ----------
        V : ArrayLike
            - If :const:`solver = qr`, the vandermonde matrix V of shape (N_samples_test, N_samples_train, n+1).
            - Otherwise, the Gram matrix :math:`G = V^T V` of shape (N_samples_test, n+1, n+1).
        v : ArrayLike
            The vector v of shape (n+1,) based on the polynomial basis used.
        N : int
            The number of points used to construct V : N_samples_train.

        Returns
        -------
        ArrayLike
            The scores for each sample. A 1D ArrayLike of shape (N_samples_test,).
        """
        return self.solve(V=V, v=v, N=N)
