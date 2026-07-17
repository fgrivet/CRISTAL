"""Contains the :class:`Distance <cristal.commons.distance.Distance>` class used in static detectors."""

from typing import Generic, get_args

from ..backend.base_backend import Backend
from ..types import IMPLEMENTED_DISTANCES, ArrayLike, DTypeLike
from .base_commons import BaseCommons


# pylint: disable=unused-variable
class Distance(BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Class to compute the distance between each pair of two collections of inputs.

    Parameters
    ----------
    metric : :class:`IMPLEMENTED_DISTANCES <cristal.types.IMPLEMENTED_DISTANCES>`, optional
        The distance metric to use, by default :const:`euclidean`.

    Attributes
    ----------
    metric : :class:`IMPLEMENTED_DISTANCES <cristal.types.IMPLEMENTED_DISTANCES>`
        The distance metric to use.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    Raises
    ------
    ValueError
        If the distance :const:`metric` is not valid.

    See Also
    --------
    cristal.types.IMPLEMENTED_DISTANCES : For more details on how the methods work.

    Examples
    --------
    >>> distance = Distance(metric="euclidean")
    >>> distance.backend = NumpyBackend()
    >>> X = np.random.rand(10, 5)
    >>> Y = np.random.rand(20, 5)
    >>> D = distance(X, X)  # Computes the distance between X and X resulting shape of (10, 10)
    >>> D = distance(X, Y)  # Computes the distance between X and Y resulting shape of (10, 20)
    """

    requires = ["backend"]

    def __init__(self, metric: IMPLEMENTED_DISTANCES = "euclidean"):
        """Class constructor.
        Define the :attr:`metric` and bind the :attr:`backend`.

        Parameters
        ----------
        metric : :class:`IMPLEMENTED_DISTANCES <cristal.types.IMPLEMENTED_DISTANCES>`, optional
            The distance metric to use, by default :const:`euclidean`.

        Raises
        ------
        ValueError
            If the distance :const:`metric` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_DISTANCES : For more details on how the methods work.
        """
        if metric not in get_args(IMPLEMENTED_DISTANCES):
            raise ValueError(f"metric must be in {IMPLEMENTED_DISTANCES}. Got {metric}.")
        self.metric: IMPLEMENTED_DISTANCES = metric
        """The distance metric to use.

        See Also
        --------
        cristal.types.IMPLEMENTED_DISTANCES : For more details on how the methods work.
        """

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        """The backend to use for the computation."""

    def _cdist_euclidean(self, X: ArrayLike, Y: ArrayLike | None = None, p: int = 2) -> ArrayLike:
        if self.backend is None:
            raise ValueError("A backend must be bound to the Distance class before using it.")

        # Compute X_norms and Y_norms
        X_norms = self.backend.norm2D(X)
        if Y is None:
            new_Y = X
            Y_norms = X_norms
        else:
            new_Y = Y
            Y_norms = self.backend.norm2D(new_Y)

        # Use D = ||X - Y||_2^2 = ||X||_2^2 + ||Y||_2^2^T - 2 * X * Y^T
        D = X_norms[:, None] + Y_norms[None, :] - 2 * X @ new_Y.T

        # Clamp to avoid numerical errors (no negative values)
        D = self.backend.clip(D, min_=0.0)
        # If distances between X and X, diagonal must be 0
        if Y is None:
            D = self.backend.fill_diagonal(D, 0.0)

        # Returns ||X - Y||_2^p
        if p != 2:
            D = D ** (p / 2)
        return D

    def _compute_covariance_matrix(self, X: ArrayLike) -> ArrayLike:
        if self.backend is None:
            raise ValueError("A backend must be bound to the Distance class before using it.")

        # Compute the covariance matrix
        X_mean = self.backend.mean(X, axis=0)
        X_centered = X - X_mean
        N: int = X_centered.shape[0]
        Sigma = (X_centered.T @ X_centered) / (N - 1)
        return Sigma

    def cdist(self, X: ArrayLike, Y: ArrayLike | None = None, p=2) -> ArrayLike:
        """Compute the distance matrix between X and Y.

        Parameters
        ----------
        X : ArrayLike
            The first collection of points in :math:`\\mathbb{R}^d` of shape (M, d).
        Y : ArrayLike | None, optional
            The second collection of points in :math:`\\mathbb{R}^d` of shape (N, d). :attr:`X` if None, by default None.
        p : int, optional
            The power to apply to the distance, by default 2.

        Returns
        -------
        ArrayLike
            The distance between each pair (x, y). A matrix of shape (M, N).

        Examples
        --------
        >>> distance = Distance(metric="euclidean")
        >>> distance.backend = NumpyBackend()
        >>> X = np.random.rand(10, 5)
        >>> Y = np.random.rand(20, 5)
        >>> D = distance(X, X)  # Computes the distance between X and X resulting shape of (10, 10)
        >>> D = distance(X, Y)  # Computes the distance between X and Y resulting shape of (10, 20)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the Distance class before using it.")

        if self.metric == "mahalanobis":
            new_Y = X if Y is None else Y

            # Compute the covariance matrix
            Sigma = self._compute_covariance_matrix(self.backend.concat([X, new_Y], axis=0))

            # Compute the Cholesky decomposition of Sigma
            upper = False
            L = self.backend.cholesky(Sigma, upper=upper, allow_adding_reg=True)

            # Solve Z_X = X^T L^{-1}
            Z_X = self.backend.solve_triangular(L, X.T, upper=upper)
            # Solve Z_Y = Y^T L^{-1}
            Z_Y = Z_X if Y is None else self.backend.solve_triangular(L, new_Y.T, upper=upper)

            # D = D_E(L^-1 X, L^-1 Y)
            D = self._cdist_euclidean(Z_X.T, Z_Y.T, p=p)

        else:
            D = self._cdist_euclidean(X, Y, p=p)

        return D

    def __call__(self, X: ArrayLike, Y: ArrayLike | None = None, p=2) -> ArrayLike:
        """Compute the distance matrix between X and Y.

        .. hint::

            This function is a wrapper for :func:`cdist`.

        Parameters
        ----------
        X : ArrayLike
            The first collection of points in :math:`\\mathbb{R}^d` of shape (M, d).
        Y : ArrayLike | None, optional
            The second collection of points in :math:`\\mathbb{R}^d` of shape (N, d). :attr:`X` if None, by default None.
        p : int, optional
            The power to apply to the distance, by default 2.

        Returns
        -------
        ArrayLike
            The distance between each pair (x, y). A matrix of shape (M, N).
        """
        return self.cdist(X=X, Y=Y, p=p)
