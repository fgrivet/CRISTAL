"""Contains the :class:`Inverter <cristal.commons.inverter.Inverter>` class used in dynamic detectors."""

from typing import Generic, get_args

from ..backend.base_backend import Backend
from ..types import IMPLEMENTED_INVERTERS, ArrayLike, DTypeLike
from .base_commons import BaseCommons


class Inverter(BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Class to invert a matrix :attr:`X` with optional regularization :attr:`eps`.

    Attributes
    ----------
    method : :class:`IMPLEMENTED_INVERTERS <cristal.types.IMPLEMENTED_INVERTERS>`
        The inverting method to use.
    eps : float | None
        The regularization to add to the matrix before inverting it.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    Examples
    --------
    >>> inverter = Inverter(method="solve", eps=None)
    >>> inverter.backend = NumpyBackend()
    >>> X_inv = inverter(X)
    """

    requires = ["backend"]

    inds_cache = {}

    def __init__(self, method: IMPLEMENTED_INVERTERS = "solve", eps: float | None = None):
        """Class constructor.
        Define the inverting :attr:`method`, the regularization :attr:`eps`, and bind the :attr:`backend`.

        Parameters
        ----------
        method : IMPLEMENTED_INVERTERS, optional
            The inverting method to use, by default "solve".
        eps : float | None, optional
            The regularization to add to the matrix before inverting it, by default None.

        Raises
        ------
        ValueError
            If the inverting :const:`method` is not valid.

        See Also
        --------
        cristal.types.IMPLEMENTED_INVERTERS : For more details on how the methods work.
        """
        if method not in get_args(IMPLEMENTED_INVERTERS):
            raise ValueError(f"method must be in {IMPLEMENTED_INVERTERS}. Got {method}.")
        if eps is not None and eps <= 0:
            raise ValueError(f"eps must be positive or None. Got {eps}.")
        self.method = method
        """The inverting method to use.
        
        See Also
        --------
        cristal.types.IMPLEMENTED_INVERTERS : For more details on how the methods work.
        """
        self.eps = eps  #: The regularization to add to the matrix before inverting it.

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]  #: The backend to use for the computation.

    def invert(self, X: ArrayLike, eps: float | None = None) -> ArrayLike:
        """Invert the matrix :attr:`X` with optional regularization :const:`eps`.

        Parameters
        ----------
        X : ArrayLike
            The matrix to invert.
        eps : float | None, optional
            The regularization to add to the matrix before inverting it: :math:`X_{inv} = (X + \\epsilon I)^{-1}`. If None, defaults to :attr:`eps`, by default None.

        Returns
        -------
        ArrayLike
            The inverse of the matrix attr:`X`.

        Examples
        --------
        >>> inverter = Inverter(method="solve", eps=None)
        >>> inverter.backend = NumpyBackend()
        >>> X_inv = inverter(X)
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the Inverter class before using it.")

        eps = eps or self.eps  # Possibly override the default eps value
        if eps is not None:
            X += self.backend.eye(X.shape[-1]) * eps

        # inv
        if self.method == "inv":
            return self.backend.inv(X)

        # pseudo
        if self.method == "pseudo":
            return self.backend.pinv(X)

        # solve
        if self.method == "solve":
            I = self.backend.eye(X.shape[-1])
            if X.ndim == 3:
                I = self.backend.stack([I for _ in range(len(X))], axis=0)
            return self.backend.solve(X, I)

        # fpd
        return self.backend.inverse_cholesky(X, upper=False, allow_adding_reg=True)

    def __call__(self, X: ArrayLike, eps: float | None = None) -> ArrayLike:
        """Invert the matrix :attr:`X` with optional regularization :const:`eps`.

        .. hint::

            This function is a wrapper for :func:`invert`.

        Parameters
        ----------
        X : ArrayLike
            The matrix to invert.
        eps : float | None, optional
            The regularization to add to the matrix before inverting it: :math:`X_{inv} = (X + \\eps I)^{-1}`. If None, defaults to :attr:`eps`, by default None.

        Returns
        -------
        ArrayLike
            The inverse of the matrix attr:`X`.
        """
        return self.invert(X, eps=eps)
