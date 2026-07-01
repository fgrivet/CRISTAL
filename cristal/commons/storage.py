"""Contains the :class:`Storage <cristal.commons.storage.Storage>` class used in detectors."""

from typing import Iterator, get_args

from ..types import IMPLEMENTED_STORAGES, ArrayLike
from .base_commons import BaseCommons


class Storage(BaseCommons):
    """Class to loop over the data with a limited size to avoid OOM errors.

    Attributes
    ----------
    method : :class:`IMPLEMENTED_STORAGES <cristal.types.IMPLEMENTED_STORAGES>`
        The method to use.
    batch_size : int
        The size of each batch.

    Examples
    --------
    >>> storage = Storage(method="batch", batch_size=4096)
    >>> # Iterate through X
    >>> for X_batch in storage(X):
    >>>     ...
    """

    def __init__(self, method: IMPLEMENTED_STORAGES = "batch", batch_size: int = 4096) -> None:
        """Class constructor.
        Define the storage :attr:`method`, and the :attr:`batch_size`.

        Parameters
        ----------
        method : IMPLEMENTED_STORAGES, optional
            The method to use, by default "batch".
        batch_size : int, optional
            The size of each batch if :attr:`method` is :const:`batch`, by default 4096.

        Raises
        ------
        ValueError
            If :const:`method` is not valid.

            If :const:`batch_size` is lower or equal to 0.

        See Also
        --------
        cristal.types.IMPLEMENTED_STORAGES : For more details on how the storages work.
        """
        if method not in get_args(IMPLEMENTED_STORAGES):
            raise ValueError(f"method must be in {IMPLEMENTED_STORAGES}. Got {method}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive. Got {batch_size}.")

        self.method = method
        """The method to use.
        
        See Also
        --------
        cristal.types.IMPLEMENTED_STORAGES : For more details on how the storages work.
        """
        self.batch_size = batch_size  #: The size of each batch.

    def iterate(self, X: ArrayLike) -> Iterator[ArrayLike]:
        """Allow to loop over the data with a limited size to avoid OOM errors.

        Parameters
        ----------
        X : ArrayLike
            The matrix to iterate over.

        Yields
        ------
        Iterator[ArrayLike]
            X crop in batches of size :attr:`batch_size`.

            If :attr:`method` is :const:`full`, the :attr:`batch_size` is set to :math:`|X|`.

        .. note ::

            The dimension of the last iteration can be lower than :attr:`batch_size`.

        Examples
        --------
        >>> storage = Storage(method="batch", batch_size=4096)
        >>> # Iterate through X
        >>> for X_batch in storage(X):
        >>>     ...
        """
        # Full method
        if self.method == "full":
            self.batch_size = len(X)
            yield X

        # Batch method
        else:
            for i in range(0, len(X), self.batch_size):
                yield X[i : i + self.batch_size]

    def __call__(self, X: ArrayLike) -> Iterator[ArrayLike]:
        """Allow to loop over the data with a limited size to avoid OOM errors.

        .. hint::

            This function is a wrapper for :func:`iterate`.

        Parameters
        ----------
        X : ArrayLike
            The matrix to iterate over.

        Yields
        ------
        Iterator[ArrayLike]
            X crop in batches of size :attr:`batch_size`.

            If :attr:`method` is :const:`full`, the :attr:`batch_size` is set to :math:`|X|`.
        """
        yield from self.iterate(X)
