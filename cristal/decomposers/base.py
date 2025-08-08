"""
Base class for signal decomposition methods.
"""

from abc import ABC, abstractmethod

import numpy as np


class BaseDecomposer(ABC):
    """Base class for signal decomposition methods."""

    @staticmethod
    def _check_number_of_coefficients(indices: np.ndarray, coefficients: np.ndarray) -> None:
        """Check if the number of coefficients matches the number of indices.

        Parameters
        ----------
        indices : np.ndarray (n_coefs,)
            The indices corresponding to the coefficients.
        coefficients : np.ndarray (n_coefs,)
            The coefficients to check.

        Raises
        ------
        ValueError
            If the number of coefficients does not match the number of indices.
        """
        if coefficients.shape[0] != len(indices):
            raise ValueError(f"The number of coefficients must match the number of indices ({len(indices)}). Got: {coefficients.shape[0]}.")

    @staticmethod
    @abstractmethod
    def full_decompose(signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return the full decomposition (all coefficients) of the signal.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        np.ndarray (?,)
            The full decomposition of the signal, where ? is the number of coefficients and depends on the decomposition method.
        """

    @staticmethod
    @abstractmethod
    def decompose(signal: np.ndarray, n_coefs: int, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Decompose the signal into n_coefs components.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of coefficients to retain.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        tuple[np.ndarray (n_coefs,), np.ndarray (n_coefs,)]
            A tuple containing the indices and coefficients of the decomposition.
        """

    @staticmethod
    @abstractmethod
    def reconstruct(N: int, indices: np.ndarray, coefficients: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Reconstruct the signal from the indices and coefficients.

        Parameters
        ----------
        N : int
            The length of the original signal.
        indices : np.ndarray (n_coefs,)
            The indices of the coefficients to use for reconstruction.
        coefficients : np.ndarray (n_coefs,)
            The coefficients to use for reconstruction.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.
        """

    @classmethod
    def transform(cls, signal: np.ndarray, n_coefs: int, *args, **kwargs) -> np.ndarray:
        """Reconstruct the signal from its decomposition into n_coefs components.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of coefficients to retain.
        args
            See :meth:`decompose` and :meth:`reconstruct` for specific positional arguments.
        kwargs
            See :meth:`decompose` and :meth:`reconstruct` for specific keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.
        """
        indices, coefficients = cls.decompose(signal, n_coefs, *args, **kwargs)
        return cls.reconstruct(len(signal), indices, coefficients, *args, **kwargs)

    @classmethod
    def get_coefficients_at_indices(cls, signal: np.ndarray, indices: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Get the coefficients of the signal at the specified indices.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        indices : np.ndarray (n_coefs,)
            The indices at which to retrieve the coefficients.
        args
            See :meth:`full_decompose` for specific positional arguments.
        kwargs
            See :meth:`full_decompose` for specific keyword arguments.

        Returns
        -------
        np.ndarray (n_coefs,)
            The coefficients of the signal at the specified indices.
        """
        coefficients = cls.full_decompose(signal, *args, **kwargs)
        return coefficients[indices]
