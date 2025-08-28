"""
Base class for signal decomposition methods using a sliding window approach.
"""

import logging
from abc import ABC
from collections import Counter

import numpy as np

from .base import BaseDecomposer

logger = logging.getLogger(__name__)


# TODO Ajouter un paramètre `shift` pour le décalage entre chaque fenêtre
# TODO Implémenter la bande de fréquence du papier AAAI
# TODO Ajouter la possibilité de passer un signal déjà découpé en fenêtres


class BaseWindowDecomposer(BaseDecomposer, ABC):
    """Base class for signal decomposition methods using a sliding window approach."""

    decomposer_cls: type[BaseDecomposer]  #: The not windowed decomposer class to be set by child classes
    default_L: int = 100  #: Default window size for decomposition
    default_margin: int | float = 2  #: Default margin for windowed decomposition

    @staticmethod
    def _check_kwargs(kwargs: dict) -> dict:
        """Check and return the kwargs for the decomposer.
        Needs to be implemented by child classes to check specific parameters.

        Parameters
        ----------
        kwargs : dict
            The keyword arguments to check and return.

        Returns
        -------
        dict
            The checked and potentially modified keyword arguments.
        """
        return kwargs

    @staticmethod
    def _check_L(L: int, N: int) -> int:
        """Check if L is a valid window size.

        Parameters
        ----------
        L : int
            The window size to check.
        N : int
            The length of the signal.

        Returns
        -------
        int
            The checked window size.

        Raises
        ------
        ValueError
            If L is not a positive integer.
        """
        # Check if L is a positive integer
        if not isinstance(L, int) or L <= 0:
            raise ValueError(f"L must be a positive integer. Got: {L}")
        # Check if L is greater than the length of the signal
        if L > N:
            logger.warning("L (%d) is greater than the length of the signal (%d). Using L = N = %d.", L, N, N)
            return N
        return L

    @staticmethod
    def _check_margin(margin: int | float) -> int | float:
        """Check if margin is a valid value.

        Parameters
        ----------
        margin : int | float
            The margin value to check.

        Returns
        -------
        int | float
            The checked margin value.

        Raises
        ------
        ValueError
            If margin is not a positive integer or float.
        """
        if not isinstance(margin, (int, float)) or margin <= 0:
            raise ValueError(f"Margin must be a positive integer or float. Got: {margin}")
        return margin

    @staticmethod
    def top_n_most_frequent_indices(indices_list: list[int], n: int) -> np.ndarray:
        """Get the top n most frequent indices from a list.

        Parameters
        ----------
        indices_list : list[int]
            The list of indices to analyze.
        n : int
            The number of top frequent indices to return.

        Returns
        -------
        np.ndarray[int] (n,)
            An array containing the top n most frequent indices.
        """
        frequences = Counter(indices_list)
        return np.array([index for index, _ in frequences.most_common(n)])

    @classmethod
    def full_decompose(cls, signal: np.ndarray, *args, L: int = default_L, **kwargs) -> np.ndarray:
        """Return the full decomposition (all coefficients) of the signal.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        L : int, optional
            The window size for the decomposition, by default :attr:`default_L`.
        args
            See :meth:`full_decompose` for specific positional arguments.
        kwargs
            See :meth:`full_decompose` for specific keyword arguments.

        Returns
        -------
        np.ndarray (n_window=N-L+1, ?)
            The full decomposition of the signal,
            where each row corresponds to the decomposition in ? coefficients (depending on the decomposition method) for each window of size L.
        """
        # Initialize parameters and check validity
        kwargs = cls._check_kwargs(kwargs)
        N = len(signal)
        L = cls._check_L(L, N)
        n_window = N - L + 1

        # Do the first decomposition to initialize the array with the right shape (depending on the decomposer)
        decomposition_0 = cls.decomposer_cls.full_decompose(signal[0:L], *args, **kwargs)
        decompositions = np.zeros((n_window, len(decomposition_0)), dtype=decomposition_0.dtype)
        decompositions[0, :] = decomposition_0

        # Iterate over the windows and collect decompositions
        for i in range(1, n_window):
            decomposition_i = cls.decomposer_cls.full_decompose(signal[i : i + L], *args, **kwargs)
            decompositions[i, :] = decomposition_i

        return decompositions

    @classmethod
    def decompose(
        cls, signal: np.ndarray, n_coefs: int, *args, L: int = default_L, margin: int | float = default_margin, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose the signal into n_coefs components for each window.
        Select the `n_coefs` most represented indices from the `margin*n_coefs` best coefficients of all windows.
        Thus each window will have the same indices ensuring consistency.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of coefficients to retain.
        L : int, optional
            The window size for the decomposition, by default :attr:`default_L`.
        margin : int | float, optional
            The margin for the decomposition, by default :attr:`default_margin`.
        args
            See :meth:`full_decompose` for specific positional arguments.
        kwargs
            See :meth:`full_decompose` for specific keyword arguments.

        Returns
        -------
        tuple[np.ndarray (n_window=N-L+1, n_coefs), np.ndarray (n_window=N-L+1, n_coefs)]
            A tuple containing the indices and coefficients of the decomposition for each window of size L.
        """
        # Add L in kwargs to check its validity
        kwargs["L"] = L
        # Initialize parameters and check validity
        kwargs = cls._check_kwargs(kwargs)
        margin = cls._check_margin(margin)
        n_coefs_with_margin = int(margin * n_coefs)

        # Get the full decomposition of the signal
        decompositions = cls.full_decompose(signal, *args, **kwargs)
        indices = np.argsort(np.abs(decompositions), axis=1)[:, ::-1][:, :n_coefs_with_margin]

        # Get the most frequent indices
        most_frequent_indices = cls.top_n_most_frequent_indices(indices.flatten().tolist(), n_coefs)

        # Get the coefficients corresponding to the most frequent indices
        coefs = decompositions[:, most_frequent_indices]

        return most_frequent_indices, coefs

    @classmethod
    def reconstruct(cls, N: int, indices: np.ndarray, coefficients: np.ndarray, *args, L: int = default_L, **kwargs) -> np.ndarray:
        """Reconstruct the signal from the indices and coefficients.
        Reconstruct each window of size L from the coefficients and indices, then average the results.

        Parameters
        ----------
        N : int
            The length of the original signal.
        indices : np.ndarray (n_coefs,)
            The indices of the coefficients to use for reconstruction.
        coefficients : np.ndarray (n_window=N-L+1, n_coefs)
            The coefficients of each window to use for reconstruction.
        L : int, optional
            The window size for the decomposition, by default :attr:`default_L`.
        args
            See :meth:`reconstruct` of :attr:`decomposer_cls` for specific positional arguments.
        kwargs
            See :meth:`reconstruct` of :attr:`decomposer_cls` for specific keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.

        Raises
        ------
        ValueError
            If the number of coefficient rows does not match the number of windows (N-L+1)
            or if the number of coefficient columns does not match the number of indices.
        """
        # Initialize parameters and check validity
        kwargs = cls._check_kwargs(kwargs)
        L = cls._check_L(L, N)
        n_window = N - L + 1
        if coefficients.shape[0] != n_window:
            raise ValueError(f"The number of coefficient rows must match the number of windows ({n_window}). Got: {coefficients.shape[0]}.")
        if coefficients.shape[1] != len(indices):
            raise ValueError(f"The number of coefficient columns must match the number of indices ({len(indices)}). Got: {coefficients.shape[1]}.")

        # Initialize the reconstructed signal and counts
        reconstructed_signal = np.zeros(N, dtype=float)
        counts = np.zeros(N)  # Counts the number of times each index is used in the reconstruction, to mean the signal correctly

        # Iterate over the windows and reconstruct the signal
        for i in range(n_window):
            window_reconstructed = cls.decomposer_cls.reconstruct(L, indices, coefficients[i, :], *args, **kwargs)
            reconstructed_signal[i : i + L] += window_reconstructed
            counts[i : i + L] += 1

        # Avoid division by zero
        counts[counts == 0] = 1

        # Normalize the reconstructed signal by the counts
        reconstructed_signal = reconstructed_signal / counts
        return reconstructed_signal

    @classmethod
    def get_coefficients_at_indices(
        cls, signal: np.ndarray, indices: np.ndarray, *args, L: int = default_L, margin: int | float = default_margin, **kwargs
    ) -> np.ndarray:
        """Get the coefficients of the signal at the specified indices for each window.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        indices : np.ndarray (n_coefs,)
            The indices at which to retrieve the coefficients.
        L : int, optional
            The window size for the decomposition, by default :attr:`default_L`.
        margin : int | float, optional
            The margin for the decomposition, by default :attr:`default_margin`.
        args
            See :meth:`full_decompose` for specific positional arguments.
        kwargs
            See :meth:`full_decompose` for specific keyword arguments.

        Returns
        -------
        np.ndarray (n_window=N-L+1, n_coefs)
            The coefficients of the signal at the specified indices.
        """
        coefficients = cls.full_decompose(signal, *args, L=L, margin=margin, **kwargs)
        return coefficients[:, indices]
