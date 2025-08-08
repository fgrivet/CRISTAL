"""
Classes for signal decomposition using Fourier Transform. \
:class:`FourierDecomposer` decomposes a signal using the Fourier Transform, \
:class:`WindowFourierDecomposer` extends this functionality to handle signals with a sliding window approach.
"""

import numpy as np
from scipy import fftpack

from .base import BaseDecomposer
from .window import BaseWindowDecomposer


class FourierDecomposer(BaseDecomposer):
    """Class for signal decomposition using Fourier Transform."""

    @staticmethod
    def full_decompose(signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return the first half of the Fourier Transform spectrum of the signal.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        np.ndarray (N//2 + 1,)
            The first half of the Fourier Transform spectrum of the signal.
        """
        N = len(signal)
        middle = N // 2
        fs = fftpack.fft(signal)[: middle + 1]  # Take only the first half of the spectrum
        return fs

    @staticmethod
    def decompose(signal: np.ndarray, n_coefs: int, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Decompose the signal into the n_coefs largest Fourier coefficients in absolute value.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of coefficients to retain.
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        tuple[np.ndarray (n_coefs,), np.ndarray (n_coefs,)]
            The indices and values of the largest Fourier coefficients in absolute value in the first half of the spectrum.
        """
        fs = FourierDecomposer.full_decompose(signal, *args, **kwargs)
        ind = np.argsort(np.abs(fs))[::-1][:n_coefs]
        coefs = fs[ind]
        return ind, coefs

    @staticmethod
    def reconstruct(N: int, indices: np.ndarray, coefficients: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Reconstruct the signal from the specified Fourier coefficients at the given indices (should be in the first half of the spectrum).
        Ensure symmetry in the Fourier spectrum by setting the conjugate coefficients in the second half of the spectrum.

        Parameters
        ----------
        N : int
            The length of the original signal.
        indices : np.ndarray (n_coefs,)
            The indices of the Fourier coefficients to use for reconstruction.
        coefficients : np.ndarray (n_coefs,)
            The Fourier coefficients to use for reconstruction.
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.

        Raises
        ------
        ValueError
            If the number of coefficients differ from the number of indices
            or if the indices are not in the first half of the Fourier spectrum.
        """
        BaseDecomposer._check_number_of_coefficients(indices, coefficients)
        middle = N // 2
        fsrec = np.zeros(N, dtype=complex)
        # For each index, set the coefficient in the first half of the spectrum
        for ind, coef in zip(indices, coefficients):
            if ind > middle:
                raise ValueError("Indices must be in the first half of the Fourier spectrum.")
            fsrec[ind] = coef
            # Set the conjugate for the second half of the spectrum to ensure symmetry (except for the mean value, i.e. the frequency at index 0)
            if ind != 0:
                fsrec[N - ind] = np.conj(coef)
        # Reconstruct the signal using the inverse Fourier Transform on the modified spectrum
        f_reconstruct = np.real(fftpack.ifft(fsrec))
        return f_reconstruct


class WindowFourierDecomposer(BaseWindowDecomposer):
    """Class for signal decomposition using Fourier Transform with a sliding window approach."""

    decomposer_cls = FourierDecomposer
