"""
Class for signal decomposition using Wavelet Transform. \
:class:`WaveletDecomposer` decomposes a signal using the Wavelet Transform, \
:class:`WindowWaveletDecomposer` extends this functionality to handle signals with a sliding window approach.
"""

import logging

import numpy as np
import pywt

from .base import BaseDecomposer
from .window import BaseWindowDecomposer

logger = logging.getLogger(__name__)


class WaveletDecomposer(BaseDecomposer):
    """Class for signal decomposition using Wavelet Transform."""

    default_wavelet: str = "haar"  #: Default wavelet for decomposition
    default_level: int = 8  #: Default level for decomposition
    default_mode: str = "per"  #: Default mode for wavelet reconstruction

    @staticmethod
    def _check_level(level: int, N: int, wavelet: str) -> int:
        """Check and validate the wavelet decomposition level.

        Parameters
        ----------
        level : int
            The level of decomposition to check.
        N : int
            The length of the input signal.
        wavelet : str
            The wavelet to use for decomposition.

        Returns
        -------
        int
            The validated level of decomposition.
            The maximum level if the provided level is greater than the maximum level for the given signal length and wavelet.

        Raises
        ------
        ValueError
            If the level is not a positive integer.
        """
        # Check if level is a positive integer
        if not isinstance(level, int) or level <= 0:
            raise ValueError(f"Level must be a positive integer. Got: {level}")
        level_max = pywt.dwt_max_level(N, pywt.Wavelet(wavelet).dec_len)  # type: ignore
        # Check if level is greater than the maximum level for the given signal length and wavelet
        if level > level_max:
            logger.warning("level (%d) is greater than the maximum level (%d). Using level = %d.", level, level_max, level_max)
            return level_max
        return level

    @staticmethod
    def full_decompose(
        signal: np.ndarray, *args, wavelet: str = default_wavelet, level: int = default_level, mode: str = default_mode, kwargs
    ) -> np.ndarray:
        """Returns the full decomposition (all coefficients) of the signal using wavelet decomposition.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        wavelet : str, optional
            The wavelet to use for decomposition, by default :attr:`default_wavelet`.
            See `pywt.wavelist(kind='discrete')` for available wavelets.
        level : int, optional
            The level of decomposition, by default :attr:`default_level`.
        mode : str, optional
            The mode to use for wavelet reconstruction, by default :attr:`default_mode`.
            See `pywt.Modes.modes` for available modes.
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        np.ndarray (?,)
            The array of wavelet coefficients. The shape depends on the wavelet and level used.
        """
        N = len(signal)
        level = WaveletDecomposer._check_level(level, N, wavelet)
        WT = pywt.wavedec(signal, wavelet, mode=mode, level=level)
        arr, _ = pywt.coeffs_to_array(WT)
        return arr

    @staticmethod
    def decompose(
        signal: np.ndarray, n_coefs: int, *args, wavelet: str = default_wavelet, level: int = default_level, mode: str = default_mode, kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Decompose the signal into n_coefs wavelet coefficients.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of wavelet coefficients to return.
        wavelet : str, optional
            The wavelet to use for decomposition, by default :attr:`default_wavelet`.
            See `pywt.wavelist(kind='discrete')` for available wavelets.
        level : int, optional
            The level of decomposition, by default :attr:`default_level`.
        mode : str, optional
            The mode to use for wavelet reconstruction, by default :attr:`default_mode`.
            See `pywt.Modes.modes` for available modes.
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        tuple[np.ndarray (n_coefs,), np.ndarray (n_coefs,)]
            The indices and coefficients of the wavelet decomposition.
        """
        arr = WaveletDecomposer.full_decompose(signal, *args, wavelet=wavelet, level=level, mode=mode, **kwargs)
        ind = np.argsort(np.abs(arr))[::-1][:n_coefs]
        coefs = arr[ind]
        return ind, coefs

    @staticmethod
    def reconstruct(
        N: int,
        indices: np.ndarray,
        coefficients: np.ndarray,
        *args,
        wavelet: str = default_wavelet,
        level: int = default_level,
        mode: str = default_mode,
        **kwargs,
    ) -> np.ndarray:
        """Reconstruct the signal from the specified wavelet coefficients at the given indices.

        Parameters
        ----------
        N : int
            The length of the original signal.
        indices : np.ndarray (n_coefs,)
            The indices of the wavelet coefficients to use for reconstruction.
        coefficients : np.ndarray (n_coefs,)
            The coefficients to use for reconstruction.
        wavelet : str, optional
            The wavelet to use for reconstruction, by default :attr:`default_wavelet`
        level : int, optional
            The level of decomposition to use for reconstruction, by default :attr:`default_level`
        mode : str, optional
            The mode to use for wavelet reconstruction, by default :attr:`default_mode`
        args
            Unused positional arguments.
        kwargs
            Unused keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.
        """
        # Check the number of coefficients and indices and the level validity
        BaseDecomposer._check_number_of_coefficients(indices, coefficients)
        level = WaveletDecomposer._check_level(level, N, wavelet)

        # Create a dummy signal to get the shape of the coefficients array
        dummy_signal = np.zeros(N)
        dummy_coeffs = pywt.wavedec(dummy_signal, wavelet=wavelet, mode=mode, level=level)
        dummy_arr, dummy_slices = pywt.coeffs_to_array(dummy_coeffs)

        # Create the array of coefficients with the same shape as the dummy array and fill it with the provided coefficients at the specified indices
        coeff_arr = np.zeros(dummy_arr.shape, dtype="complex")
        coeff_arr[indices] = coefficients

        # Convert the array of coefficients back to the wavelet coefficients format and reconstruct the signal
        coeffs_from_arr = pywt.array_to_coeffs(coeff_arr, dummy_slices, output_format="wavedec")
        S_reconstruct = np.real(pywt.waverec(coeffs_from_arr, wavelet=wavelet, mode=mode))

        # Ensure the reconstructed signal has the same length as the original signal
        all_S_reconstruct = np.zeros(N)
        all_S_reconstruct[: len(S_reconstruct)] = S_reconstruct
        return all_S_reconstruct


class WindowWaveletDecomposer(BaseWindowDecomposer):
    """Class for signal decomposition using Wavelet Transform with a sliding window approach."""

    decomposer_cls = WaveletDecomposer

    @staticmethod
    def _check_kwargs(kwargs: dict) -> dict:
        """Check and return the kwargs for the decomposer.
        Check if the level is valid given the window size and wavelet (avoid repeating the warning in :class:`WaveletDecomposer` for each window)

        Parameters
        ----------
        kwargs : dict
            The keyword arguments to check and return.

        Returns
        -------
        dict
            The checked and potentially modified keyword arguments.
        """
        # Get the parameters from kwargs or set defaults
        level = kwargs.get("level", WaveletDecomposer.default_level)
        wavelet = kwargs.get("wavelet", WaveletDecomposer.default_wavelet)
        L = kwargs.get("L", BaseWindowDecomposer.default_L)
        # Check the level and wavelet validity
        if level is not None and wavelet is not None and L is not None:
            kwargs["level"] = WaveletDecomposer._check_level(level, L, wavelet)
        return kwargs
