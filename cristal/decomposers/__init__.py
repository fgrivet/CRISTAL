"""
CRISTAL Decomposers Subpackage \
Contains various decomposer classes for time series analysis, including Fourier and Wavelet decompositions.
"""

from enum import Enum

from .base import BaseDecomposer
from .fourier import FourierDecomposer, WindowFourierDecomposer
from .wavelet import WaveletDecomposer, WindowWaveletDecomposer
from .window import BaseWindowDecomposer


class IMPLEMENTED_DECOMPOSERS(Enum):
    """
    The implemented decomposers classes.
    """

    FOURIER = FourierDecomposer
    WAVELET = WaveletDecomposer
    WINDOW_FOURIER = WindowFourierDecomposer
    WINDOW_WAVELET = WindowWaveletDecomposer


__all__ = [
    "BaseDecomposer",
    "BaseWindowDecomposer",
    "FourierDecomposer",
    "IMPLEMENTED_DECOMPOSERS",
    "WaveletDecomposer",
    "WindowFourierDecomposer",
    "WindowWaveletDecomposer",
]
