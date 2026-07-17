"""Contains all functions needed for matrix operations.
The Backend to use (either :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>` or
:class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>`).
"""

from .numpy_backend import NumpyBackend
from .torch_backend import TorchBackend

__all__ = ["NumpyBackend", "TorchBackend"]
