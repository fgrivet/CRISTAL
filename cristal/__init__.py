"""
CRISTAL: ChRISToffel Anomaly Locator
A Python package for anomaly detection using the Christoffel function.
"""

from cristal.plotter import DyCFPlotter
from cristal.christoffel import DyCF, DyCG
from cristal.__version__ import __version__, __date__

__all__ = [
    "DyCF",
    "DyCG",
    "DyCFPlotter",
]

print(f"CRISTAL version {__version__} (date: {__date__}) loaded successfully.")
