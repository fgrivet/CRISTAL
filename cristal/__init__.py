"""
**CRISTAL**: ChRISToffel Anomaly Locator

A Python package for anomaly detection using the Christoffel function.
"""

import logging
import os

from cristal.__version__ import __author__, __date__, __version__
from cristal.christoffel import BaggingDyCF, DyCF, DyCG
from cristal.helper_classes import (
    IMPLEMENTED_INCREMENTERS_OPTIONS,
    IMPLEMENTED_INVERSION_OPTIONS,
    IMPLEMENTED_POLYNOMIAL_BASIS,
    IMPLEMENTED_REGULARIZATION_OPTIONS,
    PolynomialsBasisGenerator,
)
from cristal.plotter import DyCFPlotter

__all__ = [
    "BaggingDyCF",
    "DyCF",
    "DyCG",
    "DyCFPlotter",
    "IMPLEMENTED_INCREMENTERS_OPTIONS",
    "IMPLEMENTED_INVERSION_OPTIONS",
    "IMPLEMENTED_POLYNOMIAL_BASIS",
    "IMPLEMENTED_REGULARIZATION_OPTIONS",
    "PolynomialsBasisGenerator",
]

logger = logging.getLogger("CRISTAL")

try:
    login = os.getlogin()
except Exception:
    login = "unknown"


if login == __author__:
    # Set the logger to DEBUG level for development
    level = logging.DEBUG
else:
    # Set the logger to INFO level for production
    level = logging.INFO

console = logging.StreamHandler()
console.setLevel(level)
formatter = logging.Formatter("%(asctime)s \t %(levelname)s \t %(name)s.%(module)s.%(funcName)s \t %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(level)

logger.debug("CRISTAL version %s (date: %s) loaded successfully.", __version__, __date__)
