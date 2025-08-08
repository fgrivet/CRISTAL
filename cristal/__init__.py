"""
**CRISTAL**: ChRISToffel Anomaly Locator

A Python package for anomaly detection using the Christoffel function.
"""

import logging
import os

from . import decomposers, detectors, evaluation, incrementers, inverters, moments_matrix, plotters, polynomials, regularizers, type_checking
from .__version__ import __author__, __date__, __version__
from .decomposers import IMPLEMENTED_DECOMPOSERS
from .detectors import BaggingDyCF, DyCF, DyCG
from .incrementers import IMPLEMENTED_INCREMENTERS
from .inverters import IMPLEMENTED_INVERTERS
from .moments_matrix import MomentsMatrix
from .plotters import DyCFPlotter
from .polynomials import IMPLEMENTED_POLYNOMIALS, MultivariatePolynomialBasis
from .regularizers import IMPLEMENTED_REGULARIZERS

__all__ = [
    "BaggingDyCF",
    "DyCF",
    "DyCFPlotter",
    "DyCG",
    "IMPLEMENTED_DECOMPOSERS",
    "IMPLEMENTED_INCREMENTERS",
    "IMPLEMENTED_INVERTERS",
    "IMPLEMENTED_POLYNOMIALS",
    "IMPLEMENTED_REGULARIZERS",
    "MomentsMatrix",
    "MultivariatePolynomialBasis",
    "decomposers",
    "detectors",
    "evaluation",
    "incrementers",
    "inverters",
    "moments_matrix",
    "plotters",
    "polynomials",
    "regularizers",
    "type_checking",
]

logger = logging.getLogger("cristal")

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
formatter = logging.Formatter("%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)
logger.setLevel(level)

logger.debug("CRISTAL version %s (date: %s) loaded successfully.", __version__, __date__)
