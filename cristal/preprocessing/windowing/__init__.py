"""Windowing preprocessing for CRISTAL detectors.

This module provides windowing utilities for transforming time series data
into sliding windows for anomaly detection.
"""

from .window import Windowizer

__all__ = ["Windowizer"]
