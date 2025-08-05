"""
Generate distributions for testing CRISTAL methods.
"""

import logging

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from cristal.utils.type_checking import check_types, positive_integer

logger = logging.getLogger("CRISTAL")


@check_types({"n_samples": positive_integer})
def make_T_rotated(n_samples: int = 1000, scaler: TransformerMixin | None = MinMaxScaler((-1, 1))) -> np.ndarray:
    """Generate a 2D "T" rotated distribution.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1000
    scaler : TransformerMixin | None, optional
        The scaler to use for transforming the data, by default MinMaxScaler((-1, 1))

    Returns
    -------
    np.ndarray
        The generated 2D "T" rotated distribution.
    """
    np.random.seed(42)

    if (N := 2 * (n_samples // 2)) != n_samples:  # Ensure N is even for two normal distributions
        logger.warning("n_samples must be even to create two normal distributions. Adjusting n_samples = %d to %d.", n_samples, N)

    # Generate the "T" rotated data
    norm_1 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 50]], size=N // 2)
    norm_1 = np.dot(norm_1, np.array([[np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), np.cos(np.pi / 4)]])) + np.array([[20, 20]])
    norm_2 = np.random.multivariate_normal(mean=[0, 0], cov=[[1, 0], [0, 50]], size=N // 2)
    norm_2 = np.dot(norm_2, np.array([[-np.cos(np.pi / 4), -np.sin(np.pi / 4)], [np.sin(np.pi / 4), -np.cos(np.pi / 4)]]))
    data = np.concatenate([norm_1, norm_2])

    # Scale the data for better performance of DyCF
    if scaler is not None:
        data = scaler.fit_transform(data)

    return data
