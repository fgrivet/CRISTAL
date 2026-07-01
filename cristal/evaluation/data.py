"""
Contains functions to generate distributions for testing CRISTAL detectors.
"""

import logging

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def make_T_rotated(n_samples: int = 1000, scaler: TransformerMixin | None = MinMaxScaler((-1, 1))) -> np.ndarray:  # pragma: no cover
    """Generate a 2D "T" rotated distribution.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate. If :const:`n_samples` is odd, takes :const:`n_samples - 1`,  by default 1000.
    scaler : TransformerMixin | None, optional
        The scaler to use for transforming the data, by default MinMaxScaler((-1, 1)).

    Returns
    -------
    np.ndarray
        The generated 2D "T" rotated distribution of shape (2*(n_samples//2), 2).

    Examples
    --------

    .. code-block:: python

        import matplotlib.pyplot as plt
        from cristal.evaluation.data import make_T_rotated
        data = make_T_rotated(n_samples=1000)
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
        plt.show()

    .. image:: /_static/make_T_rotated.png
        :width: 60%
        :alt: T rotated distribution
        :align: center

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

    # Scale the data
    if scaler is not None:
        data = scaler.fit_transform(data)

    return data


def make_uniform_square(n_samples: int = 1000, scaler: TransformerMixin | None = MinMaxScaler((-1, 1))) -> np.ndarray:  # pragma: no cover
    """Generate a 2D square from uniform distribution.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate, by default 1000.
    scaler : TransformerMixin | None, optional
        The scaler to use for transforming the data, by default MinMaxScaler((-1, 1)).

    Returns
    -------
    np.ndarray
        The generated square distribution of shape (n_samples, 2).

    Examples
    --------

    .. code-block:: python

        import matplotlib.pyplot as plt
        from cristal.evaluation.data import make_uniform_square
        data = make_uniform_square(n_samples=1000)
        plt.figure()
        plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
        plt.show()

    .. image:: /_static/make_uniform_square.png
        :width: 60%
        :alt: T rotated distribution
        :align: center
    """
    np.random.seed(42)
    data = np.random.random((n_samples, 2))

    # Scale the data
    if scaler is not None:
        data = scaler.fit_transform(data)

    return data
