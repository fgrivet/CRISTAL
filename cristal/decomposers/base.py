"""
Base class for signal decomposition methods.
"""

import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class BaseDecomposer(ABC):
    """Base class for signal decomposition methods."""

    @staticmethod
    def _check_number_of_coefficients(indices: np.ndarray, coefficients: np.ndarray) -> None:
        """Check if the number of coefficients matches the number of indices.

        Parameters
        ----------
        indices : np.ndarray (n_coefs,)
            The indices corresponding to the coefficients.
        coefficients : np.ndarray (n_coefs,)
            The coefficients to check.

        Raises
        ------
        ValueError
            If the number of coefficients does not match the number of indices.
        """
        if coefficients.shape[0] != len(indices):
            raise ValueError(f"The number of coefficients must match the number of indices ({len(indices)}). Got: {coefficients.shape[0]}.")

    @staticmethod
    @abstractmethod
    def full_decompose(signal: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return the full decomposition (all coefficients) of the signal.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        np.ndarray (?,)
            The full decomposition of the signal, where ? is the number of coefficients and depends on the decomposition method.
        """

    @staticmethod
    @abstractmethod
    def decompose(signal: np.ndarray, n_coefs: int, *args, **kwargs) -> tuple[np.ndarray, np.ndarray]:
        """Decompose the signal into n_coefs components.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int
            The number of coefficients to retain.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        tuple[np.ndarray (n_coefs,), np.ndarray (n_coefs,)]
            A tuple containing the indices and coefficients of the decomposition.
        """

    @staticmethod
    @abstractmethod
    def reconstruct(N: int, indices: np.ndarray, coefficients: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Reconstruct the signal from the indices and coefficients.

        Parameters
        ----------
        N : int
            The length of the original signal.
        indices : np.ndarray (n_coefs,)
            The indices of the coefficients to use for reconstruction.
        coefficients : np.ndarray (n_coefs,)
            The coefficients to use for reconstruction.
        args
            See child classes for specific positional arguments.
        kwargs
            See child classes for specific keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.
        """

    @classmethod
    def transform(cls, signal: np.ndarray, *args, n_coefs: int | None = None, indices: np.ndarray | None = None, **kwargs) -> np.ndarray:
        """Reconstruct the signal from its decomposition into n_coefs components.

        .. hint::

            Pass either :attr:`n_coefs` or :attr:`indices`.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        n_coefs : int | None, by default None
            The number of coefficients to retain.
        indices : np.ndarray | None, by default None
            The indices of the coefficients to use for reconstruction.
        args
            See :meth:`decompose` and :meth:`reconstruct` for specific positional arguments.
        kwargs
            See :meth:`decompose` and :meth:`reconstruct` for specific keyword arguments.

        Returns
        -------
        np.ndarray (N,)
            The reconstructed signal.

        Raises
        ------
        ValueError
            If neither :attr:`n_coefs` nor :attr:`indices` is provided.
        """
        # Check either n_coefs or indices is passed
        if indices is None and n_coefs is None:
            raise ValueError("Either n_coefs or indices must be provided.")
        # If indices is not provided, decompose the signal
        if indices is None:
            indices, coefficients = cls.decompose(signal, n_coefs, *args, **kwargs)  # type: ignore
        else:
            if n_coefs is not None and len(indices) != n_coefs:
                logger.warning("Indices length (%d) does not match the number of coefficients (%d). Continuing with indices.", len(indices), n_coefs)
            coefficients = cls.get_coefficients_at_indices(signal, indices, *args, **kwargs)
        return cls.reconstruct(len(signal), indices, coefficients, *args, **kwargs)

    @classmethod
    def get_coefficients_at_indices(cls, signal: np.ndarray, indices: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Get the coefficients of the signal at the specified indices.

        Parameters
        ----------
        signal : np.ndarray (N,)
            The input signal to decompose.
        indices : np.ndarray (n_coefs,)
            The indices at which to retrieve the coefficients.
        args
            See :meth:`full_decompose` for specific positional arguments.
        kwargs
            See :meth:`full_decompose` for specific keyword arguments.

        Returns
        -------
        np.ndarray (n_coefs,)
            The coefficients of the signal at the specified indices.
        """
        coefficients = cls.full_decompose(signal, *args, **kwargs)
        return coefficients[indices]

    @classmethod
    def plot(
        cls,
        signal: np.ndarray,
        indices: np.ndarray,
        *args,
        x: np.ndarray | None = None,
        show: bool = True,
        save: bool = False,
        save_title: str = "decomposer_plot.png",
        close: bool = True,
        fig: Figure | None = None,
        ax: Axes | None = None,
        **kwargs,
    ) -> None | tuple[Figure, Axes]:
        """Plot the original and reconstructed signals.

        Parameters
        ----------
        signal : np.ndarray
            The original signal.
        indices : np.ndarray
            The indices of the coefficients used for reconstruction.
        x : np.ndarray | None, optional
            The x-coordinates for the plot, by default None
        show : bool, optional
            Whether to show the plot, by default True
        save : bool, optional
            Whether to save the plot, by default False
        save_title : str, optional
            The title for the saved plot, by default "decomposer_plot.png"
        close : bool, optional
            Whether to close the plot after saving or showing, by default True
        fig : Figure | None, optional
            The figure to plot on, by default None
        ax : Axes | None, optional
            The axes to plot on, by default None

        Returns
        -------
        None | tuple[Figure, Axes]
            The figure and axes objects if not closing the plot, otherwise None.
        """
        if x is not None and len(x) != len(signal):
            logger.warning("The length of x (%d) does not match the length of the signal (%d). Ignoring x.", len(x), len(signal))
            x = None
        if x is None:
            x = np.arange(len(signal))

        if fig is None and ax is not None:
            logger.warning("ax is provided but fig is None, creating a new figure.")
        elif fig is not None and ax is None:
            logger.warning("fig is provided but ax is None, creating a new axes.")
        if ax is None or fig is None:
            fig, ax = plt.subplots()

        reconstruction = cls.transform(signal, *args, indices=indices, **kwargs)
        ax.plot(x, signal, label="Signal", alpha=0.8)
        ax.plot(x, reconstruction, label="Reconstructed Signal")
        ax.legend()

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
            to_return = None
        else:
            to_return = (fig, ax)
        return to_return
