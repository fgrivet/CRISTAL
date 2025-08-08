"""
Base class for outlier detection algorithms plotters.
"""

from abc import ABC, abstractmethod

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class BasePlotter(ABC):
    """
    Base class for outlier detection results plotters. This requires that data is 2-dimensional.
    """

    @abstractmethod
    def levelset(
        self,
        x: np.ndarray,
        *,
        n_x1: int,
        n_x2: int,
        levels: list | None,
        percentiles: list | None,
        show: bool,
        save: bool,
        save_title: str,
        close: bool,
    ) -> None | tuple[Figure, Axes]:
        """Generates the levelset plot and optionally show, save and close.

        Parameters
        ----------
        x : np.ndarray
            The input data to plot, shape (n_samples, n_features=2).
        n_x1 : int
            Number of points along the first dimension for the grid.
        n_x2 : int
            Number of points along the second dimension for the grid.
        levels : list[int | float] | None
            The levels to plot.
        percentiles : list[int | float] | None
            The percentiles to compute and plot as levels.
        show : bool
            Whether to show the plot.
        save : bool
            Whether to save the plot to a file.
        save_title : str
            Title for the saved plot file.
        close : bool
            Whether to close the plot or return it after saving or showing.

        Returns
        -------
        None | tuple[Figure, Axes]
            The figure and axes objects if not closing the plot, otherwise None.
        """

    @abstractmethod
    def boundary(self, x, *, n_x1, n_x2, show, save, save_title, close) -> None | tuple[Figure, Axes]:
        """Generates the boundary plot and optionally show, save and close.

        Parameters
        ----------
        x : np.ndarray
            The input data to plot, shape (n_samples, n_features=2).
        n_x1 : int
            Number of points along the first dimension for the grid.
        n_x2 : int
            Number of points along the second dimension for the grid.
        show : bool
            Whether to show the plot.
        save : bool
            Whether to save the plot to a file.
        save_title : str
            Title for the saved plot file.
        close : bool
            Whether to close the plot or return it after saving or showing.

        Returns
        -------
        None | tuple[Figure, Axes]
            The figure and axes objects if not closing the plot, otherwise None.
        """
