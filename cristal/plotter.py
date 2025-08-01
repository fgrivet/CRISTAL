"""
Implementation of the CFPlotter class for plotting level sets and boundaries of DyCF models.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure

from cristal.helper_classes import BasePlotter
from cristal.christoffel import DyCF
from cristal.utils.type_checking import check_all_int_or_float, check_none, check_types, positive_integer


class DyCFPlotter(BasePlotter):
    """
    Plotter for levelsets and boundaries associated with a DyCF model.

    Attributes
    ----------
    model: DyCF
        The model to study, which should have a dimension d of 2.
        This model should be fitted before plotting.

    Methods
    -------
    See BasePlotter methods.

    """

    @check_types()
    def __init__(self, model: DyCF):
        """Initialize the CFPlotter.

        Parameters
        ----------
        model : DyCF | DyCG
            The model to study, which should have a dimension d of 2.
        """
        assert model.is_fitted(), "Model must be fitted before plotting."
        assert model.d == 2, "Model must be 2-dimensional for plotting."
        self.model = model

    @check_types(
        {
            "n_x1": positive_integer,
            "n_x2": positive_integer,
            "levels": lambda x: check_none(x) or check_all_int_or_float(x),
            "percentiles": lambda x: check_none(x) or check_all_int_or_float(x),
        }
    )
    def levelset(
        self,
        x: np.ndarray,
        *,
        n_x1: int = 100,
        n_x2: int = 100,
        levels: list | None = None,
        percentiles: list | None = None,
        show: bool = True,
        save: bool = False,
        save_title: str = "CF_levelset.png",
        close: bool = True,
    ) -> None | tuple[Figure, Axes]:
        """
        Plot the level sets of the model's decision function.
        The reference level 1 is plotted in red.

        Parameters
        ----------
        x: np.ndarray
            The data points to plot, shape (n_samples, n_features=2).
        n_x1: int
            Number of points along the first dimension for the grid.
        n_x2: int
            Number of points along the second dimension for the grid.
        levels: list[int | float], optional
            Specific levels to plot. If None, defaults to [].
        percentiles: list[int | float], optional
            Percentiles to compute and plot as levels. If None, defaults to [].
        show: bool
            Whether to show the plot. Defaults to True.
        save: bool
            Whether to save the plot to a file. Defaults to False.
        save_title: str
            Title for the saved plot file. Defaults to "CF_levelset.png".
        close: bool
            Whether to close the plot or return it after saving or showing. Defaults to True.
        """
        assert x.shape[1] == self.model.d
        # Make figure
        fig, ax = plt.subplots()

        # Scatter the data points
        ax.scatter(x[:, 0], x[:, 1], marker="x", s=20)

        # Make a grid and compute the function values
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 5
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 5
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        x1_values = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
        x2_values = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        grid_values = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        scores = self.model.score_samples(grid_values).reshape(x1_grid.shape)

        # Set default percentiles and levels
        if percentiles is None:
            percentiles = []
        if levels is None:
            levels = []
        levels += [np.percentile(scores, p) for p in percentiles]
        levels = sorted(set(levels))
        # Plot level sets
        cs = ax.contour(x1_values, x2_values, scores, levels=levels)
        ax.clabel(cs, inline=1)

        # Plot the reference level set (1 thanks to the regularization)
        cs_ref = ax.contour(x1_values, x2_values, scores, levels=[1], colors=["r"])
        ax.clabel(cs_ref, inline=1)

        ax.set_title(
            f"Level sets of {self.model.__class__.__name__} with degree={self.model.d} "
            f"and regularization={self.model.regularization} ({self.model.regularizer})"
        )

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
            return None
        else:
            return fig, ax

    @check_types({"n_x1": positive_integer, "n_x2": positive_integer})
    def boundary(
        self,
        x: np.ndarray,
        *,
        n_x1: int = 100,
        n_x2: int = 100,
        show: bool = True,
        save: bool = False,
        save_title: str = "CF_boundary.png",
        close: bool = True,
    ) -> None | tuple[Figure, Axes]:
        """
        Plot the boundary decision of the model.
        In green, the points where the decision function is positive (considered as inliers),
        in red, the points where the decision function is negative (considered as outliers).

        Parameters
        ----------
        x: np.ndarray
            The data points to plot, shape (n_samples, n_features=2).
        n_x1: int
            Number of points along the first dimension for the grid.
        n_x2: int
            Number of points along the second dimension for the grid.
        show: bool
            Whether to show the plot. Defaults to True.
        save: bool
            Whether to save the plot to a file. Defaults to False.
        save_title: str
            Title for the saved plot file. Defaults to "CF_boundary.png".
        close: bool
            Whether to close the plot or return it after saving or showing. Defaults to True.
        """
        assert x.shape[1] == self.model.d
        # Make figure
        fig, ax = plt.subplots()

        # Make a grid and predict the values
        x1_margin = (np.max(x[:, 0]) - np.min(x[:, 0])) / 5
        x2_margin = (np.max(x[:, 1]) - np.min(x[:, 1])) / 5
        ax.set_xlim([np.min(x[:, 0]) - x1_margin, np.max(x[:, 0]) + x1_margin])
        ax.set_ylim([np.min(x[:, 1]) - x2_margin, np.max(x[:, 1]) + x2_margin])
        x1_values = np.linspace(np.min(x[:, 0] - x1_margin), np.max(x[:, 0] + x1_margin), n_x1)
        x2_values = np.linspace(np.min(x[:, 1]) - x2_margin, np.max(x[:, 1] + x2_margin), n_x2)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        grid_values = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        predictions = self.model.predict(grid_values).reshape(x1_grid.shape)

        colors = ["red", "black", "green"]
        norm = Normalize(vmin=-1, vmax=1)  # Set the midpoint at zero
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        ax.contourf(x1_values, x2_values, predictions, cmap=cmap, norm=norm, alpha=0.7)

        # Scatter the data points
        ax.scatter(x[:, 0], x[:, 1], marker="x", s=20, alpha=0.3, color="blue")

        ax.set_title(
            f"Boundary decision of {self.model.__class__.__name__} with degree={self.model.d} "
            f"and regularization={self.model.regularization} ({self.model.regularizer})"
        )

        if save:
            plt.savefig(save_title)
        if show:
            plt.show()
        if close:
            plt.close()
        else:
            return fig, ax
