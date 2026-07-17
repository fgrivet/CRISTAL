"""Contains the class :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>`
which defines all the functions available in detectors
and its extension :class:`BaseCGDetector <cristal.core.detectors.base_detector.BaseCGDetector>`
which detects anomalies based on the growth of scores as the degree :attr:`n` increases."""

import logging
import textwrap
from abc import ABC, abstractmethod
from typing import Generic, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from ...config.detector_config import ConfigType
from ...types import ArrayLike, DTypeLike

logger = logging.getLogger(__name__)


class BaseDetector(ABC, Generic[ArrayLike, DTypeLike, ConfigType]):
    """Base class for the detectors.
    Define all the functions and attributes available in the detectors.

    Parameters
    ----------
    n : int | Literal["auto"]
        The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`.
    config : DynamicDetectorConfig[ArrayLike, DTypeLike]
        The configuration of the model.

    Attributes
    ----------
    n : int | :const:`auto`
        The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`.
    config : ConfigType
        The configuration of the model.
    N : int | None
        The number of training data in the moment matrix, set during :func:`fit`.
    d : int | None
        The dimension of training data, set during :func:`fit`.
    intrinsic_dim : int | None
        The intrinsic dimension of the data (the one in the CF),
        :const:`1` for :class:`UCF <cristal.core.detectors.univariate.UCF>` and :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`,
        and :const:`d` for :class:`DyCF <cristal.core.detectors.dynamic.DyCF>` and :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`.
    threshold : int | None
        The threshold for a score to be considered as an anomaly, set during :func:`fit`.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.
    """

    def __init__(self, n: int | Literal["auto"], config: ConfigType):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        n : int | Literal["auto"]
            The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`.
        config : DynamicDetectorConfig[ArrayLike, DTypeLike]
            The configuration of the model.
        """

        self.n: int | Literal["auto"] = n
        """The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`."""
        self.config: ConfigType = config
        """The configuration of the model."""

        # Variables defined during fitting
        self.N: int | None = None
        """The number of training data in the moment matrix, set during :func:`fit`."""
        self.d: int | None = None
        """The dimension of training data, set during :func:`fit`."""
        self.intrinsic_dim: int | None = None
        """The intrinsic dimension of the data (the one in the CF),
        :const:`1` for :class:`UCF <cristal.core.detectors.univariate.UCF>` and :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`,
        and :attr:`d` for :class:`DyCF <cristal.core.detectors.dynamic.DyCF>` and :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`."""
        self.threshold: float | None = None
        """The threshold for a score to be considered as an anomaly, set during :func:`fit`."""

    # =====================================
    #          private methods
    # =====================================

    def _compute_n(self, n: int | Literal["auto"], N: int) -> int:
        """Compute the degree n of the CF if :attr:`n` is :const:`auto`.

        Parameters
        ----------
        n : int | :const:`auto`
            The current degree of the CF.
        N : int
            The number of training data.

        Returns
        -------
        int
            The computed degree of the CF: :math:`N^{1 / (2+intrinsic\\_dim)}` if :attr:`n` is :const:`auto`, else :attr:`n`.

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        # n = N**(1/(2+d))
        if self.intrinsic_dim is None:
            raise ValueError("Model must be fitted before computing n.")

        if n == "auto":
            n = int(N ** (1 / (2 + self.intrinsic_dim)))
            logger.info("auto n =%s", n)
        if not isinstance(n, int):
            raise ValueError("Error during the computation of n.")
        return n

    @abstractmethod
    def _compute_scores(self, component_support: ArrayLike, component_X: ArrayLike) -> ArrayLike:
        """Compute the scores of each sample with the given components.

        Parameters
        ----------
        component_support : ArrayLike
            The component involving the support.
        component_X : ArrayLike
            The component involving the points to be scored.

        Returns
        -------
        ArrayLike
            The scores for each sample in :attr:`component_X`.

        .. tip::

            See also the description of this method for the detector used to have a more specific description with signification and shapes.
        """

    @abstractmethod
    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute for each sample the components needed to calculate the scores.

        Parameters
        ----------
        X : ArrayLike
            The data to analyze of shape (N_samples_test, d).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The component involving the support, The component involving the points to be scored.

        .. tip::

            See also the description of this method for the detector used to have a more specific description with signification and shapes.
        """

    @abstractmethod
    def _crop_components(self, component_support: ArrayLike, component_X: ArrayLike, n_crop: int) -> tuple[ArrayLike, ArrayLike]:
        """Crop the computed components to calculate the scores for a degree :math:`n_crop \\leq n`.

        Parameters
        ----------
        component_support : ArrayLike
            The component involving the support.
        component_X : ArrayLike
            The component involving the points to be scored.
        n_crop : int
            The maximum polynomial basis degree. Must be lower or equal to :attr:`n`.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The cropped component involving the support, The cropped component involving the points to be scored.

        .. tip::

            See also the description of this method for the detector used to have a more specific description with signification and shapes.
        """

    def _preprocess_test_data(self, X: ArrayLike) -> ArrayLike:
        """Transform the data to the ArrayLike type, then preprocess it.

        Parameters
        ----------
        X : ArrayLike
            The data points of shape (N_samples_test, d).

        Returns
        -------
        ArrayLike
            The preprocess data points. The shape depends on the  preprocessing step.

        Raises
        ------
        ValueError
            If X is not a 2D ArrayLike.

            If the model is not fitted.

            If the degree :attr:`n` is not positive.

            If the dimension of the testing data `d` is not the same as the dimension of the training data :attr:`d`.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike of shape (N_samples_test, d={self.d}). Got {X.shape}.")
        if not self.is_fitted():
            raise ValueError("Model must be fitted before scoring samples.")
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        d = X.shape[1]
        if d != self.d:
            raise ValueError(f"Dimension mismatch between training ({self.d}) and test ({d}).")

        # Preprocess the data
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.transform(X)  # type: ignore

        return X

    # =====================================
    #           public methods
    # =====================================

    @abstractmethod
    def fit(self, X: ArrayLike) -> "BaseDetector":
        """Fit the detector to the data :attr:`X`.

        Parameters
        ----------
        X : ArrayLike
            The training data of shape (N_samples_train, d).

        Returns
        -------
        BaseDetector
            The detector fitted.
        """

    @abstractmethod
    def is_fitted(self) -> bool:
        """Check if the detector is fitted.

        Returns
        -------
        bool
            :const:`True` if the detector is fitted, :const:`False` otherwise.
        """

    def score_samples(self, X: ArrayLike, online: Literal["none", "constant", "increment"] = "none", quantile=0.95) -> ArrayLike:
        """Compute the anomaly scores for each sample in :attr:`X`.
        First transform the data to the ArrayLike type, then preprocess it, and finally compute the scores with a CF of degree :attr:`n`.
        If :attr:`online` is enabled, update the support after each batch using the points with the :attr:`quantile` lowest scores.

        .. version-changed::0.0.2
            Added :attr:`online` and :attr:`quantile` to do online learning

        Parameters
        ----------
        X : ArrayLike
            The points on which to calculate the scores of shape (N_samples_test, d)
        online : :const:`none` | :const:`constant` | :const:`increment`, optional
            The online method to use, by default :const:`none`.

            `none` : No update.

            `constant` : Replace the oldest support points by the new ones based on the chosen `quantile`.

            `increment` : Add the new support points to the existing set.
        quantile : float, optional
            The quantile to use to determine if a point is added to the support, by default 0.95.

        Returns
        -------
        ArrayLike
            The scores of each sample of shape (N_samples_test,).

        Raises
        ------
        ValueError
            If X is not a 2D ArrayLike.

            If the model is not fitted.

            If the degree :attr:`n` is not positive.

            If the dimension of the testing data `d` is not the same as the dimension of the training data :attr:`d`.
        """
        scores = self.config.backend.empty(len(X))
        start_idx = 0
        for X_batch in self.config.storage(X):
            preprocessed_X_batch = self._preprocess_test_data(X_batch)

            component_support, component_X = self._compute_components(preprocessed_X_batch)
            component_support, component_X = self._crop_components(component_support, component_X, self.n)  # pyright: ignore[reportArgumentType]

            current_scores = self._compute_scores(component_support, component_X)
            end_idx = start_idx + len(current_scores)
            scores[start_idx:end_idx] = current_scores
            start_idx = end_idx

            if online != "none":
                idx_to_include = self.config.backend.where(current_scores < self.config.backend.quantile(current_scores, quantile))[0]
                self.update(preprocessed_X_batch[idx_to_include], online)

        return scores

    @abstractmethod
    def update(self, X: ArrayLike, online: Literal["constant", "increment"]) -> "BaseDetector":
        """Update the support of the detector.

        Parameters
        ----------
        X : ArrayLike
            The points to include in the support of shape (N_new_support_points, d).
        online : :const:`constant` | :const:`increment`
            The online method to use.

            `constant` : Replace the oldest support points by the new ones.

            `increment` : Add the new support points to the existing set.

        .. caution::

            `online` is not used with :class:`DyCF <cristal.core.detectors.dynamic.DyCF>`.

        Returns
        -------
        BaseDetector
            The detector with its support updated.
        """

    def decision_function(self, X: ArrayLike, online: Literal["none", "constant", "increment"] = "none", quantile=0.95) -> ArrayLike:
        """Compute the anomaly scores for each sample in :attr:`X`.
        First transform the data to the ArrayLike type, then preprocess it, and finally compute the scores with a CF of degree :attr:`n`.
        If :attr:`online` is enabled, update the support after each batch using the points with the :attr:`quantile` lowest scores.

        .. hint::

            This function is a wrapper for :func:`score_samples` for compatibility with PyOD and sklearn.

        .. version-added::0.0.3

        Parameters
        ----------
        X : ArrayLike
            The points on which to calculate the scores of shape (N_samples_test, d)
        online : :const:`none` | :const:`constant` | :const:`increment`, optional
            The online method to use, by default :const:`none`.

            `none` : No update.

            `constant` : Replace the oldest support points by the new ones based on the chosen `quantile`.

            `increment` : Add the new support points to the existing set.
        quantile : float, optional
            The quantile to use to determine if a point is added to the support, by default 0.95.

        Returns
        -------
        ArrayLike
            The scores of each sample of shape (N_samples_test,).
        """
        return self.score_samples(X, online=online, quantile=quantile)

    def predict_from_scores(self, scores: ArrayLike) -> ArrayLike:
        """Compute the prediction from the scores for each sample.
        -1 for outliers, 1 for inliers.
        If the score for a sample is greater than :attr:`threshold`, then the sample is considered as an anomaly.

        Parameters
        ----------
        scores : ArrayLike
            The scores computed for the sample of shape (N_samples_test,).

        Returns
        -------
        ArrayLike
            The prediction for each sample of shape (N_samples_test,).

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if self.threshold is None:
            raise ValueError("Model must be fitted before predicting scores.")
        return self.config.backend.where(scores > self.threshold, -1, 1)

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Compute the prediction for each sample.


        Parameters
        ----------
        X : ArrayLike
            The input data of shape (N_samples_test, d).

        Returns
        -------
        ArrayLike
            The prediction for each sample of shape (N_samples_test,).
            -1 for outliers, 1 for inliers.
            If the score for a sample is greater than :attr:`threshold`, then the sample is considered as an anomaly.
        """
        return self.predict_from_scores(self.score_samples(X))

    # =====================================
    #              plot methods
    # =====================================

    def plot_levelset(
        self,
        X: ArrayLike | np.ndarray,
        *,
        n_x1: int = 100,
        n_x2: int = 100,
        levels: list | None = None,
        percentiles: list | None = None,
        reference_level: bool = True,
        show: bool = True,
        save: bool = False,
        save_title: str = "CF_levelset.png",
        close: bool = True,
        fig: Figure | None = None,
        ax: Axes | None = None,
        cbar_fmt: str = ".5g",
    ) -> None | tuple[Figure, Axes]:  # pragma: no cover
        """
        Plot the level sets of the model's decision function.
        The reference level 1 is plotted in red if :attr:`reference_level` is :const:`True`.

        Parameters
        ----------
        X: ArrayLike | np.ndarray
            The data points to plot, shape (N_samples, n_features=2).
        n_x1: int
            Number of points along the first dimension for the grid.
        n_x2: int
            Number of points along the second dimension for the grid.
        levels: list[int | float], optional
            Specific levels to plot. If None, defaults to [].
        percentiles: list[int | float], optional
            Percentiles (between 0 and 100) to compute and plot as levels. If None, defaults to [].
        reference_level: bool, optional
            Wheter to show the level line equal to 1, defaults to True.
        show: bool
            Whether to show the plot. Defaults to True.
        save: bool
            Whether to save the plot to a file. Defaults to False.
        save_title: str
            Title for the saved plot file. Defaults to "CF_levelset.png".
        close: bool
            Whether to close the plot or return it after saving or showing. Defaults to True.
        fig: Figure | None
            The figure to plot on. If None, a new figure is created.
        ax: Axes | None
            The axes to plot on. If None, a new axes is created.
        cbar_fmt: str
            Format for the colorbar ticks. Defaults to ".5g".
        """
        if self.d != 2:
            raise ValueError("The model must be fitted for 2 dimensions data.")
        if X.shape[1] != self.d:
            raise ValueError("The input data must have the same number of dimensions as the model.")
        if (levels is None or levels == []) and (percentiles is None or percentiles == []):
            raise ValueError("At least one of levels or percentiles must be provided.")

        if fig is None and ax is not None:
            logger.warning("ax is provided but fig is None, creating a new figure and new axes.")
        elif fig is not None and ax is None:
            logger.warning("fig is provided but ax is None, creating a new figure and new axes.")
        if ax is None or fig is None:
            fig, ax = plt.subplots()
        assert ax is not None

        X_np = X if isinstance(X, np.ndarray) else self.config.backend.to_numpy(X)

        # Scatter the data points
        ax.scatter(X_np[:, 0], X_np[:, 1], marker="x", s=20)

        # Make a grid and compute the function values
        x1_margin = (np.max(X_np[:, 0]) - np.min(X_np[:, 0])) / 5
        x2_margin = (np.max(X_np[:, 1]) - np.min(X_np[:, 1])) / 5
        ax.set_xlim((np.min(X_np[:, 0]) - x1_margin, np.max(X_np[:, 0]) + x1_margin))
        ax.set_ylim((np.min(X_np[:, 1]) - x2_margin, np.max(X_np[:, 1]) + x2_margin))
        x1_values = np.linspace(np.min(X_np[:, 0] - x1_margin), np.max(X_np[:, 0] + x1_margin), n_x1)
        x2_values = np.linspace(np.min(X_np[:, 1]) - x2_margin, np.max(X_np[:, 1] + x2_margin), n_x2)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        grid_values = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        scores = self.config.backend.to_numpy(self.score_samples(grid_values)).reshape(x1_grid.shape) / self.threshold

        # Set the percentiles and levels
        if percentiles is None:
            percentiles = []
        if levels is None:
            levels = []
        levels += [np.percentile(scores, p) for p in percentiles]
        levels = sorted(set(levels))

        # Plot level sets
        cs = ax.contour(x1_grid, x2_grid, scores, levels=levels, colors=mpl.colormaps["tab20"].colors)  # pyright: ignore[reportAttributeAccessIssue]
        ax.clabel(cs, inline=1, fmt=f"%{cbar_fmt}", fontsize=8)
        cbar = plt.colorbar(cs)
        cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:{cbar_fmt}}"))

        # Plot the reference level set (1 thanks to the regularization by the threshold)
        if reference_level:
            cs_ref = ax.contour(x1_grid, x2_grid, scores, levels=[1], colors=["r"])
            ax.clabel(cs_ref, inline=1)

        title = (
            f"Level sets of {self.__class__.__name__} with degree={self.n} and "
            f"threshold scheme={self.config.threshold_scheme.scheme}={self.threshold}"
        )
        ax.set_title("\n".join(textwrap.wrap(title, width=40)))

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

    def plot_boundary(
        self,
        X: ArrayLike | np.ndarray,
        *,
        n_x1: int = 100,
        n_x2: int = 100,
        show: bool = True,
        save: bool = False,
        save_title: str = "CF_boundary.png",
        close: bool = True,
        fig: Figure | None = None,
        ax: Axes | None = None,
    ) -> None | tuple[Figure, Axes]:
        """
        Plot the boundary decision of the model.
        In green, the points where the decision function is positive (considered as inliers),
        in red, the points where the decision function is negative (considered as outliers).

        Parameters
        ----------
        X: ArrayLike | np.ndarray
            The data points to plot, shape (N_samples, n_features=2).
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
        fig: Figure | None
            The figure to plot on. If None, a new figure is created.
        ax: Axes | None
            The axes to plot on. If None, a new axes is created.
        """
        if self.d != 2:
            raise ValueError("The model must be fitted for 2 dimensions data.")
        if X.shape[1] != self.d:
            raise ValueError("The input data must have the same number of dimensions as the model.")
        if fig is None and ax is not None:
            logger.warning("ax is provided but fig is None, creating a new figure.")
        elif fig is not None and ax is None:
            logger.warning("fig is provided but ax is None, creating a new axes.")
        if ax is None or fig is None:
            fig, ax = plt.subplots()

        X_np = X if isinstance(X, np.ndarray) else self.config.backend.to_numpy(X)

        # Make a grid and predict the values
        x1_margin = (np.max(X_np[:, 0]) - np.min(X_np[:, 0])) / 5
        x2_margin = (np.max(X_np[:, 1]) - np.min(X_np[:, 1])) / 5
        ax.set_xlim((np.min(X_np[:, 0]) - x1_margin, np.max(X_np[:, 0]) + x1_margin))
        ax.set_ylim((np.min(X_np[:, 1]) - x2_margin, np.max(X_np[:, 1]) + x2_margin))
        x1_values = np.linspace(np.min(X_np[:, 0] - x1_margin), np.max(X_np[:, 0] + x1_margin), n_x1)
        x2_values = np.linspace(np.min(X_np[:, 1]) - x2_margin, np.max(X_np[:, 1] + x2_margin), n_x2)
        x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)
        grid_values = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        predictions = self.config.backend.to_numpy(self.predict(grid_values)).reshape(x1_grid.shape)

        colors = ["red", "black", "green"]
        norm = Normalize(vmin=-1, vmax=1)  # Set the midpoint at zero
        cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)
        ax.contourf(x1_grid, x2_grid, predictions, cmap=cmap, norm=norm, alpha=0.7)

        # Scatter the data points
        ax.scatter(X_np[:, 0], X_np[:, 1], marker="x", s=20, alpha=0.3, color="blue")

        ax.set_title(
            f"Boundary decision of {self.__class__.__name__} with degree={self.n} "
            f"and threshold scheme={self.config.threshold_scheme.scheme}={self.threshold}"
        )

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


# pylint: disable=unused-variable
class BaseCGDetector(BaseDetector[ArrayLike, DTypeLike, ConfigType]):
    """An extension of the :class:`BaseDetector <cristal.core.detectors.base_detector.BaseDetector>` class
    to detect anomalies based on the growth of the socres as the degree :attr:`n` increases.

    Parameters
    ----------
    detector_class: type[BaseDetector[ArrayLike, DTypeLike, ConfigType]]
        The detector to use.
    n: list[int]
        The list of degrees on which to evaluate the model.
    config: ConfigType
        The configuration of the model.

    Attributes
    ----------
    n_list : list[int]
        The list of degrees on which to evaluate the model.
    n : int
        The degree of the CF. Here the maximimum degree :attr:`n_max`.
    n_min : int
        The minimum degree on which to evaluate the model.
    n_max : int
        The maximum degree on which to evaluate the model.
    n_val : int
        The number of degrees on which to evaluate the model.
    config : ConfigType
        The configuration of the model.
    N : int | None
        The number of training data, set during :func:`fit`.
    d : int | None
        The dimension of training data, set during :func:`fit`.
    intrinsic_dim : int | None
        The intrinsic dimension of the data (the one in the CF),
        :const:`1` for :class:`UCF <cristal.core.detectors.univariate.UCF>` and :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`,
        and :const:`d` for :class:`DyCF <cristal.core.detectors.dynamic.DyCF>` and :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`.
    threshold : int | None
        The threshold for a score to be considered as an anomaly, set during :func:`fit`.
    detector: BaseDetector[ArrayLike, DTypeLike, ConfigType]
        The detector to use.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.

    See Also
    --------
    BaseDetector : For methods / attributes not redefined here.
    """

    def __init__(self, detector_class: type[BaseDetector[ArrayLike, DTypeLike, ConfigType]], n: list[int], config: ConfigType, *args, **kwargs):
        if len(n) == 0:
            raise ValueError("n must not be empty.")

        # Sort unique values of n
        n = sorted(set(n))
        # Store n_list, n_min, n_max, n_val to avoid recomputing it over and over again
        self.n_list: list[int] = n
        """The list of degrees on which to evaluate the model"""
        self.n_min: int = n[0]
        """The minimum degree on which to evaluate the model"""
        self.n_max: int = n[-1]
        """The maximum degree on which to evaluate the model"""
        self.n_val: int = len(n)
        """The number of degrees on which to evaluate the model"""

        super().__init__(self.n_min, config)
        self.detector: BaseDetector = detector_class(self.n_max, config, *args, **kwargs)
        """The detector to use."""

    # =====================================
    #              CG methods
    # =====================================
    def _polynomial_regression(self, Y: ArrayLike, degree: int, x: ArrayLike | None = None, normalize: bool = False) -> ArrayLike:
        """Compute the polynomial regression: Y = f(x) where f is a polynomial of degree :attr:`degree`

        Parameters
        ----------
        Y : ArrayLike
            The values of the polynomial regression, i.e. the scores for each sample and each degree.
            2D ArrayLike of shape (N_samples_test, k). Usually k = :attr:`n_val`.
        degree : int
            The degree of the polynomial in the regression.
        x : ArrayLike | None, optional
            The x values for the regression. If :const:`None`, a range from 0 to k-1, by default :const:`None`.
        normalize : bool, optional
            If :const:`True`, standardize the :attr:`x` values, by default :const:`False`.

        Returns
        -------
        ArrayLike
            The estimated values by the polynomial regression of shape (N_samples_test, k).

        Raises
        ------
        ValueError
            If :attr:`Y` is not a 2D ArrayLike or contains only 1 score per sample.
        """
        if Y.ndim != 2:
            raise ValueError(f"Y must be a 2D ArrayLike of shape (N_samples_test, k). Usually k={self.n_val}. Got {Y.shape}.")

        # Get the number of scores in the Y values
        _, k = Y.shape

        if k == 1:
            raise ValueError("Can't do regression with only 1 point.")
        if k != self.n_val:
            logger.warning("The number of scores in Y (%d) does not match the number of values in the detector (%d).", k, self.n_val)

        # If x values are not provided, create a default sequence of x values from 0 to k-1
        if x is None:
            x = self.detector.config.backend.arange(0, k, dtype=self.config.backend.default_dtype)

        # If normalize is True, normalize the x values to be centered and scaled
        if normalize:
            x = (cast(ArrayLike, x) - self.config.backend.mean(x)) / self.config.backend.std(x)

        # Create the A matrix such that Y = A @ B where B are the coefficients and A is [1, x, x^2, ..., x^degree]
        A = self.config.backend.vander(x, degree, increasing=True)

        # Least sqaures
        # Solve: A @ B = Y.T to obtain the coefficients B
        B = self.config.backend.lstsq(A, Y.T)

        # Compute the predicted values X_hat using the computed coefficients B
        X_hat = (A @ B).T

        return X_hat

    def _compute_R2(self, X: ArrayLike, X_hat: ArrayLike) -> ArrayLike:
        """Compute the R2 score from the data points X and the esimated points via regression X_hat.

        Parameters
        ----------
        X : ArrayLike
            The real points. Usually the scores of shape (N_samples_test, :attr:`n_val`).
        X_hat : ArrayLike
            The estimated points via regression. Usually the estimated scores of shape (N_samples_test, :attr:`n_val`).

        Returns
        -------
        ArrayLike
            The R2 scores for each sample of shape (N_samples_test).
        """
        # Compute the R2 score from the data points x and x_hat
        mean = self.config.backend.mean(X, axis=1, keepdims=True)

        ss_tot = self.config.backend.norm2D(X - mean)
        ss_res = self.config.backend.norm2D(X - X_hat)

        R2 = 1.0 - ss_res / ss_tot
        return R2

    def _compute_regressions(self, scores: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute both polynomial regression of degree :attr:`intrinsic_dim` on the scores
        and linear regression on the log of the scores and return both R2 scores.

        Parameters
        ----------
        scores : ArrayLike
            The scores of shape (N_samples_test, :attr:`n_val`).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The R2 scores for the polynomial regression on the scores of shape (N_samples_test,),
            The R2 scores for the linear regression on the log of the scores of shape (N_samples_test,).

        Raises
        ------
        ValueError
            If the model is not fitted.

            If the scores dim is not valid.
        """
        # Compute 2 regressions on the scores and return their R2
        if self.intrinsic_dim is None:
            raise ValueError("Detector must be fitted before calling _compute_regressions.")
        if scores.ndim != 2:
            raise ValueError(f"scores must be a 2D ArrayLike of shape (N_samples_test, k={self.n_val}). Got {scores.shape}.")

        scores = self.config.backend.copy(scores)

        # Get the current number of scores to get the current degree of UCG (not necessarily the max degree)
        _, n_scores = scores.shape
        current_n = n_scores + self.n_min - 1

        # Setup the regression parameters
        x = self.config.backend.arange(self.n_min, current_n + 1, dtype=self.config.backend.default_dtype)  # x values = polynomial degrees

        # Check where scores < 0 (because log = NaN)
        idx_neg = self.config.backend.where(scores <= 0)
        scores[idx_neg] = 1  # Dummy value, R2 will be changed afterwards
        log_scores = self.config.backend.log(scores)  # The log of the scores

        # Compute R2 of polynomial regression with specific degree. If well fitted then it's nominal data
        R2_poly_reg = self._compute_R2(scores, self._polynomial_regression(scores, degree=self.intrinsic_dim, x=x))

        # Compute linear regression on log scores. If well fitted then it's anomalous data
        R2_linear_reg = self._compute_R2(log_scores, self._polynomial_regression(log_scores, degree=1, x=x))

        # Negative scores ==> Anomaly ==> 0% nominal and 100% anomaly
        R2_poly_reg[idx_neg[0]] = 0
        R2_linear_reg[idx_neg[0]] = 1

        return R2_poly_reg, R2_linear_reg

    # =====================================
    #          private methods
    # =====================================

    # pylint: disable=protected-access
    def _compute_scores(self, component_support: ArrayLike, component_X: ArrayLike) -> ArrayLike:
        return self.detector._compute_scores(component_support, component_X)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return self.detector._compute_components(X)

    def _crop_components(self, component_support: ArrayLike, component_X: ArrayLike, n_crop: int) -> tuple[ArrayLike, ArrayLike]:
        return self.detector._crop_components(component_support, component_X, n_crop)

    # =====================================
    #           public methods
    # =====================================

    def fit(self, X: ArrayLike) -> "BaseCGDetector":
        # Save the number of sample, and dimensionality of training data
        self.N, self.d = X.shape
        self.detector.fit(X)
        # Add the detector attributes to the CGDetector for consistency
        self.intrinsic_dim = self.detector.intrinsic_dim
        self.n = self.detector.n
        self.threshold = self.detector.threshold
        return self

    def is_fitted(self) -> bool:
        return self.detector.is_fitted()

    def all_score_samples(self, X: ArrayLike, online: Literal["none", "constant", "increment"] = "none", quantile=0.95) -> ArrayLike:
        """Compute the anomaly scores for each sample in :attr:`X` and for each degree in :attr:`n_list`.
        First transform the data to the ArrayLike type, then preprocess it, and finally compute the scores with a CF of each degree in :attr:`n_list`.

        Parameters
        ----------
        X : ArrayLike
            The points on which to calculate the scores of shape (N_samples_test, d).
        online : :const:`none` | :const:`constant` | :const:`increment`, optional
            The online method to use, by default :const:`none`.

            `none` : No update.

            `constant` : Replace the oldest support points by the new ones based on the chosen `quantile`.

            `increment` : Add the new support points to the existing set.
        quantile : float, optional
            The quantile to use to determine if a point is added to the support, by default 0.95.

        Returns
        -------
        ArrayLike
            The scores of each sample and each degree of shape (N_samples_test, :attr:`n_val`).

        Raises
        ------
        ValueError
            If X is not a 2D ArrayLike.

            If the model is not fitted.

            If the degree :attr:`n` is not positive.

            If the dimension of the testing data `d` is not the same as the dimension of the training data :attr:`d`.
        """
        # Create the array to store all the scores
        all_scores = self.detector.config.backend.empty((len(X), self.n_val))

        # Compute the scores by batch
        start_idx = 0
        for X_batch in self.config.storage(X):
            preprocessed_X_batch = self._preprocess_test_data(X_batch)

            # Compute the full components
            component_support, component_X = self._compute_components(preprocessed_X_batch)
            end_idx = start_idx + len(component_X)

            # For each degree to compute, crop component_support and component_X and compute the corresponding scores
            for i, n in enumerate(self.n_list):
                component_support_crop, component_X_crop = self._crop_components(component_support, component_X, n)
                current_scores = self._compute_scores(component_support_crop, component_X_crop)
                all_scores[start_idx:end_idx, i] = current_scores

            # Update the support based on the mean scores on all degrees
            if online != "none":
                batch_scores = all_scores[start_idx:end_idx]
                mean_batch_scores = self.config.backend.mean(batch_scores, axis=1)
                idx_to_include = self.config.backend.where(mean_batch_scores < self.config.backend.quantile(mean_batch_scores, quantile))[0]
                self.update(preprocessed_X_batch[idx_to_include], online)

            start_idx = end_idx

        return all_scores

    def score_samples(
        self,
        X: ArrayLike,
        online: Literal["none", "constant", "increment"] = "none",
        quantile=0.95,
        method: Literal["mean", "clip", "linear"] = "mean",
    ) -> ArrayLike:
        """Compute the anomaly scores for each sample in :attr:`X`.
        First transform the data to the ArrayLike type, then preprocess it, compute the scores with a CF of each degree in :attr:`n_list`,
        compute both polynomial regression on the scores and linear regression on their log,
        and finally compute one score per sample based on the given :attr:`method`.

        Parameters
        ----------
        X : ArrayLike
            The points on which to calculate the scores of shape (N_samples_test, d)
        online : :const:`none` | :const:`constant` | :const:`increment`, optional
            The online method to use, by default :const:`none`.

            `none` : No update.

            `constant` : Replace the oldest support points by the new ones based on the chosen `quantile`.

            `increment` : Add the new support points to the existing set.
        quantile : float, optional
            The quantile to use to determine if a point is added to the support, by default 0.95.
        method : :const:`mean` | :const:`clip` | :const:`linear`, optional
            The method to use to compute the final score from the scores of each degree, by default :const:`mean`.

            `mean` : :math:`\\frac{1}{n\\_{val}} \\sum_{i=1}^{n\\_{val}} \\frac{s_{i+1} - s_{i}}{n\\_{list}_{i+1} - n\\_{list}_{i}}`.

            `clip` : :math:`min(0, R2_{lin} - R2_{poly})`. Greater than 0 means that it is an outlier.

            `linear` : :math:`(1 + R2_{lin} - R2_{poly}) / 2`. Greater than 0.5 means that it is an outlier.


        Returns
        -------
        ArrayLike
            The scores of each point between 0 (inlier) and 1 (outlier) of shape (N_samples_test,).
        """
        return self.unique_score(self.all_score_samples(X, online=online, quantile=quantile), method=method)

    def predict_from_scores(self, scores):
        # In case we got scores from all_score_samples
        if scores.ndim == 2:
            scores = self.unique_score(scores)
        return self.config.backend.where(scores > 0.5, -1, 1)

    def unique_score(self, scores: ArrayLike, method: Literal["mean", "clip", "linear"] = "mean") -> ArrayLike:
        """Based on the scores for all degree in :attr:`n_list`, aggregate them to return only one score per sample.

        Parameters
        ----------
        scores : ArrayLike
            The scores for each degree in :attr:`n_list` of shape (N_samples_test, :attr:`n_val`).
        method : :const:`mean` | :const:`clip` | :const:`linear`, optional
            The method to use to compute the final score from the scores of each degree, by default :const:`mean`.

            `mean` : :math:`\\frac{1}{n\\_{val}} \\sum_{i=1}^{n\\_{val}} \\frac{s_{i+1} - s_{i}}{n\\_{list}_{i+1} - n\\_{list}_{i}}`.

            `clip` : :math:`min(0, R2_{lin} - R2_{poly})`. Greater than 0 means that it is an outlier.

            `linear` : :math:`(1 + R2_{lin} - R2_{poly}) / 2`. Greater than 0.5 means that it is an outlier.

        Returns
        -------
        ArrayLike
            The scores of each point between 0 (inlier) and 1 (outlier) of shape (N_samples_test,).
        """
        if method == "mean":
            n_list = self.config.backend.to_array_like(self.n_list)
            return self.config.backend.mean((scores[:, 1:] - scores[:, :-1]) / (n_list[1:] - n_list[:-1]), axis=1)

        # Compute regressions
        R2_poly_reg, R2_linear_reg = self._compute_regressions(scores)

        if method == "clip":
            # 0 if nominal, linear from 0 to 1 if anomaly
            return self.config.backend.clip(R2_linear_reg - R2_poly_reg, min_=0)

        # Linear from 0 to 1
        return (R2_linear_reg - R2_poly_reg + 1) / 2

    def update(self, X: ArrayLike, online: Literal["constant", "increment"]) -> "BaseCGDetector":
        self.detector.update(X, online=online)
        return self

    def __getattr__(self, name):
        return getattr(self.detector, name)
