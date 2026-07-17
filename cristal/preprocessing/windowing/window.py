"""Windowing for time series preprocessing."""

from typing import Generic, cast

from ...backend.base_backend import Backend
from ...commons.base_commons import BaseCommons
from ...types import ArrayLike, DTypeLike
from ..base_preprocessor import BasePreprocessor


# pylint: disable=unused-variable
class Windowizer(BasePreprocessor, BaseCommons, Generic[ArrayLike, DTypeLike]):
    """Transforms time series into overlapping windows.

    .. version-added:: 0.0.3

    Parameters
    ----------
    window_size : int
        The size of each window.
    shift : int, optional
        The stride between windows, by default 1.

    Raises
    ------
    ValueError
        If window_size or shift is not positive.

    Attributes
    ----------
    window_size : int
        The size of each window.
    shift : int
        The stride between windows.
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.

    """

    requires = ["backend"]

    def __init__(self, window_size: int, shift: int = 1):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        window_size : int
            The size of each window.
        shift : int, optional
            The stride between windows, by default 1

        Raises
        ------
        ValueError
            If window_size or shift is not positive.
        """
        if window_size <= 0:
            raise ValueError(f"window_size must be positive. Got {window_size}.")
        if shift <= 0:
            raise ValueError(f"shift must be positive. Got {shift}.")

        self.window_size = window_size
        """The size of each window."""
        self.shift = shift
        """The stride between windows."""

        # Attributes bound in the configuration __init__
        self.backend: Backend[ArrayLike, DTypeLike]
        """The backend to use for the computation."""

    def fit(self, X: ArrayLike, y: ArrayLike | None = None):
        """Do nothing.

        Parameters
        ----------
        X : ArrayLike
            The input data of shape (N_points, [n_channels]).
        y : ArrayLike | None, optional
            Ignored.

        Returns
        -------
        BasePreprocessor
            The fitted Windowizer

        Raises
        ------
        ValueError
            If the backend is not bound.

            If the input data has more than 2 dimensions.
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the Windowizer class before using it.")

        if X.ndim != 1 and X.ndim != 2:
            raise ValueError("Windowizer is built for 1D (N_points,) or 2D (N_points, N_channels) arrays.")

        return self

    def transform(self, X: ArrayLike) -> ArrayLike:
        """Create overlaping windows from data.

        Parameters
        ----------
        X : ArrayLike
            Input of shape (n_samples,) or (n_samples, n_channels).

        Returns
        -------
        ArrayLike
            Windows of shape (n_windows, window_size, [n_channels]).
            With n_windows = :math:`\\lfloor \\frac{n\\_samples-window\\_size}{shift} \\rfloor + 1`.

        Raises
        ------
        ValueError
            If the backend is not bound.

            If the input data has more than 2 dimensions
        """
        if self.backend is None:
            raise ValueError("A backend must be bound to the Windowizer class before using it.")

        if X.ndim not in (1, 2):
            raise ValueError("Windowizer is built for 1D (N_points,) or 2D (N_points, N_channels) arrays.")

        windows = self.backend.make_windows(X, self.window_size, self.shift)
        # Ensure shape is (n_windows, window_size, n_channels)
        if X.ndim > 1:
            windows = self.backend.swap(windows, 1, 2)
        return windows

    def inverse_transform(self, X: ArrayLike, original_length: int | None = None) -> ArrayLike:
        """Undo the windowing by averaging the values of all windows that a point is appart of.

        Parameters
        ----------
        X : ArrayLike
            The windowed data of shape (n_windows,) or (n_windows, window_size, [n_channels])
        original_length : int | None, optional
            The length of the original signal to reconstruct the right length.
            If None, the maximum number of points with this number of windows: :math:`shift \\times N\\_windows + window\\_size - 1`, by default None

        Returns
        -------
        ArrayLike
            The reconstructed signal with shape (original_length,) for 1D inputs or (original_length, extra_dims) for multi-dimensional inputs

        Raises
        ------
        ValueError
            If the window size (second dimension of :attr:`X`) is not equal to :attr:`window_size`.
        """
        N_windows: int = X.shape[0]
        dtype = cast(DTypeLike, X.dtype)

        # Check if input is 1D (only window averages)
        input_is_1d = X.ndim == 1

        # Compute the minimum and maximum number of points that can give this number of windows given this shift and window_size
        min_length = self.shift * (N_windows - 1) + self.window_size
        max_length = self.shift * N_windows + self.window_size - 1
        # If original_length is unknown, fix it to the maximum number of point possible
        if original_length is None:
            original_length = max_length
        elif original_length < min_length or original_length > max_length:
            raise ValueError(
                f"Given the number of windows ({N_windows}), their sizes ({self.window_size}), and the shift ({self.shift}),"
                f"the original length is between {min_length} and {max_length} points. Received: {original_length}."
            )

        # For 2D+ inputs, verify window size matches
        if not input_is_1d and X.shape[1] != self.window_size:
            raise ValueError(f"The window size ({X.shape[1]}) must equal self.window_size ({self.window_size}).")

        # Compute window starts
        window_starts = self.backend.arange(0, N_windows * self.shift, self.shift, dtype=int)

        # Create all point indices
        points = self.backend.arange(original_length, dtype=int)

        # Create mask: in_window[w, i] = True if point i is in window w
        in_window = (points >= window_starts[:, None]) & (points < window_starts[:, None] + self.window_size)

        # Get indices of valid (window, point) pairs
        w_indices, i_indices = self.backend.where(in_window)

        # Gather the window values
        if input_is_1d:
            gathered = cast(ArrayLike, X[w_indices])  # shape (M,)
            extra_dims = ()  # No extra dimensions for 1D input
        else:
            # Compute offsets within each window
            offsets = points[i_indices] - window_starts[w_indices]
            gathered = cast(ArrayLike, X[w_indices, offsets])  # shape (M,) or (M, N_channels)
            extra_dims = X.shape[2:]  # (N_channels,) for 2D inputs

        # Initialize output
        values_sum = self.backend.zeros((original_length,) + extra_dims, dtype=dtype)
        values_count = self.backend.zeros(original_length, dtype=dtype)

        # Accumulate sums and counts
        self.backend.add_at(values_sum, i_indices, gathered)
        self.backend.add_at(values_count, i_indices, self.backend.ones(i_indices.shape, dtype=dtype))

        # Handle division: ensure count is broadcastable to sum_scores
        if extra_dims:
            count_expanded = values_count[(...,) + (None,) * len(extra_dims)]
        else:
            count_expanded = values_count

        # Handle division by zero (points not covered by any window)
        point_values = self.backend.divide(
            values_sum, count_expanded, out=self.backend.zeros(values_sum.shape, dtype=dtype), where=count_expanded != 0
        )

        return point_values
