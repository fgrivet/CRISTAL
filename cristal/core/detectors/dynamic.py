"""Contains the original Christoffel function based outlier detection algorithm, adapted from :cite:t:`ducharlet2025leveraging`.
This code is inspired from `kyducharlet/odds - GitHub <https://github.com/kyducharlet/odds>`_.
"""

from math import comb
from typing import Literal, cast

from ...config.detector_config import DynamicDetectorConfig
from ...types import ArrayLike, DTypeLike
from .base_detector import BaseCGDetector, BaseDetector


class DyCF(BaseDetector[ArrayLike, DTypeLike, DynamicDetectorConfig]):
    """Class to compute the original Christoffel function based outlier detection scores and predictions, adapted from :cite:t:`ducharlet2025leveraging`.

    Attributes
    ----------
    matrix_size_limit : int | None
        The maximum size of the moment matrix that can be calculated without raising an error in order to avoid OOM or timeout errors. If :const:`None`, try to calculate regardless of the size of the matrix.
    M : ArrayLike | None
        The moment matrix of size :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ n \\end{pmatrix}` calculated during :func:`fit`. :const:`None` before fitting.
    M_inv : ArrayLike | None
        The inverse of the moment matrix of size :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ n \\end{pmatrix}` calculated during :func:`fit`. :const:`None` before fitting.

    See Also
    --------
    cristal.core.detectors.base_detector.BaseDetector : For more attributes.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.
    """

    def __init__(
        self,
        n: int | Literal["auto"] = "auto",
        config: DynamicDetectorConfig[ArrayLike, DTypeLike] = DynamicDetectorConfig(),
        matrix_size_limit: int | None = None,
    ):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        n : int | Literal["auto"], optional
            The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during fitting, by default :const:`auto`.
        config : DynamicDetectorConfig[ArrayLike, DTypeLike], optional
            The configuration of the model, by default DynamicDetectorConfig().
        matrix_size_limit : int | None, optional
            The maximum size of the moment matrix that can be calculated without raising an error in order to avoid OOM or timeout errors. If :const:`None`, try to calculate regardless of the size of the matrix, by default None.
        """
        self.matrix_size_limit = matrix_size_limit
        """The maximum size of the moment matrix that can be calculated without raising an error in order to avoid OOM or timeout errors. If :const:`None`, try to calculate regardless of the size of the matrix."""
        super().__init__(n, config)

        # Variables defined during fitting specific to DyCF
        self.M: ArrayLike | None = None
        """The moment matrix of size :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ n \\end{pmatrix}` calculated during :func:`fit`. :const:`None` before fitting."""
        self.M_inv: ArrayLike | None = None
        """The inverse of the moment matrix of size :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ n \\end{pmatrix}` calculated during :func:`fit`. :const:`None` before fitting."""

    def _compute_scores(self, component_support: ArrayLike, component_X: ArrayLike) -> ArrayLike:
        """Compute the scores for each sample with the formula :math:`v(x) M^{-1} v(x)^T`.

        Parameters
        ----------
        component_support : ArrayLike
            The inverse moment matrix :math:`M^{-1}` of shape (s_d(n), s_d(n)).
        component_X : ArrayLike
            The polynomial basis of each sample :math:`v(x)` of shape (N_samples_test, s_d(n)).

        Returns
        -------
        ArrayLike
            The scores for each sample of shape (N_samples_test,)
        """
        return self.config.backend.einsum("ni,ij,nj->n", component_X, component_support, component_X)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute for each sample the components needed to calculate the scores.

        Parameters
        ----------
        X : ArrayLike
            The matrix containing the samples of shape (N_samples_test, d).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The inverse moment matrix :math:`M^{-1}` of shape (s_d(n), s_d(n)), The polynomial basis of each sample :math:`v(x)` of shape (N_samples_test, s_d(n)).

        Raises
        ------
        ValueError
            If n is not a positive integer.

            If the model is not fitted.
        """
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.M_inv is None:
            raise ValueError("M_inv must be computed.")

        # Compute the polynomial basis v(x) for each point in X
        V = self.config.polynomial_basis.vandermonde_nd(X, self.n)
        # M_inv is already computed here
        return self.M_inv, V

    def _crop_components(self, component_support: ArrayLike, component_X: ArrayLike, n_crop: int) -> tuple[ArrayLike, ArrayLike]:
        """Crop the computed components to calculate the scores for a degree :math:`n_crop \\leq n`.

        Parameters
        ----------
        component_support : ArrayLike
            The inverse moment matrix :math:`M^{-1}` of shape (s_d(n), s_d(n)).
        component_X : ArrayLike
            The polynomial basis of each sample :math:`v(x)` of shape (N_samples_test, s_d(n)).
        n_crop : int
            The maximum polynomial basis degree. Must be lower or equal to :attr:`n`.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            The cropped inverse moment matrix :math:`M^{-1}` of shape (s_d(n_crop), s_d(n_crop)), The cropped polynomial basis of each sample :math:`v(x)` of shape (N_samples_test, s_d(n_crop)).

        Raises
        ------
        ValueError
            If n_crop is not a positive integer, lower or equal to :attr:`n`.

            If the model is not fitted.
        """
        if not isinstance(self.n, int) or self.n <= 0:
            raise ValueError(f"n must be a positive integer. Got {self.n}.")
        if self.d is None or self.d <= 0:
            raise ValueError(f"d must be a positive integer. Got {self.d}.")
        if n_crop > self.n:
            raise ValueError(f"n ({n_crop}) must be lower or equal than self.n ({self.n}).")
        if self.M is None:
            raise ValueError("M must be fitted.")

        new_size = comb(self.d + n_crop, n_crop)

        if component_support.shape[0] == new_size:
            return component_support, component_X

        # Crop and then inverse again
        M_inv = self.config.inverter(self.M[:new_size, :new_size])
        return M_inv, component_X[:, :new_size]

    def fit(self, X: ArrayLike) -> BaseDetector:
        """Fit the model to the data :attr:`X`.
        Compute the moment matrix :attr:`M`, its inverse :attr:`M_inv`, and the :attr:`threshold` used for prediction.

        Parameters
        ----------
        X : ArrayLike
            The training data of shape (N_samples_train, d).

        Returns
        -------
        BaseDetector
            The fitted model.

        Raises
        ------
        ValueError
            If :attr:`X` is not a 2D ArrayLike.

            If :math:`s_d(n) = \\begin{pmatrix} n+d \\\\ n \\end{pmatrix} > matrix\\_size\\_limit`.
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike of shape (N_samples_train, d). Got {X.shape}.")

        # Define The number of training data and the diension of training data
        N, d = X.shape

        # Save the information on training data
        self.N = cast(int, N)
        self.d = cast(int, d)
        self.intrinsic_dim = self.d

        # Define the degree if set to auto using n = N**(1/(2+d))
        self.n = self._compute_n(self.n, self.N)

        # Compute the moment matrix
        matrix_size = comb(self.d + self.n, self.n)
        if self.matrix_size_limit is not None and matrix_size > self.matrix_size_limit:
            raise ValueError(f"Cannot fit the model. Matrix size ({matrix_size}) is greater than the fixed limit ({self.matrix_size_limit}).")

        M = self.config.backend.zeros((matrix_size, matrix_size))
        for X_batch in self.config.storage(X):
            # Preprocess the data
            if self.config.preprocessing is not None:
                X_batch = self.config.preprocessing.fit_transform(X_batch)  # type: ignore
            if not self.config.backend.is_array_like(X_batch):
                X_batch = self.config.backend.to_array_like(X_batch)
            # Compute the moment matrix
            V = self.config.polynomial_basis.vandermonde_nd(X_batch, self.n)
            M += V.T @ V
        M = (M + M.T) / (2 * N)  # Ensure symmetry of M and normalize by N
        self.M = cast(ArrayLike, M)

        # Compute the inverse of the moment matrix
        self.M_inv = self.config.inverter(self.M)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return (
            self.N is not None
            and self.d is not None
            and self.intrinsic_dim is not None
            and self.M is not None
            and self.M_inv is not None
            and self.threshold is not None
            and isinstance(self.n, int)
        )

    # TODO : implement these functions
    def update(self, X: ArrayLike) -> "DyCF": ...
    def save_model(self): ...
    def load_model(self): ...


class DyCG(BaseCGDetector):
    def __init__(self, n_list: list[int], config: DynamicDetectorConfig = DynamicDetectorConfig(), *args, **kwargs):
        """Class to compute the original Christoffel function based outlier detection scores, and predictions based on the growth of scores as the degree :attr:`n` increases, adapted from :cite:t:`ducharlet2025leveraging`.
        The scoring function is modified from :cite:p:`ducharlet2025leveraging` to be the same as all :class:`BaseCGDetector <cristal.core.detectors.base_detector.BaseCGDetector>` classes.

        See Also
        --------
        DyCF, cristal.core.detectors.base_detector.BaseDetector : For more attributes.
        """
        super().__init__(DyCF, n=n_list, config=config, *args, **kwargs)
