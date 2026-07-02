"""Contains the kernel version of the Christoffel function based outlier detection algorithm, adapted from :cite:t:`askari2018kernel`."""

import logging
import math
from typing import Literal, cast

from ...commons.distance import Distance
from ...config.detector_config import DynamicDetectorConfig
from ...types import ArrayLike, DTypeLike, Number
from .base_detector import BaseCGDetector, BaseDetector

logger = logging.getLogger(__name__)


class KernelCF(BaseDetector[ArrayLike, DTypeLike, DynamicDetectorConfig]):
    """Class to compute the kernel version of the Christoffel function based outlier detection scores and predictions, adapted from :cite:t:`askari2018kernel`.

    Attributes
    ----------
    kernel : Literal["linear", "rbf"]
        The kernel type.
    kernel_func : Callable[[ArrayLike, ArrayLike | None], ArrayLike]
        The kernel function associated to :attr:`kernel`.
    rho : Literal["auto"] | Number
        The parameter rho for the regularization of the kernel. If :const:`auto`, set to :math:`\\frac{\\|G\\|_F}{C \\times \\sqrt{n}}` during :func:`fit`.
    sigma : Literal["auto"] | Number
        The parameter sigma for the rbf kernel. If :const:`auto`, set to :math:`\\sqrt{d} / 2` during :func:`fit`.
    C : Number
        A constant parameter used in the computation of :attr:`rho`.
    X_train: ArrayLike | None
        The training data of shape (N_samples_train, d), set during :func:`fit`.
    G : ArrayLike | None
        The gram matrix of shape (N_samples_train, N_samples_train), set during :func:`fit`.
    G_inv : ArrayLike | None
        The inverse of the gram matrix of shape (N_samples_train, N_samples_train), set during :func:`fit`.

    See Also
    --------
    cristal.core.detectors.base_detector.BaseDetector : For more attributes.

    Examples
    --------
    See :doc:`/examples` or :doc:`/user_guide`.
    """

    def __init__(
        self,
        n: int | Literal["auto"],
        config: DynamicDetectorConfig[ArrayLike, DTypeLike] = DynamicDetectorConfig(),
        kernel: Literal["linear", "rbf"] = "rbf",
        rho: Literal["auto"] | Number = "auto",
        sigma: Literal["auto"] | Number = "auto",
        C: Number = 1,
    ):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        n : int | Literal["auto"]
            The degree of the CF. If :const:`auto`, set to :math:`N^{1 / (2+intrinsic\\_dim)}` during :func:`fit`.
        config : DynamicDetectorConfig[ArrayLike, DTypeLike], optional
            The configuration of the model, by default DynamicDetectorConfig().
        kernel : Literal["linear", "rbf"], optional
            The kernel type, by default "rbf".
        rho : Literal["auto"] | Number, optional
            The parameter rho for the regularization of the kernel. If :const:`auto`, set to :math:`\\frac{\\|G\\|_F}{C \\times \\sqrt{n}}` during :func:`fit`, by default :const:`auto`.
        sigma : Literal["auto"] | Number, optional
            The parameter sigma for the rbf kernel. If :const:`auto`, set to :math:`\\sqrt{d} / 2` during :func:`fit`tion_, by default :const:`auto`.
        C : Number, optional
            A constant parameter used in the computation of :attr:`rho`, by default 1.
        """
        super().__init__(n, config)

        # Variables specific to KernelCF
        self.kernel: Literal["linear", "rbf"] = kernel  #: The kernel type.
        self.kernel_func = self._rbf_kernel if self.kernel == "rbf" else self._linear_kernel
        """The kernel function associated to :attr:`kernel`."""
        self.rho: Literal["auto"] | Number = rho
        """The parameter rho for the regularization of the kernel. If :const:`auto`, set to :math:`\\frac{\\|G\\|_F}{C \\times \\sqrt{n}}` during :func:`fit`."""
        self.sigma: Literal["auto"] | Number = sigma
        """The parameter sigma for the rbf kernel. If :const:`auto`, set to :math:`\\sqrt{d} / 2` during :func:`fit`."""
        self.C = C  #: A constant parameter used in the computation of :attr:`rho`.

        # Variables defined during fitting specific to KernelCF
        self.X_train: ArrayLike | None = None  #: The training data of shape (N_samples_train, d).
        self.G = None  #: The gram matrix of shape (N_samples_train, N_samples_train).
        self.G_inv = None  #: The inverse of the gram matrix of shape (N_samples_train, N_samples_train).

    def _rbf_kernel(self, X: ArrayLike, Y: ArrayLike | None = None) -> ArrayLike:
        """Compute the RBF kernel between two ArrayLike: :math:`K(X, Y) = \\exp\\left( - \\frac{\\|X - Y\\|^2_2}{2 \\sigma^2} \\right)`.

        Parameters
        ----------
        X : ArrayLike
            A 2D ArrayLike of shape (N, d).
        Y : ArrayLike | None, optional
            A 2D ArrayLike of shape (M, d). If :const:`None`, set to :attr:`X`, by default None.

        Returns
        -------
        ArrayLike
            The RBF kernel of each sample of shape (N, M).

        Raises
        ------
        ValueError
            If :attr:`sigma` is not defined yet.
        """
        if self.sigma == "auto":
            raise ValueError(f"Sigma must be specified when using RBF kernel. Got {self.sigma}.")
        gamma = 1 / (2 * self.sigma**2)
        distance = Distance(metric="euclidean")
        distance.backend = self.config.backend
        D = distance(X, Y, p=2)
        return self.config.backend.exp(-gamma * D)

    def _linear_kernel(self, X: ArrayLike, Y: ArrayLike | None = None) -> ArrayLike:
        """Compute the linear kernel between two ArrayLike: :math:`K(X, Y) = \\left( 1 + X Y^T \\right)^n`.

        Parameters
        ----------
        X : ArrayLike
            A 2D ArrayLike of shape (N, d).
        Y : ArrayLike | None, optional
            A 2D ArrayLike of shape (M, d). If :const:`None`, set to :attr:`X`, by default None.

        Returns
        -------
        ArrayLike
            The linear kernel of each sample of shape (N, M).
        """
        if Y is None:
            Y = X
        return (1 + X @ Y.T) ** self.n  # type: ignore

    def _compute_sigma(self, sigma: Literal["auto"] | Number, d: int) -> float:
        """Compute the parameter sigma for the rbf kernel if :attr:`sigma` is :const:`auto`.

        Parameters
        ----------
        sigma : Literal["auto"] | Number
            The current parameter :attr:`sigma`.
        d : int
            The dimension of data.

        Returns
        -------
        float
            The computed parameter sigma :math:`\\sqrt{d} / 2` if :attr:`sigma` is :const:`auto`, else :attr:`sigma`.
        """
        if sigma == "auto":
            return math.sqrt(d) / 2
        return float(sigma)

    def _compute_rho(self, rho: Literal["auto"] | Number, n: int) -> float:
        """Compute the parameter rho for the regularization of the kernel if :attr:`rho` is :const:`auto`.

        Parameters
        ----------
        rho : Literal["auto"] | Number
            The current parameter rho.
        n : int
            The degree of the CF.

        Returns
        -------
        float
            The computed parameter rho :math:`\\frac{\\|G\\|_F}{C \\times \\sqrt{n}}` if :attr:`rho` is :const:`auto` else :attr:`rho`.

        Raises
        ------
        ValueError
            If :attr:`G` is not computed yet.
        """
        if rho == "auto":
            if self.G is None:
                raise ValueError("G must be set before computing rho.")
            return self.config.backend.norm(self.G, p="fro") / (self.C * math.sqrt(n))
        return float(rho)

    def _compute_scores(self, component_support: ArrayLike, component_X: ArrayLike) -> ArrayLike:
        """Compute the scores for each sample with the formula :math:`(\\gamma - g^T (\\rho I + G)^{-1} g) / \\rho`.

        Parameters
        ----------
        component_support : ArrayLike
            :math:`g = K(X\\_train, X)` of shape (N_samples_train, N_samples_test).
        component_X : ArrayLike
            :math:`\\gamma = (K(X, X)_{ii})_{1\\leq i\\leq N\\_samples\\_test}` of shape (N_samples_test,).

        Returns
        -------
        ArrayLike
            The scores for each sample of shape (N_samples_test,)
        """
        return (component_X - self.config.backend.einsum("ni,nn,ni->i", component_support, self.G_inv, component_support)) / self.rho

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Compute for each sample the components needed to calculate the scores.

        Parameters
        ----------
        X : ArrayLike
            The matrix containing the samples of shape (N_samples_test, d).

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            :math:`g = K(X\\_train, X)` of shape (N_samples_train, N_samples_test), :math:`\\gamma = (K(X, X)_{ii})_{1\\leq i\\leq N\\_samples\\_test}` of shape (N_samples_test,).

        Raises
        ------
        ValueError
            If the model is not fitted.
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted.")
        g = self.kernel_func(self.X_train, X)
        gamma = self.config.backend.diag(self.kernel_func(X))
        return g, gamma

    def _crop_components(self, component_support: ArrayLike, component_X: ArrayLike, n_crop: int) -> tuple[ArrayLike, ArrayLike]:
        """Do nothing, in the kernel version, matrices shapes are based on the number of samples in training :attr:`N`, not on the polynomial degree :attr:`n`.

        Parameters
        ----------
        component_support : ArrayLike
            :math:`g = K(X\\_train, X)` of shape (N_samples_train, N_samples_test).
        component_X : ArrayLike
            :math:`\\gamma = (K(X, X)_{ii})_{1\\leq i\\leq N\\_samples\\_test}` of shape (N_samples_test,).
        n_crop : int
            The maximum polynomial basis degree. Must be lower or equal to :attr:`n`.

        Returns
        -------
        tuple[ArrayLike, ArrayLike]
            :math:`g = K(X\\_train, X)` of shape (N_samples_train, N_samples_test), :math:`\\gamma = (K(X, X)_{ii})_{1\\leq i\\leq N\\_samples\\_test}` of shape (N_samples_test,).
        """
        if self.kernel == "linear":
            return component_support ** (n_crop / self.n), component_X ** (n_crop / self.n)
        return component_support, component_X

    def update(self, X: ArrayLike, online: Literal["constant", "increment"]):
        logger.warning("Update method not implemented. Doing nothing. Use the fit method with the new support instead.")
        return self

    def fit(self, X: ArrayLike) -> BaseDetector:
        """Fit the model to the data :attr:`X`.
        Compute :attr:`sigma`, the gram matrix :math:`G = K(X\\_train, X\\_train)` of shape (N_samples_train, N_samples_train),
        the regularization parameter :attr:`rho`, the inverse of :attr:`G`: :attr:`G_inv`, and the :attr:`threshold` used for prediction.

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
        """
        if X.ndim != 2:
            raise ValueError(f"X must be a 2D ArrayLike. Got {X.shape}.")

        # Define The number of training data and the diension of training data
        N, d = X.shape

        # TODO Storage

        # Preprocess the data
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.fit_transform(X)  # type: ignore
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)

        # Save the information on training data
        self.N = cast(int, N)
        self.d = cast(int, d)
        self.intrinsic_dim = self.d
        self.X_train = X

        # Define the degree if set to auto using n = N**(1/(2+d)) with d=1 because univariate
        self.n = self._compute_n(self.n, self.N)

        self.sigma = self._compute_sigma(self.sigma, self.d)
        self.G = self.kernel_func(X, X)
        self.rho = self._compute_rho(self.rho, self.n)
        self.G_inv = self.config.inverter(self.G, eps=self.rho)

        # Compute the threshold
        self.threshold = self.config.threshold_scheme(self.n, self.d)

        assert self.is_fitted(), "Error during fitting."
        return self

    def is_fitted(self) -> bool:
        return self.N is not None and self.d is not None and self.X_train is not None and self.threshold is not None and isinstance(self.n, int)


class KernelCG(BaseCGDetector):
    """Class to compute the kernel version of the Christoffel function based outlier detection scores, and predictions based on the growth of scores as the degree :attr:`n` increases, adapted from :cite:t:`askari2018kernel`.

    .. caution::

        This class only works with :const:`kernel=linear`, and may not work well.

    See Also
    --------
    KernelCF, cristal.core.detectors.base_detector.BaseDetector : For more attributes.
    """

    def __init__(
        self,
        n_list: list[int],
        config: DynamicDetectorConfig[ArrayLike, DTypeLike] = DynamicDetectorConfig(),
        kernel: Literal["linear", "rbf"] = "rbf",
        rho: Literal["auto"] | Number = "auto",
        sigma: Literal["auto"] | Number = "auto",
        C: Number = 1,
        *args,
        **kwargs,
    ):
        if kernel == "rbf":
            raise ValueError("Growth accoding to n impossible with RBF kernel since it does not depend on n.")
        super().__init__(KernelCF, n=n_list, config=config, kernel=kernel, rho=rho, sigma=sigma, C=C, *args, **kwargs)
