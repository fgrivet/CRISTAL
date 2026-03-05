from abc import ABC, abstractmethod
from typing import Generic, Literal, cast

from ...config.detector_config import ConfigType
from ..types import ArrayLike, DTypeLike


class BaseDetector(ABC, Generic[ArrayLike, DTypeLike, ConfigType]):

    def __init__(self, n: int | Literal["auto"], config: ConfigType):
        self.n: int | Literal["auto"] = n
        self.config: ConfigType = config

        # Variables defined during fitting
        self.N: int | None  #: The number of training data
        self.d: int | None  #: The dimension of training data
        self.intrinsic_dim: int | None = None  #: The intrinsic dimension of the data (the one in the CF)
        self.threshold: float | None  #: The threshold for anomaly detection

    def _compute_n(self, n: int | Literal["auto"], N: int) -> int:
        # n = N**(1/(2+d))
        assert self.intrinsic_dim is not None, "Model must be fitted before computing n."

        if n == "auto":
            n = int(N ** (1 / (2 + self.intrinsic_dim)))
            print("auto n =", n)
        assert isinstance(n, int), "Error during the computation of n."
        return n

    @abstractmethod
    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike: ...

    @abstractmethod
    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]: ...

    @abstractmethod
    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]: ...

    @abstractmethod
    def fit(self, X: ArrayLike) -> "BaseDetector": ...

    @abstractmethod
    def is_fitted(self) -> bool: ...

    def score_samples(self, X: ArrayLike) -> ArrayLike:
        assert self.is_fitted(), "Model must be fitted before scoring samples."
        assert isinstance(self.n, int) and self.n > 0, "n must be a positive integer."
        d = X.shape[1]
        assert d == self.d, f"Dimension mismatch between training ({self.d}) and test ({d})."

        # Preprocess the data
        if self.config.preprocessing is not None:
            X = self.config.preprocessing.transform(X)
        if not self.config.backend.is_array_like(X):
            X = self.config.backend.to_array_like(X)

        component_support, component_x = self._compute_components(X)
        component_support, component_x = self._crop_components(component_support, component_x, self.n)
        return self._compute_scores(component_support, component_x)

    def predict_from_scores(self, scores: ArrayLike) -> ArrayLike:
        assert self.threshold is not None, "Model must be fitted before predicting scores."
        return self.config.backend.where(scores > self.threshold, -1, 1)

    def predict(self, X) -> ArrayLike:
        return self.predict_from_scores(self.score_samples(X))


class BaseCGDetector(BaseDetector[ArrayLike, DTypeLike, ConfigType]):
    def __init__(self, detector_class: type[BaseDetector[ArrayLike, DTypeLike, ConfigType]], n: list[int], config: ConfigType, *args, **kwargs):
        if len(n) == 0:
            raise ValueError("n must not be empty")

        # Sort unique values of n
        n = sorted(set(n))
        # Store n_list, n_min, n_max, n_val to avoid recomputing it over and over again
        self.n_list = n
        self.n_min = n[0]
        self.n_max = n[-1]
        self.n_val = len(n)

        # The detector instanciated
        self.detector = detector_class(self.n_max, config, *args, **kwargs)

    # =====================================
    #              CG methods
    # =====================================
    def _polynomial_regression(self, Y: ArrayLike, degree: int, x: ArrayLike | None = None, normalize: bool = False) -> ArrayLike:
        # Compute a polynomial regression of degree 'degree' such as y = f(x)
        assert Y.ndim == 2, "Y must be a 2D ArrayLike."

        # Get the number of scores in the Y values
        _, k = Y.shape

        if k == 1:
            raise ValueError("Can't do regression with only 1 point")

        # If x values are not provided, create a default sequence of x values from 0 to k-1
        if x is None:
            x = self.detector.config.backend.arange(0, k, dtype=self.dtype)

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
        # Compute the R2 score from the data points x and x_hat
        mean = self.config.backend.mean(X, axis=1, keepdims=True)

        ss_tot = self.config.backend.norm2D(X - mean)
        ss_res = self.config.backend.norm2D(X - X_hat)

        R2 = 1.0 - ss_res / ss_tot
        return R2

    def _compute_regressions(self, scores: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        # Compute 2 regressions on the scores and return their R2
        assert self.d is not None, "Detector must be fitted before calling _compute_regressions"
        assert scores.ndim == 2, "scores must be a 2D ArrayLike."

        # Get the current number of scores to get the current degree of UCG (not necessarily the max degree)
        _, n_scores = scores.shape
        current_n = n_scores + self.n_min - 1

        # Setup the regression parameters
        x = self.config.backend.arange(self.n_min, current_n + 1, dtype=self.dtype)  # x values = polynomial degrees
        log_scores = self.config.backend.log(scores)  # The log of the scores
        # The degree of the theoretical polynomial that should fit normal data : the dimension of data
        degree = self.d  # TODO : Change it for 1 with Univariate / Needle

        # Compute R2 of polynomial regression with specific degree. If well fitted then it's nominal data
        R2_poly_reg = self._compute_R2(scores, self._polynomial_regression(scores, degree=degree, x=x))

        # Check where scores < 0 (because log = NaN)
        idx_neg = self.config.backend.where(scores <= 0)
        log_scores[idx_neg] = 1  # Replace with dummy low value to compute the regression. The R2 will be changed back after

        # Compute linear regression on log scores. If well fitted then it's anomalous data
        R2_linear_reg = self._compute_R2(log_scores, self._polynomial_regression(log_scores, degree=1, x=x))

        # Negative scores ==> Anomaly
        R2_poly_reg[idx_neg[0]] = 0
        R2_linear_reg[idx_neg[0]] = 1

        return R2_poly_reg, R2_linear_reg

    # =====================================
    #          private methods
    # =====================================

    def _compute_scores(self, component_support: ArrayLike, component_x: ArrayLike) -> ArrayLike:
        return self.detector._compute_scores(component_support, component_x)

    def _compute_components(self, X: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        return self.detector._compute_components(X)

    def _crop_components(self, component_support: ArrayLike, component_x: ArrayLike, n: int) -> tuple[ArrayLike, ArrayLike]:
        return self.detector._crop_components(component_support, component_x, n)

    # =====================================
    #           public methods
    # =====================================

    def fit(self, X: ArrayLike) -> BaseDetector:
        # Save the number of samples, and dimensionality of training data
        self.N, self.d = self.backend.shape(X)
        return self.detector.fit(X)

    def is_fitted(self) -> bool:
        return self.detector.is_fitted()

    def score_samples(self, X: ArrayLike) -> ArrayLike:
        N = X.shape[0]

        # Create the array to store all the scores
        all_scores = self.detector.config.backend.zeros((N, self.n_val), dtype=cast(DTypeLike, X.dtype))
        # Compute the full M and V
        M, V = self._compute_components(X)

        # For each degree to compute, crop M and V and compute the corresponding scores
        for i, n in enumerate(self.n_list):
            M_crop, V_crop = self._crop_components(M, V, n)
            all_scores[:, i] = self._compute_scores(M_crop, V_crop)

        return all_scores

    def predict_from_scores(self, scores):
        return self.detector.predict_from_scores(self.unique_score(scores))

    def unique_score(self, scores: ArrayLike, method: Literal["clip", "linear"] = "linear") -> ArrayLike:
        # Based on the scores for all n, aggregate them to return only on score per data point

        # Compute regressions
        R2_poly_reg, R2_linear_reg = self._compute_regressions(scores)

        if method == "clip":
            # 0 if nominal, linear from 0 to 1 if anomaly
            return self.backend.clip(R2_linear_reg - R2_poly_reg, min_=0)

        # Linear from 0 to 1
        return (R2_linear_reg - R2_poly_reg + 1) / 2

    def __getattr__(self, name):
        return getattr(self.detector, name)
