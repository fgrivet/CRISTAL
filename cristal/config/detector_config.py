"""Contains the configurations needed in the detectors.

- :class:`DynamicDetectorConfig <cristal.config.detector_config.DynamicDetectorConfig>` for:
    - :class:`DyCF <cristal.core.detectors.dynamic.DyCF>`,
    - :class:`DyCG <cristal.core.detectors.dynamic.DyCG>`,
    - :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`,
    - :class:`KernelCG <cristal.core.detectors.kernel.KernelCG>`.

- :class:`StaticDetectorConfig <cristal.config.detector_config.DynamicDetectorConfig>` for:
    - :class:`UCF <cristal.core.detectors.univariate.UCF>`,
    - :class:`UCG <cristal.core.detectors.univariate.UCG>`,
    - :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`,
    - :class:`NeedleCG <cristal.core.detectors.needle.NeedleCG>`.
"""

from typing import TYPE_CHECKING, Generic, TypeVar, Union

from sklearn.pipeline import Pipeline

from ..backend.base_backend import Backend
from ..backend.numpy_backend import NumpyBackend
from ..backend.torch_backend import TorchBackend
from ..commons.base_commons import BaseCommons
from ..commons.distance import Distance
from ..commons.incrementer import Incrementer
from ..commons.inverter import Inverter
from ..commons.polynomial_basis import PolynomialBasis
from ..commons.solver import Solver
from ..commons.storage import Storage
from ..commons.threshold_scheme import ThresholdScheme
from ..preprocessing.base_preprocessor import BasePreprocessor
from ..types import (
    IMPLEMENTED_BACKEND,
    IMPLEMENTED_DISTANCES,
    IMPLEMENTED_INCREMENTERS,
    IMPLEMENTED_INVERTERS,
    IMPLEMENTED_POLYNOMIAL_BASIS,
    IMPLEMENTED_SOLVERS,
    IMPLEMENTED_STORAGES,
    IMPLEMENTED_THRESHOLD_SCHEMES,
    TORCH_AVAILABLE,
    ArrayLike,
    DTypeLike,
)

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


class DetectorConfig(Generic[ArrayLike, DTypeLike]):
    """Class to store the backend and commons classes for all detectors.

    Parameters
        ----------
        backend : Backend[ArrayLike, DTypeLike] | None, optional
            The backend to use for the computation.
            If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
            else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
        preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
            The preprocessing to class(es) to apply before the computation.
            If provided, must have a :const:`fit_transform` and a :const:`transform` method.
            If :const:`None`, no preprocessing is applied, by default :const:`None`.
        polynomial_basis : PolynomialBasis | None, optional
            The polynomial basis used to generate the moment matrix.
            If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
            by default :const:`None`.
        storage : Storage | None
            The storage to use based on memory available.
            If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
        threshold_scheme : ThresholdScheme | None
            The threshold scheme to use to predict anomalies based on the scores.
            If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
            by default :const:`None`.

    Attributes
    ----------
    backend : :class:`Backend <cristal.backend.base_backend.Backend>`
        The backend to use for the computation.
    preprocessing : :class:`BasePreprocessor <cristal.preprocessing.base_preprocessor.BasePreprocessor>` | TransformerMixin | Pipeline | None
        The preprocessing to class(es) to apply before the computation. Must have a :const:`fit_transform` and a :const:`transform` method.
    polynomial_basis : :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`
        The polynomial basis used to generate the moment matrix.
    storage : :class:`Storage <cristal.commons.storage.Storage>`
        The storage to use based on memory available.
    threshold_scheme : :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`
        The threshold scheme to use to predict anomalies based on the scores.
    """

    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | IMPLEMENTED_BACKEND | None = None,
        preprocessing: Union[BasePreprocessor, "TransformerMixin", "Pipeline", None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | IMPLEMENTED_POLYNOMIAL_BASIS | None = None,
        storage: Storage | IMPLEMENTED_STORAGES | None = None,
        threshold_scheme: ThresholdScheme | IMPLEMENTED_THRESHOLD_SCHEMES | None = None,
    ):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        backend : Backend[ArrayLike, DTypeLike] | None, optional
            The backend to use for the computation.
            If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
            else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
        preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
            The preprocessing to class(es) to apply before the computation.
            If provided, must have a :const:`fit_transform` and a :const:`transform` method.
            If :const:`None`, no preprocessing is applied, by default :const:`None`.
        polynomial_basis : PolynomialBasis | None, optional
            The polynomial basis used to generate the moment matrix.
            If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
            by default :const:`None`.
        storage : Storage | None
            The storage to use based on memory available.
            If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
        threshold_scheme : ThresholdScheme | None
            The threshold scheme to use to predict anomalies based on the scores.
            If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
            by default :const:`None`.
        """
        if backend == "numpy":
            self.backend = NumpyBackend()
        elif backend == "torch":
            if not TORCH_AVAILABLE:
                raise ValueError("torch is not installed and backend is set to torch. Consider installing torch or using NumpyBackend.")
            self.backend = TorchBackend()
        else:
            if backend is not None:
                self.backend: Backend = backend
                """The backend to use for the computation."""
            elif TORCH_AVAILABLE:
                self.backend = TorchBackend()
            else:
                self.backend = NumpyBackend()
        self.preprocessing: Union[BasePreprocessor, "TransformerMixin", "Pipeline", None] = preprocessing
        """The preprocessing to class(es) to apply before the computation. Must have a :const:`fit_transform` and a :const:`transform` method."""
        if isinstance(polynomial_basis, str):
            self.polynomial_basis: PolynomialBasis = PolynomialBasis(basis=polynomial_basis)
            """The polynomial basis used to generate the moment matrix."""
        else:
            self.polynomial_basis = polynomial_basis or PolynomialBasis()
        if isinstance(storage, str):
            self.storage: Storage = Storage(method=storage)
            """The storage to use based on memory available."""
        else:
            self.storage = storage or Storage()
        if isinstance(threshold_scheme, str):
            self.threshold_scheme: ThresholdScheme = ThresholdScheme(scheme=threshold_scheme)
            """The threshold scheme to use to predict anomalies based on the scores."""
        else:
            self.threshold_scheme = threshold_scheme or ThresholdScheme()

        self._wire()  # Wire up dependencies with the config object

    def _wire(self):
        def _bind_recursive(obj):
            if isinstance(obj, BaseCommons):
                obj.bind(self)
            elif isinstance(obj, Pipeline):
                for step in obj.named_steps.values():
                    _bind_recursive(step)
            elif hasattr(obj, "__dict__"):  # To handle objects with attributes
                for attr in vars(obj).values():
                    _bind_recursive(attr)

        _bind_recursive(self)


# pylint: disable=unused-variable
ConfigType = TypeVar("ConfigType", bound=DetectorConfig)  #: The DetectorConfig type.


class DynamicDetectorConfig(DetectorConfig[ArrayLike, DTypeLike]):
    """Class to store the backend and commons classes for dynamic detectors:
    :class:`DyCF <cristal.core.detectors.dynamic.DyCF>`, :class:`DyCG <cristal.core.detectors.dynamic.DyCG>`,
    :class:`KernelCF <cristal.core.detectors.kernel.KernelCF>`, :class:`KernelCG <cristal.core.detectors.kernel.KernelCG>`.

    Parameters
    ----------
    backend : Backend[ArrayLike, DTypeLike] | None, optional
        The backend to use for the computation.
        If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
        else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
    preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
        The preprocessing to class(es) to apply before the computation.
        If provided, must have a :const:`fit_transform` and a :const:`transform` method.
        If :const:`None`, no preprocessing is applied, by default :const:`None`.
    polynomial_basis : PolynomialBasis | None, optional
        The polynomial basis used to generate the moment matrix.
        If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
        by default :const:`None`.
    storage : Storage | None
        The storage to use based on memory available.
        If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
    threshold_scheme : ThresholdScheme | None
        The threshold scheme to use to predict anomalies based on the scores.
        If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
        by default :const:`None`.
    inverter : Inverter | None
        The inversion class to use.
        If :const:`None`, the default value of :class:`Inverter <cristal.commons.inverter.Inverter>`, by default :const:`None`.
    incrementer : Incrementer | None
        The incrementer class to use.
        If :const:`None`, the default value of :class:`Incrementer <cristal.commons.incrementer.Incrementer>`, by default :const:`None`.

    Attributes
    ----------
    inverter : :class:`Inverter <cristal.commons.inverter.Inverter>`
        The inversion class to use.
    incrementer : :class:`Incrementer <cristal.commons.incrementer.Incrementer>`
        The incrementer class to use.

    See Also
    --------
    DetectorConfig : For more attributes.
    """

    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | IMPLEMENTED_BACKEND | None = None,
        preprocessing: Union[BasePreprocessor, "TransformerMixin", "Pipeline", None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | IMPLEMENTED_POLYNOMIAL_BASIS | None = None,
        storage: Storage | IMPLEMENTED_STORAGES | None = None,
        threshold_scheme: ThresholdScheme | IMPLEMENTED_THRESHOLD_SCHEMES | None = None,
        inverter: Inverter[ArrayLike, DTypeLike] | IMPLEMENTED_INVERTERS | None = None,
        incrementer: Incrementer[ArrayLike, DTypeLike] | IMPLEMENTED_INCREMENTERS | None = None,
    ):
        """Class constructor.
        Define the attributes.

        Parameters
        ----------
        backend : Backend[ArrayLike, DTypeLike] | None, optional
            The backend to use for the computation.
            If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
            else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
        preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
            The preprocessing to class(es) to apply before the computation.
            If provided, must have a :const:`fit_transform` and a :const:`transform` method.
            If :const:`None`, no preprocessing is applied, by default :const:`None`.
        polynomial_basis : PolynomialBasis | None, optional
            The polynomial basis used to generate the moment matrix.
            If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
            by default :const:`None`.
        storage : Storage | None
            The storage to use based on memory available.
            If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
        threshold_scheme : ThresholdScheme | None
            The threshold scheme to use to predict anomalies based on the scores.
            If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
            by default :const:`None`.
        inverter : Inverter | None
            The inversion class to use.
            If :const:`None`, the default value of :class:`Inverter <cristal.commons.inverter.Inverter>`, by default :const:`None`.
        incrementer : Incrementer | None
            The incrementer class to use.
            If :const:`None`, the default value of :class:`Incrementer <cristal.commons.incrementer.Incrementer>`, by default :const:`None`.
        """
        if isinstance(inverter, str):
            self.inverter: Inverter = Inverter(method=inverter)
            """The inversion class to use."""
        else:
            self.inverter = inverter or Inverter()
        if isinstance(incrementer, str):
            self.incrementer: Incrementer = Incrementer(method=incrementer)
            """The incrementer class to use."""
        else:
            self.incrementer = incrementer or Incrementer()

        super().__init__(
            backend=backend,
            preprocessing=preprocessing,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
        )


class StaticDetectorConfig(DetectorConfig[ArrayLike, DTypeLike]):
    """Class to store the backend and commons classes for static detectors:
    :class:`UCF <cristal.core.detectors.univariate.UCF>`, :class:`UCG <cristal.core.detectors.univariate.UCG>`,
    :class:`NeedleCF <cristal.core.detectors.needle.NeedleCF>`, :class:`NeedleCG <cristal.core.detectors.needle.NeedleCG>`.

    Parameters
    ----------
    backend : Backend[ArrayLike, DTypeLike] | None, optional
        The backend to use for the computation.
        If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
        else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
    preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
        The preprocessing to class(es) to apply before the computation.
        If provided, must have a :const:`fit_transform` and a :const:`transform` method.
        If :const:`None`, no preprocessing is applied, by default :const:`None`.
    polynomial_basis : PolynomialBasis | None, optional
        The polynomial basis used to generate the moment matrix.
        If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
        by default :const:`None`.
    storage : Storage | None
        The storage to use based on memory available.
        If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
    threshold_scheme : ThresholdScheme | None
        The threshold scheme to use to predict anomalies based on the scores.
        If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
        by default :const:`None`.
    distance : Distance | None
        The distance class to use.
        If :const:`None`, the default value of :class:`Distance <cristal.commons.distance.Distance>`, by default :const:`None`.
    solver : Solver | None
        The solver class to use.
        If :const:`None`, the default value of :class:`Solver <cristal.commons.solver.Solver>`, by default :const:`None`.

    Attributes
    ----------
    distance : :class:`Distance <cristal.commons.distance.Distance>`
        The distance class to use.
    solver : :class:`Solver <cristal.commons.solver.Solver>`
        The solver class to use.

    See Also
    --------
    DetectorConfig : For more attributes.
    """

    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | IMPLEMENTED_BACKEND | None = None,
        preprocessing: Union[BasePreprocessor, "TransformerMixin", "Pipeline", None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | IMPLEMENTED_POLYNOMIAL_BASIS | None = None,
        storage: Storage | IMPLEMENTED_STORAGES | None = None,
        threshold_scheme: ThresholdScheme | IMPLEMENTED_THRESHOLD_SCHEMES | None = None,
        distance: Distance[ArrayLike, DTypeLike] | IMPLEMENTED_DISTANCES | None = None,
        solver: Solver[ArrayLike, DTypeLike] | IMPLEMENTED_SOLVERS | None = None,
    ):
        """Class constructor.
         Define the attributes.

        Parameters
        ----------
        backend : Backend[ArrayLike, DTypeLike] | None, optional
            The backend to use for the computation.
            If :const:`None`, defaults to :class:`TorchBackend <cristal.backend.torch_backend.TorchBackend>` if torch is installed,
            else :class:`NumpyBackend <cristal.backend.numpy_backend.NumpyBackend>`, by default :const:`None`.
        preprocessing : BasePreprocessor | TransformerMixin | Pipeline | None, optional
            The preprocessing to class(es) to apply before the computation.
            If provided, must have a :const:`fit_transform` and a :const:`transform` method.
            If :const:`None`, no preprocessing is applied, by default :const:`None`.
        polynomial_basis : PolynomialBasis | None, optional
            The polynomial basis used to generate the moment matrix.
            If :const:`None`, the default value of :class:`PolynomialBasis <cristal.commons.polynomial_basis.PolynomialBasis>`,
            by default :const:`None`.
        storage : Storage | None
            The storage to use based on memory available.
            If :const:`None`, the default value of :class:`Storage <cristal.commons.storage.Storage>`, by default :const:`None`.
        threshold_scheme : ThresholdScheme | None
            The threshold scheme to use to predict anomalies based on the scores.
            If :const:`None`, the default value of :class:`ThresholdScheme <cristal.commons.threshold_scheme.ThresholdScheme>`,
            by default :const:`None`.
        distance : Distance | None
            The distance class to use.
            If :const:`None`, the default value of :class:`Distance <cristal.commons.distance.Distance>`, by default :const:`None`.
        solver : Solver | None
            The solver class to use.
            If :const:`None`, the default value of :class:`Solver <cristal.commons.solver.Solver>`, by default :const:`None`.
        """
        if isinstance(distance, str):
            self.distance: Distance = Distance(metric=distance)
            """The distance class to use."""
        else:
            self.distance = distance or Distance()
        if isinstance(solver, str):
            self.solver: Solver = Solver(solver=solver)
            """The solver class to use."""
        else:
            self.solver = solver or Solver()

        super().__init__(
            backend=backend,
            preprocessing=preprocessing,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
        )
