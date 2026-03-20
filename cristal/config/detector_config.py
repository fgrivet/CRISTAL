from typing import TYPE_CHECKING, Generic, TypeVar, Union

from ..backend.base_backend import Backend
from ..backend.numpy_backend import NumpyBackend
from ..commons.base_commons import BaseCommons
from ..commons.distance import Distance
from ..commons.incrementer import Incrementer
from ..commons.inverter import Inverter
from ..commons.polynomial_basis import PolynomialBasis
from ..commons.solver import Solver
from ..commons.storage import Storage
from ..commons.threshold_scheme import ThresholdScheme
from ..types import ArrayLike, DTypeLike, Number
from ..preprocessing.base_preprocessor import BasePreprocessor

if TYPE_CHECKING:
    from sklearn.base import OneToOneFeatureMixin
    from sklearn.pipeline import Pipeline


class DetectorConfig(Generic[ArrayLike, DTypeLike]):
    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | None = None,
        preprocessing: Union["Pipeline", BasePreprocessor, "OneToOneFeatureMixin", None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | None = None,
        storage: Storage | None = None,
        threshold_scheme: ThresholdScheme | None = None,
        C: Number = 1,
    ):
        self.backend: Backend = backend or NumpyBackend()
        self.preprocessing = preprocessing
        self.polynomial_basis = polynomial_basis or PolynomialBasis()
        self.storage = storage or Storage()
        self.threshold_scheme = threshold_scheme or ThresholdScheme()
        self.C = C

        self._wire()  # Wire up dependencies with the config object

    def _wire(self):
        for attr in vars(self).values():
            if isinstance(attr, BaseCommons):
                attr.bind(self)


ConfigType = TypeVar("ConfigType", bound=DetectorConfig)


class DynamicDetectorConfig(DetectorConfig[ArrayLike, DTypeLike]):
    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | None = None,
        preprocessing: Union["Pipeline", BasePreprocessor, None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | None = None,
        storage: Storage | None = None,
        threshold_scheme: ThresholdScheme | None = None,
        C: Number = 1,
        inverter: Inverter[ArrayLike, DTypeLike] | None = None,
        incrementer: Incrementer[ArrayLike, DTypeLike] | None = None,
    ):
        self.inverter = inverter or Inverter()
        self.incrementer = incrementer or Incrementer()

        super().__init__(
            backend=backend, preprocessing=preprocessing, polynomial_basis=polynomial_basis, storage=storage, threshold_scheme=threshold_scheme, C=C
        )


class StaticDetectorConfig(DetectorConfig[ArrayLike, DTypeLike]):
    def __init__(
        self,
        backend: Backend[ArrayLike, DTypeLike] | None = None,
        preprocessing: Union["Pipeline", BasePreprocessor, None] = None,
        polynomial_basis: PolynomialBasis[ArrayLike, DTypeLike] | None = None,
        storage: Storage | None = None,
        threshold_scheme: ThresholdScheme | None = None,
        C: Number = 1,
        distance: Distance[ArrayLike, DTypeLike] | None = None,
        solver: Solver[ArrayLike, DTypeLike] | None = None,
    ):
        self.distance = distance or Distance()
        self.solver = solver or Solver()

        super().__init__(
            backend=backend, preprocessing=preprocessing, polynomial_basis=polynomial_basis, storage=storage, threshold_scheme=threshold_scheme, C=C
        )
