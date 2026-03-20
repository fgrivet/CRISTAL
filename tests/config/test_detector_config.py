import unittest
from unittest.mock import Mock

from cristal.backend.numpy_backend import NumpyBackend
from cristal.commons.distance import Distance
from cristal.commons.incrementer import Incrementer
from cristal.commons.inverter import Inverter
from cristal.commons.polynomial_basis import PolynomialBasis
from cristal.commons.solver import Solver
from cristal.commons.storage import Storage
from cristal.commons.threshold_scheme import ThresholdScheme
from cristal.config.detector_config import DetectorConfig, DynamicDetectorConfig, StaticDetectorConfig


class TestDetectorConfig(unittest.TestCase):
    """Test the DetectorConfig class functionality"""

    def test_detector_config_initialization(self):
        """Test DetectorConfig initialization with valid parameters"""
        # Create config with default values
        config = DetectorConfig()
        self.assertIsInstance(config.backend, NumpyBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)
        self.assertEqual(config.C, 1)

    def test_detector_config_custom_params(self):
        """Test DetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()
        C = 5

        config = DetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
            C=C,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)
        self.assertEqual(config.C, C)

    def test_detector_config_wire_method(self):
        """Test _wire method binds dependencies correctly"""
        config = DetectorConfig()

        # Polynomial basis is the only class of DetectorConfig that has a requirement (backend)

        # Check that polynomial_basis is bound
        self.assertIn(config.backend, config.polynomial_basis._dependencies.values())  # type: ignore


class TestDynamicDetectorConfig(unittest.TestCase):
    """Test the DynamicDetectorConfig class functionality"""

    def test_dynamic_detector_config_initialization(self):
        """Test DynamicDetectorConfig initialization with valid parameters"""
        config = DynamicDetectorConfig()
        self.assertIsInstance(config.backend, NumpyBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)
        self.assertEqual(config.C, 1)
        self.assertIsInstance(config.inverter, Inverter)
        self.assertIsInstance(config.incrementer, Incrementer)

    def test_dynamic_detector_config_custom_params(self):
        """Test DynamicDetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()
        C = 5
        inverter = Inverter()
        incrementer = Incrementer()

        config = DynamicDetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
            C=C,
            inverter=inverter,
            incrementer=incrementer,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)
        self.assertEqual(config.C, C)
        self.assertEqual(config.inverter, inverter)
        self.assertEqual(config.incrementer, incrementer)

    def test_dynamic_detector_config_wire_method(self):
        """Test _wire method in DynamicDetectorConfig binds all dependencies"""
        config = DynamicDetectorConfig()

        # Check that polynomial_basis is bound
        self.assertIn(config.backend, config.polynomial_basis._dependencies.values())  # type: ignore

        # Check that inverter is bound
        self.assertIn(config.backend, config.inverter._dependencies.values())  # type: ignore

        # Check that incrementer is bound
        self.assertIn(config.backend, config.incrementer._dependencies.values())  # type: ignore
        self.assertIn(config.inverter, config.incrementer._dependencies.values())  # type: ignore
        self.assertIn(config.polynomial_basis, config.incrementer._dependencies.values())  # type: ignore


class TestStaticDetectorConfig(unittest.TestCase):
    """Test the StaticDetectorConfig class functionality"""

    def test_static_detector_config_initialization(self):
        """Test StaticDetectorConfig initialization with valid parameters"""
        config = StaticDetectorConfig()
        self.assertIsInstance(config.backend, NumpyBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)
        self.assertEqual(config.C, 1)
        self.assertIsInstance(config.distance, Distance)
        self.assertIsInstance(config.solver, Solver)

    def test_static_detector_config_custom_params(self):
        """Test StaticDetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()
        C = 5
        distance = Distance()
        solver = Solver()

        config = StaticDetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
            C=C,
            distance=distance,
            solver=solver,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)
        self.assertEqual(config.C, C)
        self.assertEqual(config.distance, distance)
        self.assertEqual(config.solver, solver)

    def test_static_detector_config_wire_method(self):
        """Test _wire method in StaticDetectorConfig binds all dependencies"""
        config = StaticDetectorConfig()

        # Check that polynomial_basis is bound
        self.assertIn(config.backend, config.polynomial_basis._dependencies.values())  # type: ignore

        # Check that distance is bound
        self.assertIn(config.backend, config.distance._dependencies.values())  # type: ignore

        # Check that solver is bound
        self.assertIn(config.backend, config.solver._dependencies.values())  # type: ignore
