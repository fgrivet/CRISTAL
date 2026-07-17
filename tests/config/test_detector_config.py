import unittest
from unittest.mock import patch

from sklearn.pipeline import Pipeline

from cristal.backend.numpy_backend import NumpyBackend
from cristal.backend.torch_backend import TorchBackend
from cristal.commons.distance import Distance
from cristal.commons.incrementer import Incrementer
from cristal.commons.inverter import Inverter
from cristal.commons.polynomial_basis import PolynomialBasis
from cristal.commons.solver import Solver
from cristal.commons.storage import Storage
from cristal.commons.threshold_scheme import ThresholdScheme
from cristal.config.detector_config import DetectorConfig, DynamicDetectorConfig, StaticDetectorConfig
from cristal.preprocessing.scalers import MinMaxScaler


class TestDetectorConfig(unittest.TestCase):
    """Test the DetectorConfig class functionality"""

    def test_detector_config_initialization(self):
        """Test DetectorConfig initialization with valid parameters"""
        # Create config with default values
        config = DetectorConfig()
        self.assertIsInstance(config.backend, TorchBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)

    def test_detector_config_custom_params(self):
        """Test DetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()

        config = DetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)

    def test_detector_config_custom_str_params(self):
        """Test DetectorConfig with custom parameters as str"""
        polynomial_basis = "monomials"
        storage = "full"
        threshold_scheme = "comb"

        config = DetectorConfig(
            backend="numpy",
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
        )

        self.assertIsInstance(config.backend, NumpyBackend)
        self.assertEqual(config.polynomial_basis.basis, polynomial_basis)
        self.assertEqual(config.storage.method, storage)
        self.assertEqual(config.threshold_scheme.scheme, threshold_scheme)

        config = DetectorConfig(
            backend="torch",
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
        )
        self.assertIsInstance(config.backend, TorchBackend)

    def test_detector_config_torch_not_available_with_torch_backend(self):
        """Test that DetectorConfig raises ValueError when torch is not available and backend='torch'"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            with self.assertRaises(ValueError) as context:
                DetectorConfig(backend="torch")
            self.assertIn("torch is not installed", str(context.exception))

    def test_detector_config_torch_not_available_default_backend(self):
        """Test that DetectorConfig uses NumpyBackend when torch is not available and no backend specified"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            config = DetectorConfig()
            self.assertIsInstance(config.backend, NumpyBackend)

    def test_detector_config_wire_method_with_pipeline(self):
        """Test _wire method binds dependencies correctly for sklearn Pipeline"""
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
            ]
        )

        backend = NumpyBackend()

        # Check that backend is bound to the pipeline's scaler
        self.assertIsNone(pipeline.named_steps["scaler"].backend)

        config = DetectorConfig(backend=backend, preprocessing=pipeline)

        # Check that backend is bound to the pipeline's scaler
        self.assertEqual(backend, pipeline.named_steps["scaler"].backend)

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
        self.assertIsInstance(config.backend, TorchBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)
        self.assertIsInstance(config.inverter, Inverter)
        self.assertIsInstance(config.incrementer, Incrementer)

    def test_dynamic_detector_config_custom_params(self):
        """Test DynamicDetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()
        inverter = Inverter()
        incrementer = Incrementer()

        config = DynamicDetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
            inverter=inverter,
            incrementer=incrementer,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)
        self.assertEqual(config.inverter, inverter)
        self.assertEqual(config.incrementer, incrementer)

    def test_dynamic_detector_config_custom_str_params(self):
        """Test DynamicDetectorConfig with custom parameters as str"""
        inverter = "fpd"
        incrementer = "inverse"

        config = DynamicDetectorConfig(
            inverter=inverter,
            incrementer=incrementer,
        )

        self.assertEqual(config.inverter.method, inverter)
        self.assertEqual(config.incrementer.method, incrementer)

    def test_dynamic_detector_config_torch_not_available_with_torch_backend(self):
        """Test that DynamicDetectorConfig raises ValueError when torch is not available and backend='torch'"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            with self.assertRaises(ValueError) as context:
                DynamicDetectorConfig(backend="torch")
            self.assertIn("torch is not installed", str(context.exception))

    def test_dynamic_detector_config_torch_not_available_default_backend(self):
        """Test that DynamicDetectorConfig uses NumpyBackend when torch is not available and no backend specified"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            config = DynamicDetectorConfig()
            self.assertIsInstance(config.backend, NumpyBackend)

    def test_dynamic_detector_config_wire_method_with_pipeline(self):
        """Test _wire method in DynamicDetectorConfig binds dependencies for sklearn Pipeline"""
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
            ]
        )

        backend = NumpyBackend()

        # Check that backend is bound to the pipeline's scaler
        self.assertIsNone(pipeline.named_steps["scaler"].backend)

        config = DynamicDetectorConfig(backend=backend, preprocessing=pipeline)

        # Check that backend is bound to the pipeline's scaler
        self.assertEqual(backend, pipeline.named_steps["scaler"].backend)

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
        self.assertIsInstance(config.backend, TorchBackend)
        self.assertIsInstance(config.polynomial_basis, PolynomialBasis)
        self.assertIsInstance(config.storage, Storage)
        self.assertIsInstance(config.threshold_scheme, ThresholdScheme)
        self.assertIsInstance(config.distance, Distance)
        self.assertIsInstance(config.solver, Solver)

    def test_static_detector_config_custom_params(self):
        """Test StaticDetectorConfig with custom parameters"""
        backend = NumpyBackend()
        polynomial_basis = PolynomialBasis()
        storage = Storage()
        threshold_scheme = ThresholdScheme()
        distance = Distance()
        solver = Solver()

        config = StaticDetectorConfig(
            backend=backend,
            preprocessing=None,
            polynomial_basis=polynomial_basis,
            storage=storage,
            threshold_scheme=threshold_scheme,
            distance=distance,
            solver=solver,
        )

        self.assertEqual(config.backend, backend)
        self.assertEqual(config.polynomial_basis, polynomial_basis)
        self.assertEqual(config.storage, storage)
        self.assertEqual(config.threshold_scheme, threshold_scheme)
        self.assertEqual(config.distance, distance)
        self.assertEqual(config.solver, solver)

    def test_static_detector_config_custom_str_params(self):
        """Test StaticDetectorConfig with custom parameters as str"""
        distance = "mahalanobis"
        solver = "inverse"

        config = StaticDetectorConfig(
            distance=distance,
            solver=solver,
        )

        self.assertEqual(config.distance.metric, distance)
        self.assertEqual(config.solver.solver, solver)

    def test_static_detector_config_torch_not_available_with_torch_backend(self):
        """Test that StaticDetectorConfig raises ValueError when torch is not available and backend='torch'"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            with self.assertRaises(ValueError) as context:
                StaticDetectorConfig(backend="torch")
            self.assertIn("torch is not installed", str(context.exception))

    def test_static_detector_config_torch_not_available_default_backend(self):
        """Test that StaticDetectorConfig uses NumpyBackend when torch is not available and no backend specified"""
        with patch("cristal.config.detector_config.TORCH_AVAILABLE", False):
            config = StaticDetectorConfig()
            self.assertIsInstance(config.backend, NumpyBackend)

    def test_static_detector_config_wire_method_with_pipeline(self):
        """Test _wire method in StaticDetectorConfig binds dependencies for sklearn Pipeline"""
        pipeline = Pipeline(
            [
                ("scaler", MinMaxScaler()),
            ]
        )

        backend = NumpyBackend()

        # Check that backend is bound to the pipeline's scaler
        self.assertIsNone(pipeline.named_steps["scaler"].backend)

        config = StaticDetectorConfig(backend=backend, preprocessing=pipeline)

        # Check that backend is bound to the pipeline's scaler
        self.assertEqual(backend, pipeline.named_steps["scaler"].backend)

    def test_static_detector_config_wire_method(self):
        """Test _wire method in StaticDetectorConfig binds all dependencies"""
        config = StaticDetectorConfig()

        # Check that polynomial_basis is bound
        self.assertIn(config.backend, config.polynomial_basis._dependencies.values())  # type: ignore

        # Check that distance is bound
        self.assertIn(config.backend, config.distance._dependencies.values())  # type: ignore

        # Check that solver is bound
        self.assertIn(config.backend, config.solver._dependencies.values())  # type: ignore
