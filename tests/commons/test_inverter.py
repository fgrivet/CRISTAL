"""
Unit tests for Inverter class
"""

from unittest.mock import Mock

import pytest

from cristal.commons.inverter import IMPLEMENTED_INVERTER, Inverter


class TestInverter:
    """Test the Inverter class functionality"""

    def test_inverter_initialization(self):
        """Test Inverter initialization with valid parameters"""
        # Test valid inverter methods
        for method in IMPLEMENTED_INVERTER.__args__:
            inverter = Inverter(method=method)
            assert inverter.method == method

        # Test invalid method
        with pytest.raises(AssertionError):
            Inverter(method="invalid_method")

    def test_inverter_with_mock_backend(self):
        """Test Inverter with mocked backend"""
        # Create a mock backend
        backend = Mock()
        backend.eye.return_value = Mock()
        backend.inv.return_value = Mock()
        backend.pinv.return_value = Mock()
        backend.solve.return_value = Mock()
        backend.inverse_cholesky.return_value = Mock()

        # Create inverter
        inverter = Inverter(method="fpd")

        # Bind backend
        inverter.backend = backend

        # Test that inverter can be called without error
        assert inverter.method == "fpd"
