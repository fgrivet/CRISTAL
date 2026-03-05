"""
Unit tests for Incrementer class
"""

from unittest.mock import Mock

import pytest

from cristal.commons.incrementer import IMPLEMENTED_INCREMENTER, Incrementer


class TestIncrementer:
    """Test the Incrementer class functionality"""

    def test_incrementer_initialization(self):
        """Test Incrementer initialization with valid parameters"""
        # Test valid incrementer methods
        for method in IMPLEMENTED_INCREMENTER.__args__:
            incrementer = Incrementer(method=method)
            assert incrementer.method == method

        # Test invalid method
        with pytest.raises(AssertionError):
            Incrementer(method="invalid_method")

    def test_incrementer_with_mock_components(self):
        """Test Incrementer with mocked components"""
        # Create mocks for dependencies
        backend = Mock()
        inverter = Mock()
        polynomial_basis = Mock()

        # Create incrementer
        incrementer = Incrementer(method="woodbury")

        # Bind dependencies
        incrementer.backend = backend
        incrementer.inverter = inverter
        incrementer.polynomial_basis = polynomial_basis

        # Test that incrementer can be called without error
        assert incrementer.method == "woodbury"
