"""
Unit tests for ThresholdScheme class
"""

import pytest

from cristal.commons.threshold_scheme import IMPLEMENTED_THRESHOLD_SCHEME, ThresholdScheme


class TestThresholdScheme:
    """Test the ThresholdScheme class functionality"""

    def test_threshold_scheme_initialization(self):
        """Test ThresholdScheme initialization with valid parameters"""
        # Test valid threshold schemes
        for scheme in IMPLEMENTED_THRESHOLD_SCHEME.__args__:
            threshold_scheme = ThresholdScheme(scheme=scheme)
            assert threshold_scheme.scheme == scheme

        # Test invalid scheme
        with pytest.raises(AssertionError):
            ThresholdScheme(scheme="invalid_scheme")

    def test_compute_threshold_comb(self):
        """Test comb threshold computation"""
        threshold_scheme = ThresholdScheme(scheme="comb")

        # Test with integer n
        result = threshold_scheme.compute_threshold(2, 3, 1)
        assert isinstance(result, int) or isinstance(result, float)

    def test_compute_threshold_vu(self):
        """Test vu threshold computation"""
        threshold_scheme = ThresholdScheme(scheme="vu")

        # Test with integer n
        result = threshold_scheme.compute_threshold(2, 3, 1)
        assert isinstance(result, int) or isinstance(result, float)

    def test_compute_threshold_vuC(self):
        """Test vuC threshold computation"""
        threshold_scheme = ThresholdScheme(scheme="vuC")

        # Test with integer n
        result = threshold_scheme.compute_threshold(2, 3, 2)
        assert isinstance(result, int) or isinstance(result, float)

    def test_compute_threshold_constant(self):
        """Test constant threshold computation"""
        threshold_scheme = ThresholdScheme(scheme="constant")

        # Test with integer n
        result = threshold_scheme.compute_threshold(2, 3, 5)
        assert result == 5
