"""
Unit tests for ThresholdScheme class
"""

import unittest

import numpy as np

from cristal.commons.threshold_scheme import IMPLEMENTED_THRESHOLD_SCHEMES, ThresholdScheme


class TestThresholdScheme(unittest.TestCase):
    """Test the ThresholdScheme class functionality"""

    def test_threshold_scheme_initialization(self):
        """Test ThresholdScheme initialization with valid parameters"""
        # Test valid threshold schemes
        for scheme in IMPLEMENTED_THRESHOLD_SCHEMES.__args__:
            C = np.random.randint(1, 8)
            threshold_scheme = ThresholdScheme(scheme=scheme, C=C)
            self.assertEqual(threshold_scheme.scheme, scheme)
            self.assertEqual(threshold_scheme.C, C)

        # Test invalid scheme
        self.assertRaises(ValueError, ThresholdScheme, scheme="invalid_scheme")

    def test_compute_threshold_comb(self):
        """Test comb threshold computation"""

        # Test with n=2, d=3
        comb_threshold_scheme = ThresholdScheme(scheme="comb", C=1)
        result_2_3_1 = comb_threshold_scheme(2, 3)
        comb_threshold_scheme = ThresholdScheme(scheme="comb", C=8)
        result_2_3_8 = comb_threshold_scheme(2.0, 3)

        self.assertEqual(result_2_3_1, 10)
        self.assertEqual(result_2_3_8, 10)

        # Test with n=5, d=8
        comb_threshold_scheme = ThresholdScheme(scheme="comb", C=4.0)
        result_5_8_4 = comb_threshold_scheme(5, 8)
        self.assertEqual(result_5_8_4, 1287)

    def test_compute_threshold_vu(self):
        """Test vu threshold computation"""

        # Test with n=2, d=3
        vu_threshold_scheme = ThresholdScheme(scheme="vu", C=1)
        result_2_3_1 = vu_threshold_scheme(2, 3)
        vu_threshold_scheme = ThresholdScheme(scheme="vu", C=8)
        result_2_3_8 = vu_threshold_scheme(2.0, 3)
        self.assertEqual(result_2_3_1, 2 ** (3 * 3 / 2))
        self.assertEqual(result_2_3_8, 2 ** (3 * 3 / 2))

        # Test with n=5, d=8
        vu_threshold_scheme = ThresholdScheme(scheme="vu", C=4.0)
        result_5_8_4 = vu_threshold_scheme(5, 8)
        self.assertEqual(result_5_8_4, 5 ** (3 * 8 / 2))

    def test_compute_threshold_vuC(self):
        """Test vuC threshold computation"""

        # Test with n=2, d=3
        vuC_threshold_scheme = ThresholdScheme(scheme="vuC", C=1)
        result_2_3_1 = vuC_threshold_scheme(2, 3)
        vuC_threshold_scheme = ThresholdScheme(scheme="vuC", C=8)
        result_2_3_8 = vuC_threshold_scheme(2.0, 3)
        self.assertEqual(result_2_3_1, 2 ** (3 * 3 / 2))
        self.assertEqual(result_2_3_8, 2 ** (3 * 3 / 2) / 8)

        # Test with n=5, d=8
        vuC_threshold_scheme = ThresholdScheme(scheme="vuC", C=4.0)
        result_5_8_4 = vuC_threshold_scheme(5, 8)
        self.assertEqual(result_5_8_4, 5 ** (3 * 8 / 2) / 4)

    def test_compute_threshold_constant(self):
        """Test constant threshold computation"""

        # Test with n=2, d=3
        constant_threshold_scheme = ThresholdScheme(scheme="constant", C=1)
        result_2_3_1 = constant_threshold_scheme(2, 3)
        constant_threshold_scheme = ThresholdScheme(scheme="constant", C=8)
        result_2_3_8 = constant_threshold_scheme(2.0, 3)
        self.assertEqual(result_2_3_1, 1)
        self.assertEqual(result_2_3_8, 8)

        # Test with n=5, d=8
        constant_threshold_scheme = ThresholdScheme(scheme="constant", C=4.0)
        result_5_8_4 = constant_threshold_scheme(5, 8)
        self.assertEqual(result_5_8_4, 4)
