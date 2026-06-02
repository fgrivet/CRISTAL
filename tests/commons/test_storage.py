"""
Unit tests for Storage class
"""

import unittest

import numpy as np

from cristal.commons.storage import IMPLEMENTED_STORAGES, Storage


class TestStorage(unittest.TestCase):
    """Test the Storage class functionality"""

    def test_storage_initialization(self):
        """Test Storage initialization with valid parameters"""
        # Test valid storage methods
        for method in IMPLEMENTED_STORAGES.__args__:
            storage = Storage(method=method, batch_size=10)
            self.assertEqual(storage.method, method)
            self.assertEqual(storage.batch_size, 10)

        # Test invalid method
        self.assertRaises(ValueError, Storage, method="invalid_method")

        # Test invalid batch size
        self.assertRaises(ValueError, Storage, method="batch", batch_size=0)
        self.assertRaises(ValueError, Storage, method="batch", batch_size=-5)

    def test_iterate_full_method(self):
        """Test full iteration method"""
        storage = Storage(method="full", batch_size=10)
        X = np.array([[1, 2], [3, 4]])

        # Verify batch_size is changing
        self.assertEqual(storage.batch_size, 10)

        # Test iteration
        iterator = storage(X)
        result = list(iterator)
        self.assertEqual(len(result), 1)
        np.testing.assert_equal(result[0], X)

        # Verify batch_size is changing
        self.assertEqual(storage.batch_size, 2)

    def test_iterate_batch_method(self):
        """Test batch iteration method"""
        storage = Storage(method="batch", batch_size=1)
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Test iteration
        iterator = storage(X)
        result = list(iterator)
        self.assertEqual(len(result), 3)
        np.testing.assert_equal(result[0], np.array([[1, 2]]))
        np.testing.assert_equal(result[1], np.array([[3, 4]]))
        np.testing.assert_equal(result[2], np.array([[5, 6]]))
