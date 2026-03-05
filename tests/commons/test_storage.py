"""
Unit tests for Storage class
"""

import numpy as np
import pytest

from cristal.commons.storage import IMPLEMENTED_STORAGE, Storage


class TestStorage:
    """Test the Storage class functionality"""

    def test_storage_initialization(self):
        """Test Storage initialization with valid parameters"""
        # Test valid storage methods
        for method in IMPLEMENTED_STORAGE.__args__:
            storage = Storage(method=method, batch_size=10)
            assert storage.method == method
            assert storage.batch_size == 10

        # Test invalid method
        with pytest.raises(AssertionError):
            Storage(method="invalid_method", batch_size=10)

        # Test invalid batch size
        with pytest.raises(AssertionError):
            Storage(method="full", batch_size=0)

    def test_iterate_full_method(self):
        """Test full iteration method"""
        storage = Storage(method="full", batch_size=10)
        X = np.array([[1, 2], [3, 4]])

        # Test iteration
        iterator = storage.iterate(X)
        result = list(iterator)
        assert len(result) == 1
        assert np.array_equal(result[0], X)

    def test_iterate_batch_method(self):
        """Test batch iteration method"""
        storage = Storage(method="batch", batch_size=1)
        X = np.array([[1, 2], [3, 4], [5, 6]])

        # Test iteration
        iterator = storage.iterate(X)
        result = list(iterator)
        assert len(result) == 3
        assert np.array_equal(result[0], np.array([[1, 2]]))
        assert np.array_equal(result[1], np.array([[3, 4]]))
        assert np.array_equal(result[2], np.array([[5, 6]]))
