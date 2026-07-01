"""
Unit tests for BaseCommons class
"""

import unittest
from unittest.mock import Mock

from cristal.commons.base_commons import BaseCommons


class TestBaseCommons(unittest.TestCase):
    """Test the BaseCommons class functionality"""

    def test_bind_method(self):
        """Test binding dependencies to BaseCommons"""
        # Create a mock config with dependencies
        config = Mock()
        config.dependency1 = "value1"
        config.dependency2 = "value2"

        # Create a subclass of BaseCommons that requires dependencies
        class TestClass(BaseCommons):
            requires = ["dependency1", "dependency2"]

        obj = TestClass()

        # Check that _dependencies is initially None before binding
        self.assertIsNone(obj._dependencies)

        # Bind the configuration
        obj.bind(config)

        # Check that dependencies are correctly bound
        self.assertEqual(obj._dependencies["dependency1"], "value1")  # type: ignore
        self.assertEqual(obj._dependencies["dependency2"], "value2")  # type: ignore

    def test_bind_method_error(self):
        """Test binding dependencies to BaseCommons"""
        # Create a mock config with dependencies
        config = Mock(spec="dependency1")
        config.dependency1 = "value1"

        # Create a subclass of BaseCommons that requires dependencies
        class TestClass(BaseCommons):
            requires = ["dependency1", "dependency2"]

        obj = TestClass()

        # Check that _dependencies is initially None before binding
        self.assertIsNone(obj._dependencies)

        # Bind the configuration -> Raises error because dependency2 is missing
        self.assertRaises(ValueError, obj.bind, config)

    def test_getattr_method(self):
        """Test __getattr__ method for accessing bound dependencies"""
        # Create a mock config with dependencies
        config = Mock()
        config.test_attr = "test_value"

        # Create a subclass of BaseCommons that requires dependencies
        class TestClass(BaseCommons):
            requires = ["test_attr"]

        obj = TestClass()
        obj.bind(config)

        # Check that __getattr__ works correctly
        self.assertEqual(obj.test_attr, "test_value")

    def test_getattr_missing_attribute(self):
        """Test __getattr__ raises AttributeError for missing attributes"""

        class TestClass(BaseCommons):
            requires = []

        obj = TestClass()
        obj._dependencies = {}

        # Expect AttributeError
        self.assertRaises(AttributeError, obj.__getattr__, "non_existent_attribute")
