"""
This module provides a decorator for type checking function arguments.
"""

from collections.abc import Callable
from functools import wraps
from inspect import signature
from typing import Any


def check_types(conditions: dict[str, Callable[[Any], bool]] | None = None) -> Callable:
    """Decorator to check types of function arguments.

    Parameters
    ----------
    conditions : dict[str, Callable[[Any], bool]] | None, optional
        Custom conditions for argument validation, by default None

    Returns
    -------
    Any
        The decorated function with type checks applied.

    Raises
    ------
    TypeError
        If an argument does not match its expected type or fails a custom condition.

    Example
    -------
    >>> @check_types({"a": lambda x: x > 0})
    ... def my_function(a: int, b: str | None = None) -> None:
    ...     pass

    This will raise a TypeError if `a` is not an int greater than 0 or if `b` is not a str or None.

    >>> @check_types()
    ... def another_function(x: float) -> None:
    ...     pass

    This will raise a TypeError if `x` is not float.
    """
    if conditions is None:
        conditions = {}

    def decorator(func: Callable) -> Callable:
        # Obtain the type annotations of the function
        type_hints = func.__annotations__
        sig = signature(func)

        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Bind the passed arguments to the function's signature
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Check each argument against its type annotation
            for param_name, param_value in bound_args.arguments.items():
                if param_name in type_hints:
                    expected_type = type_hints[param_name]
                    # Type check
                    if not isinstance(param_value, expected_type):
                        raise TypeError(f"Argument '{param_name}' must be of type {expected_type}")

                # Check custom conditions
                if param_name in conditions:
                    if not conditions[param_name](param_value):
                        condition_name = getattr(conditions[param_name], "__name__", None)
                        condition_str = f": {condition_name}({param_value})" if condition_name else ""
                        raise TypeError(f"Argument '{param_name}' does not satisfy the custom condition{condition_str}")
            return func(*args, **kwargs)

        return wrapper

    return decorator


def check_multiple_conditions(value: Any, conditions: list[Callable[[Any], bool]]) -> bool:
    """Check if the value satisfies all conditions in the list."""
    return all(condition(value) for condition in conditions)


def check_any_condition(value: Any, conditions: list[Callable[[Any], bool]]) -> bool:
    """Check if the value satisfies any condition in the list."""
    return any(condition(value) for condition in conditions)


def positive_integer(value: Any) -> bool:
    """Check if the value is a positive integer."""
    return isinstance(value, int) and value > 0


def check_all_of_type(value: Any, expected_type: type | tuple[type, ...]) -> bool:
    """Check if all elements in the value are of the expected type."""
    if isinstance(value, list):
        return all(isinstance(item, expected_type) for item in value)
    return isinstance(value, expected_type)


def check_none(value: Any) -> bool:
    """Check if the value is None."""
    return value is None


def check_all_int_or_float(value: Any) -> bool:
    """Check if all elements in the value are either int or float."""
    return check_all_of_type(value, (int, float))


def check_in_list(value: Any, valid_values: list | dict) -> bool:
    """Check if the value is in the list of valid values."""
    return value in valid_values
