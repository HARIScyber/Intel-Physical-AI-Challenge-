"""NumPy and complex object JSON serialization utilities.

Provides safe serialization for trajectory data containing NumPy objects,
complex nested structures, and other non-JSON-serializable types.
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np


class JSONSerializationError(TypeError):
    """Raised when JSON serialization encounters an unsupported object."""
    pass


def to_jsonable(value: Any, path: str = "root") -> Any:
    """Recursively convert NumPy and other objects to JSON-serializable types.
    
    Args:
        value: Value to convert
        path: Dot-separated path for error reporting
        
    Returns:
        JSON-serializable version of the value
        
    Raises:
        JSONSerializationError: If value cannot be converted to JSON-serializable
    """
    try:
        if isinstance(value, (np.ndarray, np.generic)):
            return _convert_numpy_type(value, path)
        
        elif isinstance(value, dict):
            return {k: to_jsonable(v, f"{path}.{k}") for k, v in value.items()}
        
        elif isinstance(value, (list, tuple)):
            return [to_jsonable(item, f"{path}[{i}]") for i, item in enumerate(value)]
        
        elif isinstance(value, (str, int, float, bool)) or value is None:
            return value
        
        elif dataclasses.is_dataclass(value):
            return _convert_dataclass(value, path)
        
        else:
            raise JSONSerializationError(
                f"Object type {type(value).__name__} at path '{path}' is not JSON serializable"
            )
            
    except Exception as e:
        if isinstance(e, JSONSerializationError):
            raise
        raise JSONSerializationError(f"Error converting value at path '{path}': {e}") from e


def _convert_numpy_type(value: Any, path: str) -> Any:
    """Convert NumPy types to Python native types."""
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            # Handle object arrays by converting each element
            return [to_jsonable(item, f"{path}[{i}]") for i, item in enumerate(value)]
        return value.tolist()
    
    elif isinstance(value, np.integer):
        return int(value)
    
    elif isinstance(value, np.floating):
        return float(value)
    
    elif isinstance(value, np.bool_):
        return bool(value)
    
    elif isinstance(value, np.generic):
        # Handle other numpy scalar types
        return value.item()
    
    return value


def _convert_dataclass(obj: Any, path: str) -> dict[str, Any]:
    """Convert dataclass to dict using asdict."""
    try:
        return {k: to_jsonable(v, f"{path}.{k}") for k, v in dataclasses.asdict(obj).items()}
    except Exception as e:
        raise JSONSerializationError(
            f"Error converting dataclass {type(obj).__name__} at path '{path}': {e}"
        ) from e


def safe_json_dumps(obj: Any, indent: int = 2, **kwargs) -> str:
    """Safely dump Python object to JSON string with NumPy support.
    
    Args:
        obj: Python object to serialize
        indent: JSON indentation level
        **kwargs: Additional arguments to json.dumps
        
    Returns:
        JSON string
        
    Raises:
        JSONSerializationError: If serialization fails
    """
    try:
        jsonable_obj = to_jsonable(obj, "root")
        return json.dumps(jsonable_obj, indent=indent, **kwargs)
    except Exception as e:
        if isinstance(e, JSONSerializationError):
            raise
        raise JSONSerializationError(f"JSON serialization failed: {e}") from e


def test_numpy_json_serialization():
    """Unit tests for NumPy JSON serialization."""
    # Test ndarray
    ndarray = np.array([1, 2, 3], dtype=np.int32)
    result = to_jsonable(ndarray, "test.ndarray")
    assert result == [1, 2, 3], f"Expected [1, 2, 3], got {result}"
    
    # Test numpy float
    numpy_float = np.float64(3.14)
    result = to_jsonable(numpy_float, "test.float")
    assert result == 3.14, f"Expected 3.14, got {result}"
    
    # Test numpy int
    numpy_int = np.int32(42)
    result = to_jsonable(numpy_int, "test.int")
    assert result == 42, f"Expected 42, got {result}"
    
    # Test nested dict with numpy
    nested = {"arr": np.array([1, 2]), "value": np.float32(2.5)}
    result = to_jsonable(nested, "test.nested")
    expected = {"arr": [1, 2], "value": 2.5}
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test nested list with numpy
    nested_list = [np.array([1, 2]), np.int64(100)]
    result = to_jsonable(nested_list, "test.nested_list")
    expected = [[1, 2], 100]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test tuple
    tuple_val = (np.array([1, 2]), 3)
    result = to_jsonable(tuple_val, "test.tuple")
    expected = [[1, 2], 3]
    assert result == expected, f"Expected {expected}, got {result}"
    
    # Test boolean
    numpy_bool = np.bool_(True)
    result = to_jsonable(numpy_bool, "test.bool")
    assert result is True, f"Expected True, got {result}"
    
    # Test normal Python values
    normal_values = {
        "string": "hello",
        "int": 42,
        "float": 3.14,
        "bool": True,
        "none": None,
        "list": [1, 2, 3],
        "dict": {"a": 1, "b": 2}
    }
    result = to_jsonable(normal_values, "test.normal")
    assert result == normal_values, f"Expected {normal_values}, got {result}"
    
    # Test dataclass
    @dataclasses.dataclass
    class TestDataclass:
        name: str
        value: int
        data: list[float]
    
    dataclass_obj = TestDataclass("test", 42, [1.0, 2.5, 3.14])
    result = to_jsonable(dataclass_obj, "test.dataclass")
    expected = {"name": "test", "value": 42, "data": [1.0, 2.5, 3.14]}
    assert result == expected, f"Expected {expected}, got {result}"
    
    print("All JSON serialization tests passed!")


if __name__ == "__main__":
    test_numpy_json_serialization()