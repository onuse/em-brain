"""
Error Handling Utilities

Provides consistent error handling and validation for the field brain system.
"""

import torch
from typing import Any, Optional, Union, Tuple, List
from functools import wraps
import logging

# Set up logging
logger = logging.getLogger(__name__)


class BrainError(Exception):
    """Base exception for brain-related errors."""
    pass


class TensorShapeError(BrainError):
    """Raised when tensor shapes don't match expectations."""
    pass


class ConfigurationError(BrainError):
    """Raised when configuration is invalid."""
    pass


class AdapterError(BrainError):
    """Raised when adapter conversion fails."""
    pass


def validate_tensor_shape(tensor: torch.Tensor, 
                         expected_shape: Union[Tuple[int, ...], List[int]],
                         name: str = "tensor") -> None:
    """
    Validate that a tensor has the expected shape.
    
    Args:
        tensor: Tensor to validate
        expected_shape: Expected shape (use -1 for any size in that dimension)
        name: Name for error messages
        
    Raises:
        TensorShapeError: If shape doesn't match
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    if len(tensor.shape) != len(expected_shape):
        raise TensorShapeError(
            f"{name} has wrong number of dimensions: "
            f"expected {len(expected_shape)}, got {len(tensor.shape)}"
        )
    
    for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
        if expected != -1 and actual != expected:
            raise TensorShapeError(
                f"{name} dimension {i} mismatch: "
                f"expected {expected}, got {actual}"
            )


def validate_tensor_device(tensor: torch.Tensor,
                          expected_device: torch.device,
                          name: str = "tensor") -> torch.Tensor:
    """
    Validate and optionally move tensor to expected device.
    
    Args:
        tensor: Tensor to validate
        expected_device: Expected device
        name: Name for error messages
        
    Returns:
        Tensor on the correct device
    """
    if tensor.device != expected_device:
        logger.debug(f"Moving {name} from {tensor.device} to {expected_device}")
        return tensor.to(expected_device)
    return tensor


def safe_tensor_op(func):
    """
    Decorator for safe tensor operations with error handling.
    
    Catches common tensor operation errors and provides helpful messages.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"GPU out of memory in {func.__name__}")
                raise BrainError(
                    f"GPU out of memory. Try reducing batch size or field resolution."
                ) from e
            elif "dimension" in str(e) or "size" in str(e):
                logger.error(f"Tensor dimension error in {func.__name__}: {e}")
                raise TensorShapeError(
                    f"Tensor operation failed in {func.__name__}: {e}"
                ) from e
            else:
                logger.error(f"Runtime error in {func.__name__}: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    
    return wrapper


def validate_list_input(input_list: List[float],
                       expected_length: int,
                       name: str = "input",
                       min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> List[float]:
    """
    Validate list input for sensory/motor values.
    
    Args:
        input_list: List to validate
        expected_length: Expected length
        name: Name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Validated list
        
    Raises:
        ValueError: If validation fails
    """
    if not isinstance(input_list, list):
        raise TypeError(f"{name} must be a list, got {type(input_list)}")
    
    if len(input_list) != expected_length:
        raise ValueError(
            f"{name} has wrong length: expected {expected_length}, got {len(input_list)}"
        )
    
    # Validate values
    for i, val in enumerate(input_list):
        if not isinstance(val, (int, float)):
            raise TypeError(f"{name}[{i}] must be numeric, got {type(val)}")
        
        if min_val is not None and val < min_val:
            raise ValueError(f"{name}[{i}] = {val} is below minimum {min_val}")
        
        if max_val is not None and val > max_val:
            raise ValueError(f"{name}[{i}] = {val} is above maximum {max_val}")
    
    return input_list


def safe_divide(numerator: Union[torch.Tensor, float],
                denominator: Union[torch.Tensor, float],
                eps: float = 1e-8) -> Union[torch.Tensor, float]:
    """
    Safe division with zero protection.
    
    Args:
        numerator: Numerator
        denominator: Denominator  
        eps: Small value to prevent division by zero
        
    Returns:
        Result of division
    """
    if isinstance(denominator, torch.Tensor):
        return numerator / (denominator + eps)
    else:
        return numerator / (denominator + eps if abs(denominator) < eps else denominator)


def validate_config_param(config: dict,
                         param_name: str,
                         expected_type: type,
                         default: Any = None,
                         min_val: Optional[float] = None,
                         max_val: Optional[float] = None) -> Any:
    """
    Validate and extract configuration parameter.
    
    Args:
        config: Configuration dictionary
        param_name: Parameter name
        expected_type: Expected type
        default: Default value if not present
        min_val: Minimum value (for numeric types)
        max_val: Maximum value (for numeric types)
        
    Returns:
        Validated parameter value
    """
    value = config.get(param_name, default)
    
    if value is None:
        raise ConfigurationError(f"Required parameter '{param_name}' not found")
    
    if not isinstance(value, expected_type):
        raise ConfigurationError(
            f"Parameter '{param_name}' must be {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    
    if expected_type in (int, float):
        if min_val is not None and value < min_val:
            raise ConfigurationError(
                f"Parameter '{param_name}' = {value} is below minimum {min_val}"
            )
        if max_val is not None and value > max_val:
            raise ConfigurationError(
                f"Parameter '{param_name}' = {value} is above maximum {max_val}"
            )
    
    return value


class ErrorContext:
    """
    Context manager for better error messages.
    
    Usage:
        with ErrorContext("processing sensory input"):
            # code that might fail
    """
    
    def __init__(self, operation: str):
        self.operation = operation
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Error while {self.operation}: {exc_val}")
            # Re-raise with context
            if not isinstance(exc_val, BrainError):
                raise BrainError(f"Failed while {self.operation}: {exc_val}") from exc_val
        return False