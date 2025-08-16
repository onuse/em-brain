"""
Simple Error Handling - Just what we actually need.

No 50+ error codes. Just simple, clear error handling.
"""


class BrainError(Exception):
    """Base exception for brain system."""
    pass


class ConnectionError(BrainError):
    """TCP connection issues."""
    pass


class ProcessingError(BrainError):
    """Brain processing issues."""
    pass


class ConfigError(BrainError):
    """Configuration issues."""
    pass


# That's it. 4 error types. Not 50.