"""
Mock Hardware Implementations

Mock drivers that simulate hardware behavior for testing and development
without requiring actual robot hardware.
"""

from .mock_vocal_driver import MockVocalDriver

__all__ = [
    'MockVocalDriver'
]