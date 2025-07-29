"""
Shared Utilities

Common functionality for minimal brain:
- Brain logging for emergence analysis
- Configuration management (future)
- Performance metrics (future)
"""

# Only import what exists after vector stream migration
from .brain_logger import BrainLogger

__all__ = ["BrainLogger"]