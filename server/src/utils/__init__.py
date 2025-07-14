"""
Shared Utilities

Common functionality for minimal brain:
- Brain logging for emergence analysis
- Configuration management (future)
- Performance metrics (future)
"""

# Only import what exists
from .brain_logger import BrainLogger
from .adaptive_trigger import AdaptiveTrigger

__all__ = ["BrainLogger", "AdaptiveTrigger"]