"""
Shared Utilities

Common functionality for minimal brain:
- Brain logging for emergence analysis
- Configuration management (future)
- Performance metrics (future)
"""

# Only import what exists after vector stream migration
from .brain_logger import BrainLogger
from .cognitive_autopilot import CognitiveAutopilot
from .hardware_adaptation import get_hardware_adaptation

__all__ = ["BrainLogger", "CognitiveAutopilot", "get_hardware_adaptation"]