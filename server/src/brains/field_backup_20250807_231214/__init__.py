"""
Field-native brain implementation
Only PureFieldBrain remains - all complexity eliminated
"""

from .pure_field_brain import PureFieldBrain, SCALE_CONFIGS

__all__ = ['PureFieldBrain', 'SCALE_CONFIGS']

# Legacy compatibility
# These imports exist only for backward compatibility with tests
try:
    from .simplified_unified_brain import SimplifiedUnifiedBrain
except ImportError:
    SimplifiedUnifiedBrain = PureFieldBrain

try:
    from .unified_field_brain import UnifiedFieldBrain
except ImportError:
    UnifiedFieldBrain = PureFieldBrain

try:
    from .field_strategic_planner import FieldStrategicPlanner, StrategicPattern
except ImportError:
    FieldStrategicPlanner = PureFieldBrain
    class StrategicPattern:
        def __init__(self, *args, **kwargs):
            pass

try:
    from .evolved_field_dynamics import EvolvedFieldDynamics
except ImportError:
    EvolvedFieldDynamics = PureFieldBrain
