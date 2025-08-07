"""
Compatibility shim for legacy imports
Redirects to PureFieldBrain - the only brain we need
"""

from .pure_field_brain import PureFieldBrain

# Legacy exports
FieldStrategicPlanner = PureFieldBrain

class StrategicPattern:
    """Legacy class - no longer used"""
    def __init__(self, *args, **kwargs):
        pass