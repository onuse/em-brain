"""
Compatibility shim for legacy imports
Redirects to PureFieldBrain - the only brain we need
"""

from .pure_field_brain import PureFieldBrain

# Legacy non-brain classes
class FieldDimension:
    """Legacy class - no longer used"""
    def __init__(self, *args, **kwargs):
        pass

class FieldDynamicsFamily:
    """Legacy class - no longer used"""
    def __init__(self, *args, **kwargs):
        pass