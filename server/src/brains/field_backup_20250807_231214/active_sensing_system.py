"""
Compatibility shim for legacy imports
Redirects to PureFieldBrain - the only brain we need
"""

from .pure_field_brain import PureFieldBrain

# Legacy uncertainty tracking class
class UncertaintyMap:
    """Legacy uncertainty tracking - no longer used"""
    def __init__(self, *args, **kwargs):
        self.data = {}
    def update(self, *args, **kwargs):
        pass