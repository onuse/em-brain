"""
Compatibility shim for legacy imports
Redirects to PureFieldBrain - the only brain we need
"""

from .pure_field_brain import PureFieldBrain

# Legacy export
EvolvedFieldDynamics = PureFieldBrain