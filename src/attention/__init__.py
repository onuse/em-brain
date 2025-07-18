"""
Attention Systems - Unified Attention Architecture

This package contains all attention-related systems:
- Universal attention for any modality
- Cross-modal attention for object binding
"""

from .signal_attention import UniversalAttentionSystem, ModalityType
from .object_attention import CrossModalAttentionSystem, CrossModalObject, ObjectState

__all__ = [
    'UniversalAttentionSystem',
    'ModalityType', 
    'CrossModalAttentionSystem',
    'CrossModalObject',
    'ObjectState'
]