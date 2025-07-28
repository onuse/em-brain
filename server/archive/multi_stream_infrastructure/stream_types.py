#!/usr/bin/env python3
"""
Shared Types for Vector Stream Processing

Contains common enums and types used across multiple stream processing modules
to avoid circular import dependencies.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any


class StreamType(Enum):
    """Types of processing streams in the brain."""
    SENSORY = "sensory"
    MOTOR = "motor"
    TEMPORAL = "temporal"
    CONFIDENCE = "confidence"
    ATTENTION = "attention"


class ConstraintType(Enum):
    """Types of constraints that can propagate between streams."""
    PROCESSING_LOAD = "processing_load"        # High computational demand
    RESOURCE_SCARCITY = "resource_scarcity"    # Limited resources available
    URGENCY_SIGNAL = "urgency_signal"          # Time-critical processing needed
    INTERFERENCE = "interference"               # Conflicting pattern activations
    COHERENCE_PRESSURE = "coherence_pressure"  # Need for consistent outputs
    ENERGY_DEPLETION = "energy_depletion"      # Low energy reserves


@dataclass
class StreamState:
    """Current state of a processing stream."""
    stream_type: StreamType
    last_update_time: float
    processing_phase: str
    active_patterns: List[int] = None
    activation_strength: float = 0.0
    processing_budget_remaining: float = 0.0
    coordination_signals: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.active_patterns is None:
            self.active_patterns = []
        if self.coordination_signals is None:
            self.coordination_signals = {}