#!/usr/bin/env python3
"""
Field Brain Type Definitions

Shared type definitions for field brain implementations to avoid circular imports.
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum


class FieldDynamicsFamily(Enum):
    """Families of field dynamics that organize dimensions."""
    SPATIAL = "spatial"              # Position, orientation, scale
    OSCILLATORY = "oscillatory"     # Frequencies, rhythms, periods
    FLOW = "flow"                   # Gradients, momentum, direction
    TOPOLOGY = "topology"           # Stable configurations, boundaries
    ENERGY = "energy"               # Intensity, activation, depletion
    COUPLING = "coupling"           # Relationships, correlations, binding
    EMERGENCE = "emergence"         # Novelty, creativity, phase transitions


@dataclass
class FieldDimension:
    """A single dimension in the unified field."""
    name: str
    family: FieldDynamicsFamily
    index: int
    min_value: float = -1.0
    max_value: float = 1.0
    default_value: float = 0.0
    description: str = ""


@dataclass
class StreamCapabilities:
    """Describes the capabilities of connected streams."""
    input_dimensions: int
    output_dimensions: int
    input_labels: List[str] = None
    output_labels: List[str] = None
    input_ranges: List[Tuple[float, float]] = None
    output_ranges: List[Tuple[float, float]] = None
    update_frequency_hz: Optional[float] = None
    latency_ms: Optional[float] = None

    def __post_init__(self):
        if self.input_labels is None:
            self.input_labels = []
        if self.output_labels is None:
            self.output_labels = []
        if self.input_ranges is None:
            self.input_ranges = []
        if self.output_ranges is None:
            self.output_ranges = []


@dataclass
class UnifiedFieldExperience:
    """A unified field experience - pattern-based, coordinate-free."""
    timestamp: float
    raw_input_stream: torch.Tensor       # Original input stream
    field_intensity: float               # Overall field activation strength
    dynamics_family_activations: Dict[FieldDynamicsFamily, float]


@dataclass
class FieldNativeAction:
    """A field-native action - pattern-based, coordinate-free."""
    timestamp: float
    output_stream: torch.Tensor          # Generic output stream
    confidence: float                    # Action confidence from field stability
    dynamics_family_contributions: Dict[FieldDynamicsFamily, float]