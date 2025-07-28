# Archive Summary

This archive contains files that were removed from the active codebase but preserved for historical reference.

## Phase A Prototypes (phase_a_prototypes/)
Early evolutionary stages of the field dynamics system:
- **analog_field_dynamics.py** - Phase A1: 2D continuous field prototype
- **multiscale_field_dynamics.py** - Phase A2: 3D field with scale dimension
- **temporal_field_dynamics.py** - Phase A3: 4D field with time dimension
- **constraint_field_dynamics.py** - Phase A5: Early constraint integration

These were superseded by the current N-dimensional unified field implementation.

## Temporary Fixes (temporary_fixes/)
Experimental fixes that were integrated into the main codebase:
- **gradient_fix.py** - Gradient extraction experiments
- **field_dimension_fix.py** - Dimension utilization experiments

Functionality has been properly integrated into the current brain.

## Multi-Stream Infrastructure (multi_stream_infrastructure/)
Infrastructure for coordinating multiple brain streams:
- **constraint_propagation_system.py** - Cross-stream constraint propagation
- **emergent_attention_allocation.py** - Attention allocation between streams
- **adaptive_constraint_thresholds.py** - Dynamic constraint thresholds
- **constraint_pattern_inhibition.py** - Pattern selection via inhibition
- **shared_brain_state.py** - Shared state for parallel streams
- **stream_types.py** - Stream type definitions

Not used in current single-brain architecture but could be valuable for future multi-robot coordination.

## Unused Enhancements (unused_enhancements/)
Features that were explored but not integrated:
- **hierarchical_processing.py** - Multi-resolution hierarchical processing
- **attention_guided.py** - Alternative attention implementation
- **attention_super_resolution.py** - High-resolution attention regions
- **robot_interface.py** - Old field-native robot interface (Phase B2)
- **brain_maintenance_interface.py** - Maintenance scheduling interface
- **statistics_control.py** - Statistics collection control

These approaches were superseded by simpler, more elegant solutions in the unified field brain.

## Programmatic Attention (programmatic_attention/)
Programmatic solutions to organic problems:
- **integrated_attention.py** - IntegratedAttentionSystem that used coordinates
- **attention/** - Signal-based attention with modality detection, saliency maps
- **pattern_memory.py** - Memory system that depended on signal attention

These implemented explicit features (object tracking, modality detection) that should emerge organically from field dynamics. Superseded by PatternBasedAttention which uses no coordinates, just pattern salience.

## Brain Evolution (brain_evolution/)
Different stages of brain implementation:
- **dynamic_unified_brain.py** - Non-full version lacking all interesting features
- **simple_field_brain.py** - Simple test implementation
- **core_brain.py** - Original UnifiedFieldBrain (1500+ lines) with hardcoded dimensions

Consolidated to use only dynamic_unified_brain_full.py (renamed to dynamic_unified_brain.py) which has all features: constraints, spontaneous dynamics, blended reality, pattern-based systems, emergent navigation. The dynamic approach lets robots specify their capabilities rather than hardcoding assumptions.

## Unused Utilities (unused_utilities/)
- **field_logger.py** - Field logging utility, never used

## Archive Date
Archived on: 2025-01-28
Reason: Code cleanup after verifying features were either integrated or superseded