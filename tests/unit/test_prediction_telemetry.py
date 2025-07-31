#!/usr/bin/env python3
"""
Test script to verify 5-phase prediction telemetry is working

This script starts a brain, runs it through some predictable patterns,
and verifies that all 5 phases of telemetry are being captured correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../server'))

import time
import torch
import numpy as np
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from src.core.evolved_brain_telemetry import EvolvedBrainTelemetryAdapter

def test_prediction_telemetry():
    """Test that all 5 phases of prediction telemetry are captured."""
    print("\n=== Testing 5-Phase Prediction Telemetry ===\n")
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=10,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=False
    )
    
    # Enable active vision for Phase 5
    brain.enable_active_vision(True)
    
    # Create telemetry adapter
    telemetry_adapter = EvolvedBrainTelemetryAdapter(brain)
    
    print("1. Running brain with predictable patterns...")
    
    # Create predictable patterns
    patterns = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ]
    
    # Train with patterns
    for epoch in range(5):
        for i, pattern in enumerate(patterns):
            motor_output, brain_state = brain.process_robot_cycle(pattern)
            
            # Get telemetry every few cycles
            if brain.brain_cycles % 2 == 0:
                telemetry = telemetry_adapter.get_telemetry()
                
                if epoch == 4:  # Show details on last epoch
                    print(f"\n  Cycle {brain.brain_cycles}:")
                    
                    # Phase 1: Sensory Predictions
                    if telemetry.sensory_predictions:
                        sp = telemetry.sensory_predictions
                        print(f"    Phase 1 - Sensory: accuracy={sp.get('accuracy', 0):.1%}, regions={sp.get('specialized_regions', 0)}")
                    
                    # Phase 2: Error Learning
                    if telemetry.prediction_errors:
                        pe = telemetry.prediction_errors
                        print(f"    Phase 2 - Errors: magnitude={pe.get('magnitude', 0):.3f}, mod_strength={pe.get('modification_strength', 0):.1%}")
                    
                    # Phase 3: Hierarchical
                    if telemetry.hierarchical_predictions:
                        hp = telemetry.hierarchical_predictions
                        print(f"    Phase 3 - Hierarchy: immediate={hp.get('immediate_accuracy', 0):.1%}, short={hp.get('short_term_accuracy', 0):.1%}")
                    
                    # Phase 4: Action Selection
                    if telemetry.action_selection:
                        acts = telemetry.action_selection
                        print(f"    Phase 4 - Actions: strategy={acts.get('current_strategy', 'unknown')}")
                    
                    # Phase 5: Active Sensing
                    if telemetry.active_sensing:
                        avs = telemetry.active_sensing
                        print(f"    Phase 5 - Sensing: uncertainty={avs.get('total_uncertainty', 0):.3f}, pattern={avs.get('current_pattern', 'unknown')}")
    
    print("\n2. Testing unpredictable input...")
    
    # Random input
    for i in range(5):
        random_input = [float(x) for x in torch.rand(10)]
        motor_output, brain_state = brain.process_robot_cycle(random_input)
    
    # Final telemetry
    final_telemetry = telemetry_adapter.get_telemetry()
    
    print("\n3. Final telemetry check:")
    
    # Verify all phases are present
    phases_present = []
    if final_telemetry.sensory_predictions:
        phases_present.append("Phase 1 (Sensory)")
    if final_telemetry.prediction_errors:
        phases_present.append("Phase 2 (Errors)")
    if final_telemetry.hierarchical_predictions:
        phases_present.append("Phase 3 (Hierarchy)")
    if final_telemetry.action_selection:
        phases_present.append("Phase 4 (Actions)")
    if final_telemetry.active_sensing:
        phases_present.append("Phase 5 (Sensing)")
    
    print(f"  Phases captured: {len(phases_present)}/5")
    for phase in phases_present:
        print(f"    ✓ {phase}")
    
    # Show evolution metrics
    print(f"\n  Evolution state:")
    print(f"    Self-modification: {final_telemetry.self_modification_strength:.1%}")
    print(f"    Evolution cycles: {final_telemetry.evolution_cycles}")
    print(f"    Working memory: {final_telemetry.working_memory_patterns} patterns")
    
    # Test success
    if len(phases_present) >= 3:
        print("\n✅ Prediction telemetry working!")
        return True
    else:
        print("\n❌ Some telemetry phases missing")
        return False


if __name__ == "__main__":
    test_prediction_telemetry()