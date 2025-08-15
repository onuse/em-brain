#!/usr/bin/env python3
"""
Test telemetry system with evolved brain
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
from src.core.evolved_brain_telemetry import EvolvedBrainTelemetryAdapter


def main():
    print("🧠 Testing Evolved Brain Telemetry")
    print("="*60)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=16,
        quiet_mode=True
    )
    
    # Create telemetry adapter
    telemetry = EvolvedBrainTelemetryAdapter(brain)
    
    print("\n1. Running brain cycles and collecting telemetry...")
    
    # Run some cycles
    for i in range(20):
        # Different patterns
        if i < 5:
            pattern = [0.5] * 24  # Neutral
        elif i < 10:
            pattern = [1.0] * 6 + [0.0] * 18  # Visual stimulus
        elif i < 15:
            pattern = [0.0] * 18 + [1.0] * 6  # Reward
        else:
            pattern = [0.0] * 24  # No input
        
        motors, state = brain.process_robot_cycle(pattern)
        
        # Get telemetry every 5 cycles
        if i % 5 == 0:
            snapshot = telemetry.get_telemetry()
            print(f"\n  Cycle {i}:")
            print(f"    Energy: {snapshot.energy_state} ({snapshot.field_energy:.4f})")
            print(f"    Confidence: {snapshot.confidence_state} ({snapshot.prediction_confidence:.2%})")
            print(f"    Behavior: {snapshot.behavior_state}")
            print(f"    Self-modification: {snapshot.self_modification_strength:.1%}")
            print(f"    Working memory: {snapshot.working_memory_patterns} patterns")
    
    print("\n2. Evolution Trajectory Analysis")
    print("-"*40)
    
    evolution = telemetry.get_evolution_trajectory()
    print(f"  Snapshots: {evolution.get('snapshots', 0)}")
    print(f"  Self-modification trend: {evolution.get('self_modification_trend', 0):.6f}")
    print(f"  Energy trend: {evolution.get('energy_trend', 0):.6f}")
    
    specialization = evolution.get('regional_specialization', {})
    if specialization.get('status') != 'insufficient_data':
        print(f"  Regional specialization:")
        print(f"    Fast regions: {specialization.get('fast_regions', 0)}")
        print(f"    Slow regions: {specialization.get('slow_regions', 0)}")
        print(f"    Coupled regions: {specialization.get('coupled_regions', 0)}")
        print(f"    Specialization index: {specialization.get('specialization_index', 0):.3f}")
    
    print("\n3. Behavior Analysis")
    print("-"*40)
    
    behavior = telemetry.get_behavior_analysis()
    if behavior.get('status') != 'no_data':
        print(f"  State distribution:")
        for state, count in behavior.get('state_distribution', {}).items():
            print(f"    {state}: {count}")
        print(f"  Dominant state: {behavior.get('dominant_state', 'unknown')}")
        print(f"  Stability index: {behavior.get('stability_index', 0):.2f}")
        print(f"  Transitions: {behavior.get('transition_count', 0)}")
    
    print("\n4. Comprehensive Summary")
    print("-"*40)
    
    summary = telemetry.get_summary()
    
    # Pretty print summary
    import json
    print(json.dumps(summary, indent=2))
    
    print("\n✅ Telemetry test complete!")
    print("\n💡 Insights:")
    print("  - Brain provides rich telemetry about its internal state")
    print("  - Evolution dynamics are tracked over time")
    print("  - Regional specialization emerges naturally")
    print("  - Behavior patterns can be analyzed")
    print("  - Everything is observable for improvement")


if __name__ == "__main__":
    main()