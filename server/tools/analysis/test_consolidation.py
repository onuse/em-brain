#!/usr/bin/env python3
"""
Test Consolidation System

Demonstrates memory consolidation and dream states during idle periods.
"""

import sys
import os
import time
import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain


def test_consolidation():
    """Test the consolidation system."""
    print("🧠 Testing Consolidation System")
    print("=" * 60)
    
    # Create brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        spatial_resolution=32,
        quiet_mode=False,
        use_optimized=True
    )
    
    print("\n1. Training Phase - Creating patterns to consolidate")
    print("-" * 40)
    
    # Run some training cycles to create patterns
    training_patterns = [
        # Light-seeking pattern
        [1.0] * 12 + [0.0] * 12,  # Strong light on left
        [0.0] * 12 + [1.0] * 12,  # Strong light on right
        # Obstacle avoidance pattern  
        [0.0] * 6 + [1.0] * 6 + [0.0] * 12,  # Obstacle in front
        # Reward patterns
        [0.5] * 23 + [1.0],  # Positive reward
        [0.5] * 23 + [-1.0], # Negative reward
    ]
    
    for i, pattern in enumerate(training_patterns):
        print(f"   Training pattern {i+1}: ", end='')
        motors, state = brain.process_robot_cycle(pattern)
        print(f"Energy={state['field_energy']:.3f}")
        time.sleep(0.1)
    
    # Check initial state
    initial_state = brain.field_dynamics.compute_field_state(brain.unified_field)
    initial_energy = initial_state['combined_energy']
    print(f"\nInitial field energy: {initial_energy:.3f}")
    
    print("\n2. Consolidation Phase - Memory consolidation & dreams")
    print("-" * 40)
    
    # Start consolidation (simulating sleep/idle period)
    metrics = brain.start_idle_consolidation(duration_seconds=10.0)
    
    # Check post-consolidation state
    final_state = brain.field_dynamics.compute_field_state(brain.unified_field)
    final_energy = final_state['combined_energy']
    print(f"\nPost-consolidation field energy: {final_energy:.3f}")
    print(f"Energy change: {final_energy - initial_energy:+.3f}")
    
    print("\n3. Consolidation Metrics")
    print("-" * 40)
    print(f"Cycles processed: {metrics.cycles_processed}")
    print(f"Patterns strengthened: {metrics.patterns_strengthened}")
    print(f"Patterns pruned: {metrics.patterns_pruned}")
    print(f"Dream sequences: {metrics.dream_sequences}")
    print(f"Topology refined: {metrics.topology_refined}")
    print(f"Overall benefit: {metrics.consolidation_benefit:.3f}")
    
    print("\n4. Testing Recall - Do consolidated patterns persist?")
    print("-" * 40)
    
    # Test with similar patterns to see if learning was consolidated
    test_patterns = [
        [0.8] * 12 + [0.2] * 12,  # Similar to left light
        [0.2] * 12 + [0.8] * 12,  # Similar to right light
    ]
    
    for i, pattern in enumerate(test_patterns):
        print(f"   Test pattern {i+1}: ", end='')
        motors, state = brain.process_robot_cycle(pattern)
        print(f"Motor response: {[f'{m:.2f}' for m in motors[:3]]}")
    
    # Test maintenance scheduler integration
    print("\n5. Testing Maintenance Integration")
    print("-" * 40)
    brain.perform_maintenance()
    
    print("\n✅ Consolidation system test complete!")


def demonstrate_dream_states():
    """Demonstrate dream state generation."""
    print("\n\n🌙 Dream State Demonstration")
    print("=" * 60)
    
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=True,
        use_optimized=True
    )
    
    # Create diverse experiences
    print("Creating diverse experiences...")
    experiences = [
        # Day patterns (exploration)
        [np.sin(i * 0.1)] * 24 for i in range(10)
    ] + [
        # Night patterns (different dynamics)
        [np.cos(i * 0.2)] * 24 for i in range(10)
    ]
    
    for exp in experiences:
        brain.process_robot_cycle(exp)
    
    # Monitor field during dreaming
    print("\nMonitoring dream states...")
    print("Time | Energy | Variance | Dream Activity")
    print("-" * 50)
    
    # Start consolidation with monitoring
    brain.consolidation_system.start_consolidation(brain)
    start_time = time.time()
    
    for i in range(20):
        # Run brief consolidation
        brain.consolidation_system.consolidate_memories(brain, 0.5)
        
        # Check field state
        stats = brain.field_dynamics.compute_field_state(brain.unified_field)
        energy = stats['combined_energy']
        
        # Calculate variance as proxy for dream activity
        field_var = float(torch.var(brain.unified_field))
        
        elapsed = time.time() - start_time
        print(f"{elapsed:4.1f}s | {energy:6.3f} | {field_var:8.6f} | {'*' * int(field_var * 1000)}")
    
    print("\nDream metrics:")
    print(f"Total dreams: {brain.consolidation_system.metrics.dream_sequences}")
    print(f"Pattern mixing: Active")


if __name__ == "__main__":
    test_consolidation()
    demonstrate_dream_states()