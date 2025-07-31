#!/usr/bin/env python3
"""Demo showing the brain's predictive intelligence capabilities."""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'server'))

import numpy as np
import time
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain

print("Predictive Intelligence Demo")
print("="*60)
print("This demo shows how the brain learns to predict patterns\n")

# Create brain with predictive features enabled
brain = SimplifiedUnifiedBrain(
    sensory_dim=8,  # Smaller for clearer demo
    motor_dim=3,
    spatial_resolution=32,
    quiet_mode=True
)

# Enable all phases
print("Enabling predictive intelligence features:")
brain.enable_hierarchical_prediction(True)
print("  ✓ Phase 3: Hierarchical prediction")
brain.enable_action_prediction(True)
print("  ✓ Phase 4: Action as prediction testing")
brain.enable_active_vision(True)
print("  ✓ Phase 5: Active sensing\n")

# 1. Pattern Learning Demo
print("1. PATTERN LEARNING")
print("-" * 40)
print("Teaching the brain a repeating pattern...")

pattern = [
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
]

# Train on pattern
for epoch in range(5):
    errors = []
    for i, sensory in enumerate(pattern):
        motor, state = brain.process_robot_cycle(sensory)
        
        # Check prediction error
        if hasattr(brain, '_last_prediction_error'):
            error = brain._last_prediction_error
            # Convert tensor to float if needed
            if hasattr(error, 'cpu'):
                error = float(error.mean().cpu())
            else:
                error = float(error)
            errors.append(error)
    
    if errors and epoch % 2 == 0:
        avg_error = np.mean(errors)
        print(f"  Epoch {epoch+1}: Average prediction error = {avg_error:.3f}")

# Test prediction
print("\nTesting prediction on partial input:")
test_input = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
motor, state = brain.process_robot_cycle(test_input)

if brain.predictive_field.last_prediction is not None:
    pred = brain.predictive_field.last_prediction
    print(f"  Input:      {[f'{x:.1f}' for x in test_input[:4]]}...")
    print(f"  Prediction: {[f'{x:.2f}' for x in pred.values[:4].tolist()]}...")
    print(f"  Confidence: {np.mean(pred.confidence):.3f}")

# 2. Action-Outcome Learning
print("\n\n2. ACTION-OUTCOME LEARNING")
print("-" * 40)
print("Teaching action-outcome relationships...")

# Simulate that certain actions lead to rewards
for i in range(20):
    # Random sensory input
    sensory = [0.5 + 0.1 * np.random.randn() for _ in range(8)]
    motor, state = brain.process_robot_cycle(sensory)
    
    # Reward if first motor output is positive
    if motor[0] > 0.5:
        sensory_with_reward = sensory + [0.8]  # Good reward
        brain.process_robot_cycle(sensory_with_reward)
        if i % 5 == 0:
            print(f"  Cycle {i}: Action {motor[0]:.2f} → Reward!")

# Check if brain learned the relationship
print("\nChecking learned behavior:")
action_stats = {"positive": 0, "negative": 0}
for i in range(10):
    sensory = [0.5] * 8
    motor, state = brain.process_robot_cycle(sensory)
    if motor[0] > 0:
        action_stats["positive"] += 1
    else:
        action_stats["negative"] += 1

print(f"  Actions: {action_stats['positive']} positive, {action_stats['negative']} negative")
print(f"  Learning: {'Success!' if action_stats['positive'] > 7 else 'Still learning...'}")

# 3. Active Sensing Demo
print("\n\n3. ACTIVE SENSING")
print("-" * 40)
print("Demonstrating uncertainty-driven attention...")

# Create high uncertainty by showing novel patterns
novel_patterns = [
    [np.random.rand() for _ in range(8)] for _ in range(5)
]

print("Showing novel patterns to increase uncertainty:")
uncertainties = []
attention_states = []

for pattern in novel_patterns:
    motor, state = brain.process_robot_cycle(pattern)
    
    # Get uncertainty
    uncertainty_map = brain.field_dynamics.generate_uncertainty_map(brain.unified_field)
    mean_uncertainty = float(uncertainty_map.mean().cpu())
    uncertainties.append(mean_uncertainty)
    
    # Get attention state
    if brain.active_vision:
        attention = brain.active_vision.current_attention_state
        attention_states.append(attention.get('pattern', 'unknown'))

print(f"  Mean uncertainty: {np.mean(uncertainties):.3f}")
print(f"  Attention patterns: {set(attention_states)}")

# 4. Field Evolution Stats
print("\n\n4. FIELD EVOLUTION")
print("-" * 40)
evolution_props = brain.field_dynamics.get_emergent_properties()

print(f"  Evolution cycles: {brain.field_dynamics.evolution_count}")
print(f"  Self-modification: {brain.field_dynamics.self_modification_strength:.3f}")
print(f"  Confidence: {evolution_props['smoothed_confidence']:.3f}")
print(f"  Information: {evolution_props['smoothed_information']:.3f}")
print(f"  Topology regions: {len(brain.topology_region_system.regions)}")

# Show specialization
if brain.field_dynamics.specialized_regions:
    print(f"\n  Specialized regions: {len(brain.field_dynamics.specialized_regions)}")
    for region_id, params in list(brain.field_dynamics.specialized_regions.items())[:3]:
        print(f"    Region {region_id}: plasticity={params['plasticity_rate']:.3f}")

print("\n" + "="*60)
print("Demo complete! The brain demonstrates:")
print("  • Pattern prediction through experience")
print("  • Action selection based on outcomes")
print("  • Active sensing following uncertainty")
print("  • Self-modifying field dynamics")