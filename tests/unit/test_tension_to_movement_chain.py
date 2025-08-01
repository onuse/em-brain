#!/usr/bin/env python3
"""Test the complete chain from tensions to movement."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
from server.src.brains.field.field_strategic_planner import FieldStrategicPlanner
from server.src.brains.field.unified_field_brain import UnifiedFieldBrain


def test_confidence_tension_creates_exploration_pattern():
    """Test that high confidence tension generates exploration patterns."""
    print("Testing confidence tension → exploration pattern...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    planner.simulation_horizon = 10  # Faster for testing
    
    # Create field with high confidence tension
    field = torch.zeros(field_shape, device=planner.device)
    field[:, :, :, :32] = 0.5  # Medium energy
    field[:, :, :, 58] = 0.1   # Low confidence → high confidence tension
    
    # Measure tensions
    tensions = planner._measure_field_tensions(field)
    print(f"\nTensions: {tensions}")
    assert tensions['confidence'] > 0.7, f"Expected high confidence tension, got {tensions['confidence']}"
    
    # Generate pattern for this tension
    pattern = planner._generate_tension_targeted_pattern(tensions)
    
    # Check pattern has structure
    pattern_variance = pattern.std().item()
    print(f"Pattern variance: {pattern_variance:.3f}")
    assert pattern_variance > 0.1, "Pattern too uniform"
    
    # Install pattern in field
    test_field = field.clone()
    test_field[:, :, :, 32:48] = pattern
    
    # Check if pattern creates gradients
    content_after = test_field[:, :, :, :32]
    
    # Calculate gradients
    x_grad = (content_after[-1, :, :].mean() - content_after[0, :, :].mean()).item()
    y_grad = (content_after[:, -1, :].mean() - content_after[:, 0, :].mean()).item()
    
    print(f"\nGradients after pattern installation:")
    print(f"  X gradient: {x_grad:.4f}")
    print(f"  Y gradient: {y_grad:.4f}")
    
    # We expect some gradient from confidence-resolving patterns
    total_gradient = abs(x_grad) + abs(y_grad)
    print(f"  Total gradient magnitude: {total_gradient:.4f}")
    
    return pattern, total_gradient


def test_pattern_influence_on_content():
    """Test how patterns in memory channels influence content channels."""
    print("\n\nTesting pattern influence on content channels...")
    
    # Create a simple gradient pattern
    field_shape = (8, 8, 8, 64)
    pattern = torch.zeros(8, 8, 8, 16)
    
    # Create strong gradient in pattern
    for i in range(8):
        pattern[i, :, :, :] = i / 4.0 - 1.0  # -1 to 1 gradient
    
    # Create field and install pattern
    field = torch.zeros(field_shape)
    field[:, :, :, :32] = 0.3  # Uniform content
    field[:, :, :, 32:48] = pattern
    
    print(f"Pattern gradient strength: {pattern[7, 0, 0, 0].item() - pattern[0, 0, 0, 0].item():.2f}")
    
    # Simulate pattern influence (from unified_field_brain._evolve_field)
    pattern_energy = field[:, :, :, 32:48].mean(dim=-1, keepdim=True)
    gradient_influence = torch.tanh(pattern_energy) * 0.02
    field[:, :, :, :32] += gradient_influence.expand(-1, -1, -1, 32)
    
    # Check resulting content gradient
    content = field[:, :, :, :32]
    content_grad = (content[-1, :, :].mean() - content[0, :, :].mean()).item()
    
    print(f"Content gradient after influence: {content_grad:.4f}")
    print(f"Amplified by 5x for motor: {content_grad * 5:.4f}")
    
    return content_grad


def test_full_chain():
    """Test the complete chain from low confidence to movement."""
    print("\n\nTesting full chain: Low confidence → Movement...")
    
    # Create brain with reduced field size for speed
    brain = UnifiedFieldBrain(
        sensory_dim=24,
        motor_dim=4,
        device='cpu',
        quiet_mode=True,
        field_shape=(8, 8, 8, 64)  # Smaller for testing
    )
    brain.enable_strategic_planning(True)
    
    # Set very low confidence
    brain._current_prediction_confidence = 0.05
    
    # Create low-energy field (high information tension too)
    brain.unified_field = torch.zeros(8, 8, 8, 64)
    brain.unified_field[:, :, :, :32] = 0.1  # Low energy
    brain.unified_field[:, :, :, 58] = 0.05  # Low confidence indicator
    
    # Process one cycle to trigger pattern discovery
    dummy_input = [0.5] * 24
    brain.brain_cycles = 20  # Trigger pattern discovery
    
    print("Processing to trigger pattern discovery...")
    motor1 = brain.process(dummy_input)
    
    # Check if pattern was discovered
    if brain.current_strategic_pattern:
        print(f"✓ Pattern discovered with score: {brain.current_strategic_pattern.score:.2f}")
    else:
        print("✗ No pattern discovered")
    
    # Process again to see movement
    motor2 = brain.process(dummy_input)
    
    print(f"\nMotor outputs:")
    print(f"  First: {motor1}")
    print(f"  Second: {motor2}")
    
    # Check movement magnitude
    mag1 = np.linalg.norm(motor1)
    mag2 = np.linalg.norm(motor2)
    print(f"\nMovement magnitudes:")
    print(f"  First: {mag1:.3f}")
    print(f"  Second: {mag2:.3f}")
    
    # Extract motor tendencies to see raw values
    tendencies = brain._extract_motor_tendencies_from_field()
    print(f"\nRaw motor tendencies: {tendencies}")
    
    # Check field gradients
    content = brain.unified_field[:, :, :, :32]
    x_grad = (content[-1, :, :].mean() - content[0, :, :].mean()).item()
    y_grad = (content[:, -1, :].mean() - content[:, 0, :].mean()).item()
    print(f"\nField gradients:")
    print(f"  X: {x_grad:.4f} → motor: {x_grad * 5:.4f}")
    print(f"  Y: {y_grad:.4f} → motor: {y_grad * 5:.4f}")


if __name__ == "__main__":
    # Test each part of the chain
    pattern, gradient = test_confidence_tension_creates_exploration_pattern()
    content_gradient = test_pattern_influence_on_content()
    test_full_chain()
    
    print("\n\n=== Summary ===")
    if gradient < 0.01:
        print("⚠️  Issue: Confidence patterns don't create sufficient gradients")
    if content_gradient < 0.001:
        print("⚠️  Issue: Pattern influence on content is too weak (0.02)")
    print("\nThe pattern → gradient → movement chain needs strengthening.")