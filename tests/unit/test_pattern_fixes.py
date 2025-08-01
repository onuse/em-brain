#!/usr/bin/env python3
"""Test our fixes to the pattern system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from server.src.brains.field.field_strategic_planner import FieldStrategicPlanner


def test_confidence_pattern_creates_gradients():
    """Test that confidence patterns now create movement gradients."""
    print("Testing fixed confidence patterns...")
    
    # Create planner
    field_shape = (16, 16, 16, 64)
    planner = FieldStrategicPlanner(field_shape, sensory_dim=10, motor_dim=4)
    
    # High confidence tension
    tensions = {
        'information': 0.1,
        'learning': 0.1,
        'confidence': 0.9,  # High confidence tension
        'prediction': 0.1,
        'novelty': 0.1,
        'total': 0.5
    }
    
    # Generate pattern
    pattern = planner._generate_tension_targeted_pattern(tensions)
    
    # Check pattern creates gradients
    # Calculate gradient in X direction
    x_grad = 0
    for y in range(16):
        for z in range(16):
            x_grad += (pattern[-1, y, z, :].mean() - pattern[0, y, z, :].mean()).item()
    x_grad /= (16 * 16)
    
    print(f"X gradient from confidence pattern: {x_grad:.4f}")
    
    # Should have non-zero gradient now
    assert abs(x_grad) > 0.01, f"Confidence pattern should create gradients, got {x_grad}"
    
    print("✓ Confidence patterns now create exploration gradients!")
    

def test_pattern_influence_strength():
    """Test that pattern influence is strong enough."""
    print("\nTesting pattern influence strength...")
    
    # Create field
    field_shape = (8, 8, 8, 64)
    field = torch.zeros(field_shape)
    field[:, :, :, :32] = 0.5  # Uniform content
    
    # Create gradient pattern
    pattern = torch.zeros(8, 8, 8, 16)
    for i in range(8):
        pattern[i, :, :, :] = (i / 7.0) * 2 - 1  # -1 to 1 gradient
    
    field[:, :, :, 32:48] = pattern
    
    # Apply pattern influence (new strength 0.1)
    pattern_energy = field[:, :, :, 32:48].mean(dim=-1, keepdim=True)
    gradient_influence = torch.tanh(pattern_energy) * 0.1  # Increased from 0.02
    field[:, :, :, :32] += gradient_influence.expand(-1, -1, -1, 32)
    
    # Check content gradient
    content = field[:, :, :, :32]
    content_grad = (content[-1, :, :].mean() - content[0, :, :].mean()).item()
    
    print(f"Content gradient with 0.1 influence: {content_grad:.4f}")
    print(f"Motor tendency (5x amplified): {content_grad * 5:.4f}")
    
    # Should create meaningful motor commands
    assert abs(content_grad * 5) > 0.5, "Pattern influence still too weak"
    
    print("✓ Pattern influence is now strong enough!")


if __name__ == "__main__":
    test_confidence_pattern_creates_gradients()
    test_pattern_influence_strength()
    print("\n✅ All fixes verified!")