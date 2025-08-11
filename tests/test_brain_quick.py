#!/usr/bin/env python3
"""Quick test of PureFieldBrain to see performance and behavior"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))

import torch
import time
from server.src.brains.field.pure_field_brain import create_pure_field_brain

def test_brain_performance():
    """Test brain performance and basic behavior"""
    print("🧠 Testing PureFieldBrain")
    print("=" * 60)
    
    # Create a medium-sized brain
    brain = create_pure_field_brain(
        input_dim=10,
        output_dim=4, 
        size='medium',
        aggressive=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"\nBrain configuration:")
    print(brain)
    print(f"Device: {brain.device}")
    print(f"Learning rate: {brain.learning_rate}")
    
    # Test performance
    print(f"\n⚡ Performance Test (100 cycles):")
    
    # Warmup
    for _ in range(10):
        sensory = torch.randn(10, device=brain.device)
        motor = brain(sensory)
    
    # Measure
    if brain.device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    for i in range(100):
        sensory = torch.randn(10, device=brain.device)
        motor = brain(sensory)
        
        # Add some reward occasionally
        if i % 10 == 0:
            brain.learn_from_prediction_error(
                actual=torch.randn(10, device=brain.device),
                predicted=torch.randn(10, device=brain.device)
            )
    
    if brain.device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    ms_per_cycle = (elapsed / 100) * 1000
    
    print(f"Average cycle time: {ms_per_cycle:.2f}ms")
    print(f"Frequency: {1000/ms_per_cycle:.1f} Hz")
    
    # Check metrics
    print(f"\n📊 Brain Metrics after 100 cycles:")
    metrics = brain.metrics
    for key, value in sorted(metrics.items()):
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    
    # Test emergence of behavior
    print(f"\n🎯 Testing Behavioral Emergence (200 cycles):")
    
    # Track motor activity
    motor_history = []
    
    for cycle in range(200):
        # Vary sensory input
        if cycle < 50:
            # Static environment
            sensory = torch.tensor([1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=brain.device)
        elif cycle < 100:
            # Moving obstacle
            position = (cycle - 50) / 50.0
            sensory = torch.tensor([position, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=brain.device)
        elif cycle < 150:
            # Random environment
            sensory = torch.randn(10, device=brain.device) * 0.5
        else:
            # Return to static
            sensory = torch.tensor([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0], device=brain.device)
        
        motor = brain(sensory)
        motor_history.append(motor.detach().cpu().numpy())
        
        # Print progress
        if (cycle + 1) % 50 == 0:
            motor_strength = motor.abs().mean().item()
            print(f"  Cycle {cycle + 1}: Motor strength = {motor_strength:.3f}")
    
    # Final metrics
    print(f"\n🏁 Final State:")
    print(f"  Brain cycles: {brain.cycle_count}")
    print(f"  Field energy: {brain._practical_metrics['field_energy']:.3f}")
    print(f"  Sensory resonance: {brain._practical_metrics['sensory_resonance']:.3f}")
    
    # Check if behavior emerged
    import numpy as np
    motor_array = np.array(motor_history)
    motor_variance = np.var(motor_array, axis=0)
    
    if np.any(motor_variance > 0.01):
        print(f"\n✅ Behavioral variation detected! (variance: {motor_variance.max():.3f})")
    else:
        print(f"\n⚠️  Limited behavioral variation (variance: {motor_variance.max():.3f})")
    
    return brain

if __name__ == "__main__":
    brain = test_brain_performance()
    print("\n🎉 Test complete!")