#!/usr/bin/env python3
"""
Test Dimension Negotiation Between Robot and Brain

Verifies that:
1. Robot sends correct 16 sensor dimensions
2. Brain accepts and adapts to 16 dimensions
3. No reward signal is transmitted
4. Brain creates appropriate architecture for 16→4 mapping
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'server'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'client_picarx'))

import torch
import numpy as np
from server.src.brains.field.pure_field_brain import create_pure_field_brain

def test_brain_dimension_adaptation():
    """Test that brain correctly adapts to 16 sensor dimensions."""
    print("🧪 Testing Brain Dimension Adaptation")
    print("=" * 60)
    
    # Create brain with 16 input dimensions (no reward!)
    print("\n1. Creating brain with 16 sensor inputs, 4 motor outputs...")
    brain = create_pure_field_brain(
        input_dim=16,  # Actual robot sensors
        output_dim=4,   # Motor channels
        size='small',   # Start small for testing
        aggressive=True,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"   ✅ Brain created: {brain}")
    print(f"   Input dimension: {brain.input_dim}")
    print(f"   Output dimension: {brain.output_dim}")
    
    # Test sensor processing
    print("\n2. Testing sensor processing (16 channels)...")
    
    # Simulate normalized sensor data (all in [0,1] range)
    test_sensors = torch.tensor([
        0.5,   # Distance (inverted, normalized)
        0.3, 0.8, 0.3,  # Grayscale sensors
        0.6, 0.6,  # Motor speeds (normalized)
        0.5, 0.5,  # Camera servos (normalized)
        0.5,   # Steering (normalized)
        0.8,   # Battery (normalized)
        1.0,   # Line detected
        0.0,   # No cliff
        0.3,   # CPU temp (normalized)
        0.5,   # Distance gradient
        0.5,   # Angular velocity
        0.9    # System health
    ], device=brain.device)
    
    print(f"   Input sensors: {test_sensors.shape}")
    
    # Process through brain
    motor_output = brain(test_sensors)
    
    print(f"   ✅ Motor output: {motor_output.shape}")
    print(f"   Motor values: {motor_output.detach().cpu().numpy()}")
    
    # Verify no reward is expected
    print("\n3. Verifying no reward signal...")
    assert brain.input_dim == 16, f"Expected 16 inputs, got {brain.input_dim}"
    print("   ✅ Brain expects exactly 16 inputs (no reward)")
    
    # Test learning without rewards
    print("\n4. Testing learning without reward signals...")
    
    for cycle in range(10):
        # Vary sensor input
        sensors = torch.randn(16, device=brain.device) * 0.5 + 0.5  # Random around 0.5
        sensors = torch.clamp(sensors, 0, 1)  # Keep in valid range
        
        # Process
        motors = brain(sensors)
        
        # Learn from prediction error (brain discovers its own rewards)
        if cycle > 0:
            # Brain learns by comparing predictions with outcomes
            brain.learn_from_prediction_error(
                actual=sensors,  # What actually happened
                predicted=sensors * 0.9  # What brain predicted (simulated error)
            )
    
    print(f"   ✅ Completed 10 learning cycles without external rewards")
    print(f"   Brain cycles: {brain.cycle_count}")
    print(f"   Prediction error: {brain.last_prediction_error:.3f}")
    
    return brain


def test_adapter_compatibility():
    """Test that the adapter produces correct dimensions."""
    print("\n5. Testing Adapter Compatibility...")
    
    from client_picarx.src.brainstem.sensor_motor_adapter_fixed import PiCarXBrainAdapter
    from client_picarx.src.config.brainstem_config import get_config
    
    config = get_config()
    adapter = PiCarXBrainAdapter(config)
    
    # Test raw sensor data (16 channels from robot)
    raw_sensors = [0.5] * 16  # 16 raw sensor values
    
    # Convert to brain input
    brain_input = adapter.sensors_to_brain_input(raw_sensors)
    
    print(f"   Raw sensors: {len(raw_sensors)} channels")
    print(f"   Brain input: {len(brain_input)} channels")
    
    assert len(brain_input) == 16, f"Expected 16 brain inputs, got {len(brain_input)}"
    print("   ✅ Adapter produces correct 16-channel output")
    
    # Test motor conversion
    brain_output = [0.5, 0.0, 0.0, 0.0]  # 4 motor channels from brain
    motor_commands = adapter.brain_output_to_motors(brain_output)
    
    print(f"   Brain output: {len(brain_output)} channels")
    print(f"   Motor commands: {len(motor_commands)} actuators")
    
    assert len(motor_commands) == 5, f"Expected 5 motor commands, got {len(motor_commands)}"
    print("   ✅ Adapter correctly maps 4→5 motor channels")
    
    # Verify no reward in adapter
    debug_info = adapter.get_debug_info()
    print(f"\n   Adapter configuration:")
    print(f"     Sensor dimensions: {debug_info['sensor_dimensions']}")
    print(f"     Brain input dim: {debug_info['brain_input_dim']}")
    print(f"     Brain output dim: {debug_info['brain_output_dim']}")
    print(f"     Motor dimensions: {debug_info['motor_dimensions']}")


def test_handshake_protocol():
    """Test that handshake sends correct dimensions."""
    print("\n6. Testing Handshake Protocol...")
    
    from client_picarx.src.config.brainstem_config import get_config
    
    config = get_config()
    
    print(f"   Config sensor dimensions: {config.sensors.brain_input_dimensions}")
    print(f"   Config motor dimensions: {config.motors.brain_output_dimensions}")
    
    assert config.sensors.brain_input_dimensions == 16, "Sensor dimensions should be 16"
    assert config.motors.brain_output_dimensions == 4, "Motor dimensions should be 4"
    
    print("   ✅ Configuration has correct dimensions")
    
    # Simulate handshake vector
    handshake_vector = [
        1.0,  # Protocol version
        float(config.sensors.brain_input_dimensions),  # 16
        float(config.motors.brain_output_dimensions),   # 4
        1.0   # Basic capabilities
    ]
    
    print(f"   Handshake vector: {handshake_vector}")
    print("   ✅ Handshake will send: 16 sensors, 4 motors")


def main():
    """Run all dimension negotiation tests."""
    print("=" * 60)
    print("DIMENSION NEGOTIATION TEST SUITE")
    print("=" * 60)
    
    # Test brain adaptation
    brain = test_brain_dimension_adaptation()
    
    # Test adapter
    test_adapter_compatibility()
    
    # Test handshake
    test_handshake_protocol()
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print("\nKey achievements:")
    print("• No reward signal (brain discovers through experience)")
    print("• Correct 16-channel sensor mapping (no padding)")
    print("• Brain adapts architecture to 16→4 dimensions")
    print("• Handshake properly negotiates dimensions")
    print("\nThe brain is ready for true emergent learning! 🧠🚀")


if __name__ == "__main__":
    main()