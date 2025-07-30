#!/usr/bin/env python3
"""
Test Adapter Fix

Verifies that the simplified adapters work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.simplified_adapters import SimplifiedAdapterFactory, SimplifiedMotorAdapter
from src.core.interfaces import Robot, SensorChannel, MotorChannel
from src.brains.field.simplified_unified_brain import SimplifiedUnifiedBrain
import torch


def test_adapter_fix():
    """Test that adapters handle dimensions correctly."""
    print("Testing Adapter Fix")
    print("-" * 50)
    
    # Create test robot (matching biological_embodied_learning)
    robot = Robot(
        robot_id="test_robot",
        robot_type="generic_24s_4m",
        sensory_channels=[
            SensorChannel(
                index=i,
                name=f"sensor_{i}",
                range_min=0.0,
                range_max=1.0,
                unit="normalized",
                description=f"Sensor channel {i}"
            )
            for i in range(24)
        ],
        motor_channels=[
            MotorChannel(0, "motor_x", -1.0, 1.0, "velocity", "X-axis movement"),
            MotorChannel(1, "motor_y", -1.0, 1.0, "velocity", "Y-axis movement"), 
            MotorChannel(2, "motor_z", -1.0, 1.0, "velocity", "Z-axis movement"),
            MotorChannel(3, "confidence", 0.0, 1.0, "confidence", "Action confidence")
        ],
        capabilities={}
    )
    
    # Create simplified brain
    brain = SimplifiedUnifiedBrain(
        sensory_dim=24,
        motor_dim=4,
        quiet_mode=True,
        use_optimized=True
    )
    
    # Create adapter factory
    factory = SimplifiedAdapterFactory()
    motor_adapter = factory.create_motor_adapter(robot)
    
    print(f"Robot expects: {len(robot.motor_channels)} motors")
    print(f"Motor adapter expects: {motor_adapter.motor_dim} motors")
    
    # Test processing
    sensory_input = [0.5] * 24
    motors, state = brain.process_robot_cycle(sensory_input)
    
    print(f"\nBrain output: {len(motors)} motors")
    print(f"Motor values: {[f'{m:.2f}' for m in motors]}")
    
    # Test adapter conversion
    # Brain returns motor commands as list, wrap in tensor for adapter
    motor_tensor = torch.tensor(motors)
    
    try:
        adapted_motors = motor_adapter.from_field_space(motor_tensor)
        print(f"\nAdapter output: {len(adapted_motors)} motors")
        print(f"Adapted values: {[f'{m:.2f}' for m in adapted_motors]}")
        print("\n✅ Adapter fix successful!")
        
        # Verify dimensions
        assert len(adapted_motors) == 4, f"Expected 4 motors, got {len(adapted_motors)}"
        print("✅ Dimensions match robot expectations")
        
    except Exception as e:
        print(f"\n❌ Adapter error: {e}")
        return False
    
    return True


if __name__ == "__main__":
    success = test_adapter_fix()
    sys.exit(0 if success else 1)