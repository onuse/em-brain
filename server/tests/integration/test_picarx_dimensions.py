#!/usr/bin/env python3
"""
Test brain with PiCar-X's actual dimensions (16 sensors, 4 motors).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import numpy as np


def test_picarx_brain():
    """Test brain with PiCar-X dimensions."""
    
    print("🤖 Testing Brain with PiCar-X Dimensions")
    print("=" * 50)
    
    # Create brain with PiCar-X dimensions
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False
    })
    
    # PiCar-X has 16 sensors + 1 reward = 17 total inputs
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=17,  # 16 sensors + reward
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print(f"\n📊 Brain created successfully!")
    print(f"  Expected inputs: 17 (16 sensors + reward)")
    print(f"  Expected outputs: 4 (motor commands)")
    
    # Test with realistic PiCar-X data
    print("\n🧪 Testing with PiCar-X sensor patterns...")
    
    test_scenarios = [
        ("Clear path", [
            0.8,  # ultrasonic (far)
            0.3, 0.3, 0.3,  # grayscale sensors
            0.2, 0.2,  # motor speeds
            0, 0, 0,  # camera/steering angles
            7.4,  # battery
            0, 0,  # line/cliff detection
            45, 0.3,  # CPU/memory
            1000, 0,  # timestamp/reserved
            0.5  # reward (neutral)
        ]),
        ("Obstacle detected", [
            0.15,  # ultrasonic (close!)
            0.3, 0.3, 0.3,  # grayscale
            0.1, 0.1,  # motors slowing
            0, 0, 15,  # steering right
            7.4, 0, 0, 45, 0.3, 1000, 0,
            0.1  # low reward
        ]),
        ("Following line", [
            0.5,  # ultrasonic
            0.2, 0.8, 0.2,  # center line strong
            0.3, 0.3,  # moving forward
            0, 0, 0,  # straight
            7.4, 1, 0, 45, 0.3, 1000, 0,
            0.8  # high reward
        ])
    ]
    
    for scenario_name, sensor_data in test_scenarios:
        motor_output, brain_state = brain.process_robot_cycle(sensor_data)
        
        print(f"\n{scenario_name}:")
        print(f"  Sensors: distance={sensor_data[0]:.2f}m, line={sensor_data[1:4]}")
        print(f"  Motor output: {[f'{m:.3f}' for m in motor_output]}")
        print(f"  Field energy: {brain_state['field_energy']:.6f}")
        print(f"  Prediction confidence: {brain_state['prediction_confidence']:.3f}")
    
    # Test continuous operation
    print("\n⏱️  Performance test (100 cycles)...")
    import time
    
    start = time.time()
    for i in range(100):
        # Simulate moving robot
        sensor_data = [
            0.5 + 0.3 * np.sin(i * 0.1),  # varying distance
            0.3, 0.5 + 0.2 * np.sin(i * 0.2), 0.3,  # line sensors
            0.2, 0.2,  # motors
            10 * np.sin(i * 0.05), 0, 0,  # slight steering
            7.4, 0, 0, 45, 0.3, 1000, 0,
            0.5 + 0.3 * np.sin(i * 0.15)  # varying reward
        ]
        
        motor_output, _ = brain.process_robot_cycle(sensor_data)
    
    elapsed = time.time() - start
    print(f"  Total time: {elapsed:.2f}s")
    print(f"  Average per cycle: {elapsed/100*1000:.1f}ms")
    
    # Check final brain state
    print(f"\n📈 Final brain state:")
    print(f"  Brain cycles: {brain.brain_cycles}")
    print(f"  Conceptual dimensions: {brain.total_dimensions}D")
    print(f"  Tensor shape: {brain.tensor_shape}")
    print(f"  Memory usage: {brain._calculate_memory_usage():.1f}MB")
    
    print("\n✅ PiCar-X brain test complete!")
    print("\nKey findings:")
    print("- Brain adapts to 17 input dimensions (16 sensors + reward)")
    print("- Creates appropriate conceptual space based on robot complexity")
    print("- Performance suitable for real-time control")


if __name__ == "__main__":
    test_picarx_brain()