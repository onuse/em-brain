#!/usr/bin/env python3
"""
Simple test for confidence-based sensory processing.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
from src.brain_loop import DecoupledBrainLoop
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.communication.sensor_buffer import get_sensor_buffer


def test_simple():
    """Quick test to verify basic functionality."""
    
    print("\n🧠 Testing Brain Loop with Confidence-Based Sensing")
    print("=" * 50)
    
    # Create brain
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': True
    })
    
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=17,
        motor_dim=4
    )
    
    # Check if we can access the actual brain
    print(f"\nBrain wrapper type: {type(brain_wrapper)}")
    print(f"Has .brain attribute: {hasattr(brain_wrapper, 'brain')}")
    if hasattr(brain_wrapper, 'brain'):
        print(f"Brain type: {type(brain_wrapper.brain)}")
        print(f"Has cognitive_autopilot: {hasattr(brain_wrapper.brain, 'cognitive_autopilot')}")
        print(f"Has process_robot_cycle: {hasattr(brain_wrapper.brain, 'process_robot_cycle')}")
    
    # Create brain loop
    brain_loop = DecoupledBrainLoop(brain_wrapper, cycle_time_ms=20)
    
    # Get sensor buffer
    sensor_buffer = get_sensor_buffer()
    
    try:
        # Start brain loop
        brain_loop.start()
        
        # Add some sensor data
        print("\nAdding sensor data...")
        for i in range(10):
            sensors = [0.5] * 16 + [0.0]
            sensor_buffer.add_sensor_input("test_robot", sensors)
            time.sleep(0.05)
        
        # Give it a moment to process
        time.sleep(0.5)
        
        # Check stats
        stats = brain_loop.get_loop_statistics()
        print(f"\nLoop Statistics:")
        print(f"  Total cycles: {stats['total_cycles']}")
        print(f"  Active cycles: {stats['active_cycles']}")
        print(f"  Sensor skip cycles: {stats['sensor_skip_cycles']}")
        print(f"  Current mode: {stats['current_cognitive_mode']}")
        print(f"  Avg confidence: {stats['avg_confidence']:.2f}")
        
    finally:
        brain_loop.stop()
        print("\n✅ Test completed")


if __name__ == "__main__":
    test_simple()