#!/usr/bin/env python3
"""
Test Confidence-Based Sensory Processing

Demonstrates how the brain decides whether to check sensors
based on prediction confidence and cognitive mode.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import threading
from src.brain_loop import DecoupledBrainLoop
from src.core.dynamic_brain_factory import DynamicBrainFactory
from src.communication.sensor_buffer import get_sensor_buffer


def test_confidence_based_sensing():
    """Test brain's autonomous decision-making about sensor processing."""
    
    print("\n🧠 Testing Confidence-Based Sensory Processing")
    print("=" * 60)
    print("The brain will decide when to check sensors based on confidence\n")
    
    # Create brain with full features
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
    
    # Create decoupled brain loop
    brain_loop = DecoupledBrainLoop(brain_wrapper, cycle_time_ms=20)
    
    # Get sensor buffer
    sensor_buffer = get_sensor_buffer()
    
    try:
        # Start brain loop
        brain_loop.start()
        
        print("📊 Phase 1: Low Confidence (New Environment)")
        print("-" * 60)
        print("Brain should check sensors frequently when uncertain\n")
        
        # Provide varied sensor input to create low confidence
        for i in range(50):
            # Random-ish sensor patterns
            sensors = [0.3 + 0.4 * ((i + j) % 3) / 2.0 for j in range(16)] + [0.0]
            sensor_buffer.add_sensor_input("test_robot", sensors)
            time.sleep(0.02)
        
        # Check stats
        time.sleep(0.5)
        stats = brain_loop.get_loop_statistics()
        print(f"Sensor check probability: {stats['sensor_check_probability']:.1%}")
        print(f"Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"Sensor skip cycles: {stats['sensor_skip_cycles']}")
        
        print("\n📊 Phase 2: Building Confidence (Stable Pattern)")
        print("-" * 60)
        print("Brain should check sensors less as it learns the pattern\n")
        
        # Provide stable sensor pattern
        stable_pattern = [0.5, 0.7, 0.3, 0.6] * 4 + [0.3]  # Reward
        
        for i in range(100):
            sensor_buffer.add_sensor_input("test_robot", stable_pattern)
            time.sleep(0.02)
        
        # Check stats
        time.sleep(0.5)
        stats = brain_loop.get_loop_statistics()
        print(f"Sensor check probability: {stats['sensor_check_probability']:.1%}")
        print(f"Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"Sensor skip cycles: {stats['sensor_skip_cycles']}")
        print(f"Cognitive mode: {stats['current_cognitive_mode']}")
        
        print("\n📊 Phase 3: High Confidence Autopilot")
        print("-" * 60)
        print("Brain should mostly ignore sensors and follow internal dynamics\n")
        
        # Continue stable pattern longer to build high confidence
        for i in range(200):
            sensor_buffer.add_sensor_input("test_robot", stable_pattern)
            time.sleep(0.02)
        
        # Check final stats
        time.sleep(0.5)
        stats = brain_loop.get_loop_statistics()
        print(f"Sensor check probability: {stats['sensor_check_probability']:.1%}")
        print(f"Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"Sensor skip cycles: {stats['sensor_skip_cycles']}")
        print(f"Total cycles: {stats['total_cycles']}")
        print(f"Cognitive mode: {stats['current_cognitive_mode']}")
        
        # Calculate skip rate
        skip_rate = stats['sensor_skip_cycles'] / max(1, stats['total_cycles'])
        print(f"\n🎯 Overall sensor skip rate: {skip_rate:.1%}")
        
        print("\n📊 Phase 4: Disruption Test")
        print("-" * 60)
        print("Changing pattern - brain should notice and increase sensor attention\n")
        
        # Suddenly change pattern
        new_pattern = [0.8, 0.2, 0.9, 0.1] * 4 + [0.8]  # Different + high reward
        
        for i in range(50):
            sensor_buffer.add_sensor_input("test_robot", new_pattern)
            time.sleep(0.02)
        
        # Check adaptation
        time.sleep(0.5)
        stats = brain_loop.get_loop_statistics()
        print(f"Sensor check probability: {stats['sensor_check_probability']:.1%}")
        print(f"Avg confidence: {stats['avg_confidence']:.2f}")
        print(f"Cognitive mode: {stats['current_cognitive_mode']}")
        
    finally:
        # Always stop brain loop
        brain_loop.stop()
        
        # Print final report
        print("\n")
        brain_loop.print_loop_report()
        
        print("\n✨ Key Insights:")
        print("-" * 60)
        print("1. Brain autonomously decides when to check sensors")
        print("2. High confidence → less sensor checking (but never zero)")
        print("3. Pattern changes → increased sensor attention")
        print("4. This creates natural attention cycles!")


if __name__ == "__main__":
    test_confidence_based_sensing()