#!/usr/bin/env python3
"""
Simple test to debug dynamic brain issues.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.dynamic_brain_factory import DynamicBrainFactory
import time


def test_basic_cycle():
    """Test a single brain cycle."""
    
    print("Creating factory...")
    factory = DynamicBrainFactory({
        'use_dynamic_brain': True,
        'use_full_features': True,
        'quiet_mode': False  # Enable output to see what's happening
    })
    
    print("Creating brain...")
    brain_wrapper = factory.create(
        field_dimensions=None,
        spatial_resolution=4,
        sensory_dim=24,
        motor_dim=4
    )
    brain = brain_wrapper.brain
    
    print("Processing single cycle...")
    start = time.time()
    
    sensory_input = [0.5] * 24
    motor_output, brain_state = brain.process_robot_cycle(sensory_input)
    
    elapsed = time.time() - start
    
    print(f"Cycle completed in {elapsed*1000:.1f}ms")
    print(f"Motor output: {motor_output}")
    print(f"Brain state: {brain_state}")
    
    print("✅ Basic cycle working")


if __name__ == "__main__":
    test_basic_cycle()