#!/usr/bin/env python3
"""
Test individual components of the dynamic brain architecture.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.core.robot_registry import RobotRegistry
from src.core.brain_pool import BrainPool
from src.core.dynamic_brain_factory import DynamicBrainFactory
from pathlib import Path


def test_robot_registry():
    """Test robot registry component."""
    print("Testing Robot Registry...")
    
    registry = RobotRegistry()
    
    # Test PiCar-X registration
    capabilities = [1.0, 16.0, 5.0, 1.0, 3.0]
    robot = registry.register_robot(capabilities)
    
    print(f"  ✓ Registered robot: {robot.robot_type}")
    print(f"    - ID: {robot.robot_id}")
    print(f"    - Sensors: {len(robot.sensory_channels)}")
    print(f"    - Motors: {len(robot.motor_channels)}")
    print(f"    - Profile key: {robot.get_profile_key()}")


def test_brain_pool():
    """Test brain pool component."""
    print("\nTesting Brain Pool...")
    
    factory = DynamicBrainFactory({'quiet_mode': True})
    pool = BrainPool(factory)
    
    # Test brain creation for different profiles
    profiles = [
        "minimal_8s_2m",
        "picarx_16s_5m",
        "advanced_32s_8m"
    ]
    
    for profile in profiles:
        brain = pool.get_brain_for_profile(profile)
        print(f"  ✓ Created brain for {profile}: {brain.get_field_dimensions()}D")


def test_brain_factory():
    """Test brain factory component."""
    print("\nTesting Brain Factory...")
    
    factory = DynamicBrainFactory({'quiet_mode': True})
    
    # Test creating brains with different dimensions
    configs = [(16, 4), (24, 4), (32, 4)]
    
    for field_dims, spatial_res in configs:
        brain = factory.create(field_dims, spatial_res)
        print(f"  ✓ Created {field_dims}D brain with {spatial_res}³ resolution")


def main():
    """Run component tests."""
    print("🧪 Testing Dynamic Brain Architecture Components")
    print("=" * 50)
    
    try:
        test_robot_registry()
        test_brain_factory()
        test_brain_pool()
        
        print("\n✅ All component tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())