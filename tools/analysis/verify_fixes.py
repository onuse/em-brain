#!/usr/bin/env python3
"""
Verify the spatial resolution and sensory dimension fixes
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'server', 'src'))

from config.adaptive_config import create_adaptive_config
from brain_factory import BrainFactory


def verify_fixes():
    """Verify spatial resolution and sensory dimension fixes"""
    
    print("Verifying Configuration Fixes")
    print("=" * 60)
    
    # Test 1: Check adaptive configuration
    print("\n1. Testing adaptive configuration...")
    config = create_adaptive_config("settings_simple.json")
    
    spatial_res = config['brain'].get('field_spatial_resolution', 'NOT SET')
    print(f"\n   ✓ Spatial resolution: {spatial_res}³")
    
    if spatial_res == 4:
        print("   ✅ Spatial resolution correctly overridden to 4³")
    else:
        print(f"   ❌ ERROR: Spatial resolution is {spatial_res}, expected 4")
        return False
    
    # Test 2: Check brain creation
    print("\n2. Testing brain creation...")
    brain = BrainFactory(config=config, quiet_mode=True, enable_logging=False)
    
    print(f"   ✓ Expected sensory dimension: {brain.brain.expected_sensory_dim}")
    print(f"   ✓ Expected motor dimension: {brain.brain.expected_motor_dim}")
    
    if brain.brain.expected_sensory_dim == 24:
        print("   ✅ Sensory dimension correctly set to 24")
    else:
        print(f"   ❌ ERROR: Sensory dimension is {brain.brain.expected_sensory_dim}, expected 24")
        return False
    
    # Test 3: Test processing with 24D input
    print("\n3. Testing 24D sensory input processing...")
    try:
        sensory_input = [0.2] * 24  # 24D input
        action, state = brain.process_sensory_input(sensory_input)
        print("   ✅ Successfully processed 24D input")
    except Exception as e:
        print(f"   ❌ ERROR processing 24D input: {e}")
        return False
    
    brain.shutdown()
    
    print("\n" + "=" * 60)
    print("✅ ALL FIXES VERIFIED!")
    print("\nYou can now run the validation test without issues:")
    print("  Terminal 1: python3 brain_server.py")
    print("  Terminal 2: python3 tools/runners/validation_runner.py biological_embodied_learning --hours 0.5")
    
    return True


if __name__ == "__main__":
    verify_fixes()