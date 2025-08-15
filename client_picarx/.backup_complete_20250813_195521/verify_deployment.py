#!/usr/bin/env python3
"""
Verify the simplified deployment is ready.

Checks that all necessary files exist and imports work.
"""

import sys
import os
from pathlib import Path

def check_files():
    """Check all required files exist."""
    
    required_files = [
        # Main entry
        "picarx_robot_simple.py",
        
        # Brainstem
        "src/brainstem/clean_brainstem.py",
        "src/brainstem/brain_client_simple.py",
        
        # Hardware
        "src/hardware/bare_metal_hal.py", 
        "src/hardware/picarx_hardware_limits.py",
        
        # Scripts
        "deploy_simple.sh",
        "cleanup_brainstem.sh",
    ]
    
    print("🔍 Checking required files...")
    all_exist = True
    
    for file in required_files:
        path = Path(file)
        if path.exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - MISSING!")
            all_exist = False
    
    return all_exist

def check_imports():
    """Check that all imports work."""
    
    print("\n🔍 Checking imports...")
    
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
    
    try:
        from brainstem.clean_brainstem import CleanBrainstem
        print("  ✅ clean_brainstem imports")
    except ImportError as e:
        print(f"  ❌ clean_brainstem import failed: {e}")
        return False
    
    try:
        from brainstem.brain_client_simple import BrainClient
        print("  ✅ brain_client_simple imports")
    except ImportError as e:
        print(f"  ❌ brain_client_simple import failed: {e}")
        return False
    
    try:
        from hardware.bare_metal_hal import BareMetalHAL, create_hal
        print("  ✅ bare_metal_hal imports")
    except ImportError as e:
        print(f"  ❌ bare_metal_hal import failed: {e}")
        return False
    
    try:
        from hardware.picarx_hardware_limits import (
            STEERING_MAX, CAMERA_PAN_MAX,
            steering_to_microseconds, camera_pan_to_microseconds
        )
        print("  ✅ picarx_hardware_limits imports")
    except ImportError as e:
        print(f"  ❌ picarx_hardware_limits import failed: {e}")
        return False
    
    return True

def check_unnecessary_files():
    """Check for files that should be removed."""
    
    unnecessary_files = [
        "src/brainstem/integrated_brainstem.py",
        "src/brainstem/integrated_brainstem_async.py",
        "src/brainstem/sensor_motor_adapter.py",
        "src/brainstem/sensor_motor_adapter_fixed.py",
        "src/brainstem/direct_brain_adapter.py",
        "src/brainstem/bare_metal_adapter.py",
        "src/brainstem/nuclei.py",
        "src/brainstem/event_bus.py",
    ]
    
    print("\n🔍 Checking for unnecessary files...")
    found_any = False
    
    for file in unnecessary_files:
        path = Path(file)
        if path.exists():
            print(f"  ⚠️  {file} - should be removed")
            found_any = True
    
    if not found_any:
        print("  ✅ No unnecessary files found")
    
    return not found_any

def main():
    """Run all verification checks."""
    
    print("=" * 60)
    print("🤖 PICAR-X DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    # Change to client_picarx directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    files_ok = check_files()
    imports_ok = check_imports()
    clean_ok = check_unnecessary_files()
    
    print("\n" + "=" * 60)
    print("📊 RESULTS:")
    print("=" * 60)
    
    if files_ok and imports_ok:
        print("✅ Deployment is ready!")
        
        if not clean_ok:
            print("\n⚠️  Run cleanup_brainstem.sh to remove unnecessary files:")
            print("   bash cleanup_brainstem.sh")
        
        print("\n📦 To deploy to Raspberry Pi:")
        print("   export PI_HOST=pi@192.168.1.xxx")
        print("   bash deploy_simple.sh")
        
        print("\n🚀 On the Pi, run:")
        print("   cd ~/picarx_robot")
        print("   ./test.sh          # Test hardware")
        print("   ./calibrate.sh     # Calibrate servos")
        print("   ./run.sh           # Run robot")
        
        return 0
    else:
        print("❌ Deployment not ready - fix issues above")
        return 1

if __name__ == "__main__":
    sys.exit(main())