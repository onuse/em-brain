#!/usr/bin/env python3
"""
Installation Test Script

Run this script to verify that all requirements are properly installed
and the brain system is working correctly.

Usage:
    python3 test_installation.py
"""

import sys
import traceback

def test_core_dependencies():
    """Test that core brain dependencies are available."""
    print("🧠 Testing core brain dependencies...")
    
    try:
        import numpy as np
        print(f"   ✅ numpy {np.__version__} - Scientific computing")
    except ImportError as e:
        print(f"   ❌ numpy missing: {e}")
        return False
        
    try:
        import torch
        device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"   ✅ torch {torch.__version__} - GPU acceleration ({device})")
    except ImportError as e:
        print(f"   ❌ torch missing: {e}")
        return False
        
    try:
        import psutil
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)
        print(f"   ✅ psutil {psutil.__version__} - Hardware adaptation ({cpu_count} cores, {memory_gb:.1f}GB)")
    except ImportError as e:
        print(f"   ❌ psutil missing: {e}")
        return False
    
    return True

def test_demo_dependencies():
    """Test that demo dependencies are available."""
    print("\n🎮 Testing demo dependencies...")
    
    demo_deps = [
        ('pygame', '2D visualization demos'),
        ('pyglet', '3D OpenGL demos'), 
        ('matplotlib', 'Scientific plotting'),
        ('pytest', 'Testing framework')
    ]
    
    available = 0
    for package, description in demo_deps:
        try:
            if package == 'matplotlib':
                import matplotlib
                version = matplotlib.__version__
            elif package == 'pygame':
                import pygame
                version = pygame.version.ver
            elif package == 'pyglet':
                import pyglet
                version = pyglet.version
            elif package == 'pytest':
                import pytest
                version = pytest.__version__
            
            print(f"   ✅ {package} {version} - {description}")
            available += 1
        except ImportError:
            print(f"   ⚠️  {package} missing - {description} may not work")
    
    print(f"\n   📊 Demo dependencies: {available}/{len(demo_deps)} available")
    return available

def test_brain_functionality():
    """Test that the brain can be imported and initialized."""
    print("\n🧬 Testing brain functionality...")
    
    try:
        # Test brain import
        from src.brain import MinimalBrain
        print("   ✅ Brain import successful")
        
        # Test brain initialization
        brain = MinimalBrain(enable_logging=False, enable_persistence=False)
        print("   ✅ Brain initialization successful")
        
        # Test basic brain cycle
        sensory_input = [0.5, 0.3, 0.8, 0.2] * 4  # 16-dimensional input
        action, brain_state = brain.process_sensory_input(sensory_input)
        print(f"   ✅ Brain cycle successful - predicted {len(action)} action values")
        
        # Test experience storage
        outcome = [s + 0.1 for s in sensory_input]
        exp_id = brain.store_experience(sensory_input, action, outcome, action)
        print(f"   ✅ Experience storage successful - stored experience {exp_id[:8]}...")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Brain test failed: {e}")
        traceback.print_exc()
        return False

def test_hardware_adaptation():
    """Test that hardware adaptation is working."""
    print("\n🔧 Testing hardware adaptation...")
    
    try:
        from src.utils.hardware_adaptation import get_hardware_adaptation
        
        adapter = get_hardware_adaptation()
        profile = adapter.get_hardware_profile()
        limits = adapter.get_cognitive_limits()
        
        print(f"   ✅ Hardware discovered: {profile['cpu_cores']} cores, {profile['total_memory_gb']:.1f}GB")
        print(f"   ✅ Adaptive limits: WM={limits['working_memory_limit']}, Search={limits['similarity_search_limit']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Hardware adaptation test failed: {e}")
        return False

def main():
    """Run all installation tests."""
    print("🚀 Brain Installation Test")
    print("=" * 50)
    
    # Test dependencies
    core_ok = test_core_dependencies()
    demo_count = test_demo_dependencies()
    
    if not core_ok:
        print("\n❌ CORE DEPENDENCIES MISSING")
        print("   Install with: pip install numpy torch psutil")
        return False
    
    # Test brain functionality
    brain_ok = test_brain_functionality()
    hardware_ok = test_hardware_adaptation()
    
    print("\n" + "=" * 50)
    
    if brain_ok and hardware_ok:
        print("🎉 INSTALLATION TEST PASSED!")
        print(f"   ✅ Core brain system working")
        print(f"   ✅ Hardware adaptation active")
        print(f"   ✅ {demo_count}/4 demo dependencies available")
        print("\n🎯 You can now run:")
        print("   python3 demo_runner.py spatial  # Spatial learning demo")
        print("   python3 demo_runner.py brain    # Brain functionality demo")
        return True
    else:
        print("❌ INSTALLATION TEST FAILED")
        print("   Check error messages above for troubleshooting")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)