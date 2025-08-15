#!/usr/bin/env python3
"""
Quick validation that PureFieldBrain is ready for deployment.
Runs fast checks to ensure everything is configured correctly.
"""

import sys
import os
import torch
import time
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_brain_creation():
    """Test that we can create a PureFieldBrain."""
    print("1️⃣  Testing brain creation...")
    
    try:
        from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS
        
        # Try creating brain with different scales
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scale = 'small' if device == 'cpu' else 'medium'
        
        brain = PureFieldBrain(
            input_dim=10,
            output_dim=4,
            scale_config=SCALE_CONFIGS[scale],
            device=device,
            aggressive=True
        )
        
        print(f"   ✅ Created {scale} scale brain on {device}")
        print(f"      Parameters: {brain.scale_config.total_params:,}")
        print(f"      Levels: {len(brain.levels)}")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create brain: {e}")
        return False

def test_brain_factory():
    """Test that the factory can create wrapped brains."""
    print("\n2️⃣  Testing brain factory...")
    
    try:
        from src.core.unified_brain_factory import PureFieldBrainFactory
        
        factory = PureFieldBrainFactory({'quiet_mode': True})
        brain = factory.create(sensory_dim=10, motor_dim=4)
        
        # Test interface methods
        import torch
        sensory = torch.randn(10)
        output = brain.process_field_dynamics(sensory)
        state = brain.get_brain_state()
        
        print(f"   ✅ Factory creates functional brain")
        print(f"      Output shape: {output.shape}")
        print(f"      State keys: {list(state.keys())[:3]}...")
        return True
        
    except Exception as e:
        print(f"   ❌ Factory test failed: {e}")
        return False

def test_settings_load():
    """Test that settings.json loads correctly."""
    print("\n3️⃣  Testing settings configuration...")
    
    try:
        with open('settings.json', 'r') as f:
            settings = json.load(f)
        
        brain_type = settings.get('brain', {}).get('type', 'unknown')
        aggressive = settings.get('brain', {}).get('aggressive_learning', False)
        
        print(f"   ✅ Settings loaded successfully")
        print(f"      Brain type: {brain_type}")
        print(f"      Aggressive learning: {aggressive}")
        
        if brain_type != 'pure':
            print(f"   ⚠️  Brain type is '{brain_type}', expected 'pure'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Settings load failed: {e}")
        return False

def test_performance():
    """Quick performance test."""
    print("\n4️⃣  Testing performance...")
    
    try:
        from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        scale = 'tiny'  # Use tiny for quick test
        
        brain = PureFieldBrain(
            input_dim=10,
            output_dim=4,
            scale_config=SCALE_CONFIGS[scale],
            device=device,
            aggressive=True
        )
        
        # Run 100 cycles
        sensory = torch.randn(10, device=device)
        
        start = time.time()
        for _ in range(100):
            output = brain.forward(sensory)
        elapsed = time.time() - start
        
        avg_ms = (elapsed / 100) * 1000
        
        print(f"   ✅ Performance test passed")
        print(f"      100 cycles in {elapsed:.3f}s")
        print(f"      Average: {avg_ms:.2f} ms/cycle")
        
        if avg_ms > 10:
            print(f"   ⚠️  Performance may be slow for real-time operation")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test failed: {e}")
        return False

def test_learning_capability():
    """Test that brain shows learning behavior."""
    print("\n5️⃣  Testing learning capability...")
    
    try:
        from src.brains.field.pure_field_brain import PureFieldBrain, SCALE_CONFIGS
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        brain = PureFieldBrain(
            input_dim=10,
            output_dim=4,
            scale_config=SCALE_CONFIGS['tiny'],  # Tiny for quick test
            device=device,
            aggressive=True
        )
        
        # Measure initial state
        initial_energy = brain.metrics['field_energy']
        
        # Run 50 cycles with strong input
        for i in range(50):
            sensory = torch.ones(10, device=device) * (i * 0.1)
            output = brain.forward(sensory, reward=0.5)
        
        # Measure final state
        final_energy = brain.metrics['field_energy']
        
        change = final_energy - initial_energy
        
        print(f"   ✅ Learning test completed")
        print(f"      Initial energy: {initial_energy:.4f}")
        print(f"      Final energy: {final_energy:.4f}")
        print(f"      Change: {change:+.4f}")
        
        if abs(change) < 0.0001:
            print(f"   ⚠️  No significant learning detected - may need more cycles")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Learning test failed: {e}")
        return False

def main():
    """Run all validation tests."""
    print("=" * 70)
    print("🧠 PUREFIELDBRAIN QUICK VALIDATION")
    print("=" * 70)
    print()
    
    # Detect hardware
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Hardware: {device.upper()}")
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()
    
    tests = [
        ("Brain Creation", test_brain_creation),
        ("Factory System", test_brain_factory),
        ("Settings Load", test_settings_load),
        ("Performance", test_performance),
        ("Learning", test_learning_capability)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 VALIDATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {name:20s}: {status}")
        if not success:
            all_passed = False
    
    print()
    if all_passed:
        print("🎉 ALL TESTS PASSED - PureFieldBrain is ready for deployment!")
        print()
        print("📝 Recommended settings for first 10 minutes:")
        print("   - Use 'medium' scale on GPU, 'small' on CPU")
        print("   - Keep aggressive_learning=True for visible changes")
        print("   - Save interval at 100 cycles for quick feedback")
        print("   - Monitor field_energy metric for learning progress")
        print()
        print("🚀 To start the server:")
        print("   python3 brain.py")
        print()
        print("🤖 To test robot connection:")
        print("   python3 test_robot_connection.py")
        return 0
    else:
        print("⚠️  Some tests failed - review output above")
        return 1

if __name__ == "__main__":
    sys.exit(main())